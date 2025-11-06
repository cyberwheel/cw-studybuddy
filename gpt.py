# ======================================================================
# gpt.py — Study Buddy (Optimized Production Version)
# Lightweight AI Tutor for Students — Q&A + RAG (Render Free Tier Ready)
# ======================================================================

import os, csv, sqlite3, logging, json, time, subprocess, sys, threading, queue, datetime, re, uuid
from pathlib import Path
from functools import wraps, lru_cache
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import check_password_hash
import numpy as np

# Markdown + sanitization
import markdown as md
import bleach
from bleach.css_sanitizer import CSSSanitizer

# ======================================================================
# CONFIGURATION
# ======================================================================
USER_DB_PATH = 'users.db'
LOG_DB_PATH  = 'interaction.db'
MODEL_NAME   = 'gemma:2b'
SECRET_KEY   = os.environ.get('STUDY_BUDDY_SECRET', 'super_secure_secret')

NOTES_FOLDER = Path("My_Notes")
QA_FILE_PATH = NOTES_FOLDER / "qa_pairs.csv"
DB_PERSIST_DIRECTORY = 'db'
STATIC_DIR   = Path("static")

LOG_FILE     = 'activity.log'

# Logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('StudyBuddy')
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console)

# Flask setup
app = Flask(__name__)
app.secret_key = SECRET_KEY

notification_queue = queue.Queue()
QA_CACHE = {"questions": [], "answers": [], "embeds": None}

# ======================================================================
# LAZY MODEL LOADING
# ======================================================================
@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    logger.info("🔹 Loading SentenceTransformer on demand...")
    return SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache(maxsize=1)
def get_chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=DB_PERSIST_DIRECTORY)
    return client.get_or_create_collection("course_notes")

@lru_cache(maxsize=1)
def get_ollama():
    import ollama
    return ollama

# ======================================================================
# DATABASE INITIALIZATION
# ======================================================================
def init_log_db():
    conn = sqlite3.connect(LOG_DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY,
        username TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        question TEXT,
        answer TEXT,
        confidence REAL,
        success INTEGER,
        latency REAL
    );
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        history_json TEXT
    );
    """)
    conn.commit()
    conn.close()

init_log_db()

# ======================================================================
# AUTH HELPERS
# ======================================================================
def check_user_credentials(username, password):
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        conn.close()
        return check_password_hash(row[0], password) if row else False
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return False

def requires_auth(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*a, **kw)
    return wrapper

# ======================================================================
# MARKDOWN SANITIZER
# ======================================================================
def sanitize_md_to_html(text: str) -> str:
    html = md.markdown(text, extensions=['fenced_code', 'tables'])
    allowed = set(bleach.sanitizer.ALLOWED_TAGS) | {
        'p','pre','code','strong','em','ul','ol','li',
        'table','tr','td','th','h1','h2','h3','blockquote'
    }
    css = CSSSanitizer(allowed_css_properties=[
        'color','background-color','font-weight','text-align'
    ])
    clean = bleach.clean(html, tags=list(allowed), css_sanitizer=css, strip=True)
    return f"<div style='background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:12px;line-height:1.6'>{clean}</div>"

# ======================================================================
# Q&A / RAG HELPERS
# ======================================================================
def reload_qa_cache():
    global QA_CACHE
    qs, ans = [], []
    if QA_FILE_PATH.exists():
        with open(QA_FILE_PATH, 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 2:
                    qs.append(row[0].strip())
                    ans.append(row[1].strip())
    if qs:
        model = get_embedding_model()
        QA_CACHE["questions"], QA_CACHE["answers"], QA_CACHE["embeds"] = qs, ans, model.encode(qs)
        logger.info(f"Loaded {len(qs)} Q&A pairs into cache.")
    else:
        logger.info("⚠️ No Q&A data found.")

def qa_match(question):
    if not QA_CACHE["questions"]:
        reload_qa_cache()
    if not QA_CACHE["embeds"] is not None:
        return None
    q_emb = get_embedding_model().encode([question])[0]
    C = QA_CACHE["embeds"]
    if C is None or len(C) == 0:
        return None
    qn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    sims = Cn @ qn
    idx = int(np.argmax(sims))
    if sims[idx] > 0.6:
        return QA_CACHE["answers"][idx], float(sims[idx])
    return None

def retrieve_context(question):
    try:
        coll = get_chroma_collection()
        model = get_embedding_model()
        q_emb = model.encode(question)
        results = coll.query(query_embeddings=[q_emb.tolist()], n_results=6, include=['documents','metadatas'])
        docs = results.get('documents', [[]])[0]
        metas = results.get('metadatas', [[]])[0]
        return docs[:4], metas[:4]
    except Exception as e:
        logger.warning(f"RAG failed: {e}")
        return [], []

# ======================================================================
# SSE NOTIFICATIONS
# ======================================================================
@app.route("/events")
def stream_events():
    def gen():
        while True:
            yield f"data: {notification_queue.get()}\n\n"
    return app.response_class(gen(), mimetype="text/event-stream")

def broadcast(msg): notification_queue.put(msg)

# ======================================================================
# AUTH ROUTES
# ======================================================================
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        if check_user_credentials(u, p):
            session["username"] = u
            return redirect("/admin" if u == "admin" else "/")
        return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ======================================================================
# PAGE ROUTES
# ======================================================================
@app.route("/")
@requires_auth
def student_page():
    return render_template("student.html")

@app.route("/admin")
@requires_auth
def admin_page():
    if session["username"] != "admin":
        return redirect("/")
    return render_template("admin.html")

# ======================================================================
# ASK ENDPOINT — CORE RAG LOGIC
# ======================================================================
@app.route("/ask", methods=["POST"])
@requires_auth
def ask_question():
    data = request.get_json()
    question = data.get("new_question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a question."}), 200

    start = time.time()
    response = {"answer": "", "sources": []}
    try:
        qa = qa_match(question)
        if qa:
            ans, score = qa
            html = sanitize_md_to_html(f"**From Notes/Q&A:** {ans}")
            response["answer"] = html
            response["sources"] = ["qa_pairs.csv"]
        else:
            docs, metas = retrieve_context(question)
            ctx = "\n\n---\n\n".join(docs) if docs else ""
            ollama = get_ollama()
            msgs = [
                {"role": "system", "content": "You are Study Buddy — a strict syllabus-aligned AI tutor. Only use notes provided. If you take anything from online, clearly say so."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{question}"}
            ]
            resp = ollama.chat(model=MODEL_NAME, messages=msgs)
            ans = resp["message"]["content"]
            html = sanitize_md_to_html(ans)
            response["answer"] = html
            response["sources"] = [m.get("source", "notes") for m in metas if m]
    except Exception as e:
        logger.error(f"/ask error: {e}")
        response["answer"] = f"<p>Error: {e}</p>"

    latency = round(time.time() - start, 2)
    conn = sqlite3.connect(LOG_DB_PATH)
    cur = conn.cursor()
    cur.execute("""INSERT INTO interactions (username, question, answer, confidence, success, latency)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session.get('username'), question, response["answer"][:500], 1.0, 1, latency))
    conn.commit(); conn.close()

    return jsonify(response), 200

# ======================================================================
# HEALTH CHECK
# ======================================================================
@app.route("/ping")
def ping():
    return {"status": "ok", "message": "Study Buddy is running!"}, 200

# ======================================================================
# RUN SERVER
# ======================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
