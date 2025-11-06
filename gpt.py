# ======================================================================
# app.py — Study Buddy (Corporate-Grade AI, Mixed Strict Mode)
# RAG + Q&A + reranker + transparent LLM fallback (with caution banner)
# ======================================================================

import os, csv, sqlite3, logging, json, time, subprocess, sys, threading, queue, datetime
from pathlib import Path
from functools import wraps
from typing import List, Tuple, Optional

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import check_password_hash

# Vector DB + embeddings + LLM
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

# Markdown → safe HTML
import markdown as md
import bleach
from bleach.css_sanitizer import CSSSanitizer

# ======================================================================
# CONFIG
# ======================================================================
USER_DB_PATH = 'users.db'
LOG_DB_PATH  = 'interaction.db'
MODEL_NAME   = 'gemma:2b'
SECRET_KEY   = os.environ.get('STUDY_BUDDY_SECRET', 'super_secure_secret')

NOTES_FOLDER = Path("My_Notes")
QA_FILE_PATH = NOTES_FOLDER / "qa_pairs.csv"
DB_PERSIST_DIRECTORY = 'db'
LOG_FILE     = 'activity.log'

# Thresholds
CONF_QA  = 0.60    # fuzzy Q&A acceptance
CONF_RAG = 0.45    # RAG chunk similarity acceptance

# Logging
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('StudyBuddy')
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console)

# Flask
app = Flask(__name__)
app.secret_key = SECRET_KEY

# ======================================================================
# GLOBALS
# ======================================================================
AI_TOOLS_LOADED = False
embedding_model: Optional[SentenceTransformer] = None
collection = None                 # Chroma collection "course_notes"
notification_queue = queue.Queue()
QA_CACHE = {"questions": [], "answers": [], "embeds": None}  # numpy matrix

# ======================================================================
# DB INIT
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
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY,
        username TEXT,
        question TEXT,
        answer TEXT,
        feedback INTEGER,
        corrected_answer TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS user_profiles (
        username TEXT PRIMARY KEY,
        memory TEXT,
        tone_preference TEXT DEFAULT 'detailed',
        study_focus TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit(); conn.close()
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
        logger.error(f"User DB error: {e}")
        return False

def requires_auth(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*a, **kw)
    return wrapper

# ======================================================================
# USER PROFILE + MEMORY
# ======================================================================
def get_user_profile(username):
    conn = sqlite3.connect(LOG_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT memory, tone_preference, study_focus FROM user_profiles WHERE username=?", (username,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return {"memory": "", "tone": "detailed", "focus": ""}
    return {"memory": r[0] or "", "tone": r[1] or "detailed", "focus": r[2] or ""}

def set_user_tone(username, tone):
    conn = sqlite3.connect(LOG_DB_PATH)
    cur = conn.cursor()
    cur.execute("""INSERT INTO user_profiles (username, tone_preference)
                   VALUES (?, ?)
                   ON CONFLICT(username) DO UPDATE SET
                   tone_preference=?, updated_at=CURRENT_TIMESTAMP""",
                (username, tone, tone))
    conn.commit(); conn.close()

def update_user_memory(username, note):
    if not note: return
    conn = sqlite3.connect(LOG_DB_PATH)
    cur = conn.cursor()
    cur.execute("""INSERT INTO user_profiles (username, memory)
                   VALUES (?, ?)
                   ON CONFLICT(username) DO UPDATE SET
                   memory = COALESCE(memory,'') || '\n' || excluded.memory,
                   updated_at = CURRENT_TIMESTAMP""",
                (username, note[:500]))
    conn.commit(); conn.close()

def save_chat_session(username, chat_history):
    conn = sqlite3.connect(LOG_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO chat_sessions (username, history_json) VALUES (?, ?)",
                (username, json.dumps(chat_history)))
    conn.commit(); conn.close()

# ======================================================================
# SANITIZER (fixed: frozenset + CSS)
# ======================================================================
def sanitize_md_to_html(text: str) -> str:
    html = md.markdown(text, extensions=['fenced_code', 'tables'])
    allowed = set(bleach.sanitizer.ALLOWED_TAGS) | {
        'p','pre','code','h1','h2','h3','h4','h5','h6',
        'ul','ol','li','strong','em','hr','blockquote',
        'table','thead','tbody','tr','td','th',
        'img','span','br','a'
    }
    attrs = {
        'a':    ['href','title','rel','target'],
        'img':  ['src','alt','title','style'],
        'span': ['class','style'],
        'code': ['class'],
    }
    css = CSSSanitizer(allowed_css_properties=[
        'color','background-color','font-weight','text-align',
        'border','border-radius','padding','margin','max-width','width'
    ])
    clean = bleach.clean(html, tags=list(allowed), attributes=attrs, css_sanitizer=css, strip=True)
    return f"<div style='background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:14px;line-height:1.65'>{clean}</div>"

# ======================================================================
# AI LOADERS
# ======================================================================
def reload_qa_cache():
    global QA_CACHE, embedding_model
    qs, ans = [], []
    if QA_FILE_PATH.exists():
        with open(QA_FILE_PATH, 'r', encoding='utf-8') as f:
            r = csv.reader(f); next(r, None)
            for row in r:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    qs.append(row[0].strip()); ans.append(row[1].strip())
    QA_CACHE["questions"] = qs
    QA_CACHE["answers"]   = ans
    QA_CACHE["embeds"]    = embedding_model.encode(qs) if qs else None
    logger.info(f"Q&A cache loaded: {len(qs)} entries.")

def load_models():
    global AI_TOOLS_LOADED, embedding_model, collection
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=DB_PERSIST_DIRECTORY)
        collection = client.get_or_create_collection("course_notes")
        reload_qa_cache()
        try:
            ollama.chat(model=MODEL_NAME, messages=[{'role':'user','content':'hello'}])
        except Exception as e:
            logger.warning(f"Ollama warmup issue: {e}")
        AI_TOOLS_LOADED = True
        logger.info("✅ AI models loaded successfully.")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
threading.Thread(target=load_models, daemon=True).start()

# ======================================================================
# RAG HELPERS
# ======================================================================
def embed_texts(texts: List[str]) -> np.ndarray:
    return embedding_model.encode(texts, convert_to_numpy=True)

def rerank_by_cosine(query_emb: np.ndarray, docs: List[str]) -> Tuple[List[str], List[float], List[int]]:
    if not docs: return [], [], []
    doc_embs = embed_texts(docs)
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    d = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-9)
    sims = (d @ q)
    order = np.argsort(-sims)
    return [docs[i] for i in order], sims[order].tolist(), order.tolist()

def retrieve_context(question: str, top_k=8) -> Tuple[List[str], List[dict], float]:
    if not collection: return [], [], 0.0
    q_emb = embedding_model.encode(question)
    results = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=['documents','metadatas']
    )
    docs  = results.get('documents',[[]])[0]
    metas = results.get('metadatas',[[]])[0]
    if not docs: return [], [], 0.0
    ranked, sims, order = rerank_by_cosine(q_emb, docs)
    top_docs  = ranked[:4]
    top_metas = [metas[i] for i in order[:4]]
    conf = float(max(sims)) if sims else 0.0
    return top_docs, top_metas, conf

def qa_match(question: str) -> Optional[Tuple[str, float]]:
    ql = question.lower()
    # exact-ish substring first
    for q, a in zip(QA_CACHE["questions"], QA_CACHE["answers"]):
        if q.lower() in ql or ql in q.lower():
            return (a, 1.0)
    # fuzzy embeddings
    if QA_CACHE["embeds"] is not None:
        q_emb = embedding_model.encode([question])[0]
        qn = q_emb / (np.linalg.norm(q_emb)+1e-9)
        C = QA_CACHE["embeds"]
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True)+1e-9)
        sims = Cn @ qn
        idx  = int(np.argmax(sims))
        if sims[idx] >= CONF_QA:
            return (QA_CACHE["answers"][idx], float(sims[idx]))
    return None

# ======================================================================
# SSE (notifications)
# ======================================================================
@app.route("/events")
def stream_events():
    def gen():
        while True:
            msg = notification_queue.get()
            yield f"data: {msg}\n\n"
    return app.response_class(gen(), mimetype="text/event-stream")

def broadcast(msg: str):
    notification_queue.put(msg)

# ======================================================================
# AUTH ROUTES
# ======================================================================
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        u, p = request.form["username"], request.form["password"]
        if check_user_credentials(u, p):
            session["username"] = u
            # restore previous chat
            conn = sqlite3.connect(LOG_DB_PATH); cur = conn.cursor()
            cur.execute("SELECT history_json FROM chat_sessions WHERE username=?", (u,))
            rec = cur.fetchone(); conn.close()
            session['chat_history_json'] = rec[0] if rec and rec[0] else '[]'
            return redirect("/admin" if u=="admin" else "/")
        return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ======================================================================
# PAGES
# ======================================================================
@app.route("/")
@requires_auth
def student_page():
    # If you want to pre-render history, pass initial_history to student.html
    hist = session.pop('chat_history_json','[]')
    return render_template("student.html", initial_history=hist)

@app.route("/admin")
@requires_auth
def admin_page():
    if session["username"] != "admin": return redirect("/")
    return render_template("admin.html")

# ======================================================================
# CORE CHAT — Mixed Strict Mode
# ======================================================================
SYSTEM_PROMPT = """
You are "Study Buddy" — a professional, empathetic AI tutor for B.Tech students.
You must prefer verified notes & Q&A context. Do not invent citations.

Formatting (MANDATORY):
- Use bold section headings and bullet/numbered lists where useful.
- Use fenced code blocks for code/math snippets if any appear in context.
- Keep tone helpful, precise, and concise.

If the provided CONTEXT is insufficient, do NOT fabricate citations.
"""

CAUTION_PREFIX = (
    "<div style='background:#1f2937;border:1px solid #374151;color:#e5e7eb;"
    "padding:10px;border-radius:10px;margin-bottom:8px;font-size:14px'>"
    "⚠️ <strong>Note:</strong> This part is <em>not found</em> in your uploaded materials. "
    "The explanation below uses general academic knowledge for clarity."
    "</div>"
)

@app.route("/ask", methods=["POST"])
@requires_auth
def ask_question():
    start = time.time()
    data = request.get_json()
    question = (data.get("new_question","") or "").strip()
    chat_history = data.get("history",[])

    if not question:
        return jsonify({"answer":"<p>Please enter a question.</p>","sources":[]}),200
    if not AI_TOOLS_LOADED:
        return jsonify({"answer":"<p><strong>Model initializing…</strong> Try again in a moment.</p>","sources":[]}),200

    username = session.get('username','guest')
    profile = get_user_profile(username)
    tone    = profile['tone']
    memory  = profile['memory'] or ""
    response_payload = {"answer":"", "sources":[]}
    success=0; conf=0.0

    try:
        # 0) Priority Q&A (exact/fuzzy)
        qa = qa_match(question)
        if qa:
            ans, score = qa
            html = sanitize_md_to_html(
                f"**Answer (Verified Q&A):**\n\n{ans}\n\n**Sources:**\n- qa_pairs.csv"
            )
            response_payload.update({"answer":html,"sources":["qa_pairs.csv"]})
            success=1; conf=float(score)
            final_hist = chat_history + [{'role':'user','content':question},{'role':'assistant','content':ans}]
            save_chat_session(username, final_hist)
            return jsonify(response_payload),200

        # 1) RAG retrieval from Chroma
        top_docs, top_metas, conf = retrieve_context(question)
        sources = list({m.get('source','Unknown') for m in top_metas}) if top_metas else []

        if top_docs and conf >= CONF_RAG:
            ctx = "\n\n---\n\n".join(top_docs)
            msgs = [
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":
                    f"Use ONLY the following CONTEXT to answer the QUESTION.\n\n"
                    f"CONTEXT:\n{ctx}\n\n"
                    f"QUESTION:\n{question}\n\n"
                    f"Tone: {tone}. Student memory (hint only): {memory[:600]}"}
            ]
            resp = ollama.chat(model=MODEL_NAME, messages=msgs)
            raw  = resp["message"]["content"]
            # Append real sources (from notes)
            if sources:
                raw = raw + "\n\n**Sources:**\n- " + "\n- ".join(sources)
            html = sanitize_md_to_html(raw)
            response_payload.update({"answer":html, "sources":sources})
            success=1
            final_hist = chat_history + [{'role':'user','content':question},{'role':'assistant','content':raw}]
            save_chat_session(username, final_hist)
            return jsonify(response_payload),200

        # 2) Fallback with precaution banner (no reliable context)
        msgs = [
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":
                f"Provide a short, syllabus-aligned explanation.\n\nQ: {question}\n\n"
                f"Tone: {tone}. Memory hint: {memory[:600]}"}
        ]
        resp = ollama.chat(model=MODEL_NAME, messages=msgs)
        raw  = resp["message"]["content"]
        html = CAUTION_PREFIX + sanitize_md_to_html(raw)
        response_payload.update({"answer":html, "sources":["(general knowledge)"]})
        success=1
        final_hist = chat_history + [{'role':'user','content':question},{'role':'assistant','content':raw}]
        save_chat_session(username, final_hist)
        return jsonify(response_payload),200

    except Exception as e:
        logger.error(f"/ask error: {e}")
        response_payload['answer'] = sanitize_md_to_html(f"**Unexpected error:** {e}")
        return jsonify(response_payload),200

    finally:
        try:
            latency = time.time()-start
            conn = sqlite3.connect(LOG_DB_PATH); cur = conn.cursor()
            cur.execute("""INSERT INTO interactions (username, question, answer, confidence, success, latency)
                           VALUES (?,?,?,?,?,?)""",
                        (username, question, response_payload['answer'][:500] if response_payload['answer'] else "",
                         float(conf), int(success), float(latency)))
            conn.commit(); conn.close()
        except Exception as le:
            logger.error(f"log fail: {le}")

# ======================================================================
# ADMIN ACTIONS
# ======================================================================
def start_indexer():
    try:
        subprocess.Popen([sys.executable, "indexer.py"])
        logger.info("Indexer launched.")
    except Exception as e:
        logger.error(f"Indexer error: {e}")

@app.route("/admin-upload", methods=["POST"])
@requires_auth
def handle_upload():
    if session["username"] != "admin": return redirect("/")
    if not NOTES_FOLDER.exists(): NOTES_FOLDER.mkdir(parents=True, exist_ok=True)
    files = request.files.getlist("files")
    n=0
    for f in files:
        if f and f.filename:
            f.save(NOTES_FOLDER / f.filename); n+=1
    broadcast("📘 New materials uploaded! Re-indexing.")
    start_indexer()
    return render_template("admin.html", message=f"✅ Uploaded {n} file(s). Re-indexing started.")

@app.route("/admin-qa", methods=["POST"])
@requires_auth
def add_qa():
    if session["username"] != "admin": return redirect("/")
    q = (request.form.get("question","") or "").strip()
    a = (request.form.get("answer","") or "").strip()
    if q and a:
        if not NOTES_FOLDER.exists(): NOTES_FOLDER.mkdir(parents=True, exist_ok=True)
        new = not QA_FILE_PATH.exists()
        with open(QA_FILE_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new: w.writerow(["question","answer"])
            w.writerow([q,a])
        reload_qa_cache()
        broadcast("💡 New Q&A added — AI cache refreshed.")
        # Optional: also index a mirror into the DB
        start_indexer()
        return render_template("admin.html", message="✅ Q&A added successfully!")
    return render_template("admin.html", message="❌ Please fill both fields.")

# ======================================================================
# ANALYTICS
# ======================================================================
@app.route("/admin/analytics")
@requires_auth
def analytics_dashboard():
    if session["username"] != "admin": return redirect("/")
    return render_template("admin-analytics.html", now=datetime.datetime.now())

@app.route("/admin/analytics-data")
@requires_auth
def analytics_data():
    if session["username"] != "admin": return jsonify({})
    conn = sqlite3.connect(LOG_DB_PATH); cur = conn.cursor()
    cur.execute("""SELECT date(timestamp), COUNT(*), AVG(confidence)
                   FROM interactions GROUP BY date(timestamp) ORDER BY 1""")
    daily = cur.fetchall()
    cur.execute("""SELECT username, COUNT(*) 
                   FROM interactions GROUP BY username ORDER BY 2 DESC LIMIT 10""")
    users = cur.fetchall()
    conn.close()
    return jsonify({"daily": daily, "users": users})

# ======================================================================
# DAILY MEMORY SUMMARIZER (optional)
# ======================================================================
def summarize_memories_daily():
    while True:
        now = datetime.datetime.now()
        nxt = (now + datetime.timedelta(days=1)).replace(hour=2, minute=0, second=0, microsecond=0)
        time.sleep(max(1, (nxt-now).total_seconds()))
        try:
            conn = sqlite3.connect(LOG_DB_PATH); cur = conn.cursor()
            cur.execute("SELECT username, memory FROM user_profiles WHERE memory IS NOT NULL")
            rows = cur.fetchall(); conn.close()
            for u, mem in rows:
                if not mem or not mem.strip(): continue
                try:
                    resp = ollama.chat(model=MODEL_NAME, messages=[
                        {"role":"system","content":"Summarize difficulties and focus areas in 2–3 concise bullets."},
                        {"role":"user","content":mem[:3000]}
                    ])
                    summary = resp["message"]["content"]
                    conn = sqlite3.connect(LOG_DB_PATH); c = conn.cursor()
                    c.execute("UPDATE user_profiles SET study_focus=?, updated_at=CURRENT_TIMESTAMP WHERE username=?",
                              (summary, u))
                    conn.commit(); conn.close()
                except Exception as e:
                    logger.error(f"Summarizer failed for {u}: {e}")
        except Exception as e:
            logger.error(f"Summarizer loop error: {e}")

threading.Thread(target=summarize_memories_daily, daemon=True).start()

# ======================================================================
# RUN (Windows-safe: no reloader to avoid WinError 10038)
# ======================================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)



