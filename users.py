import sqlite3
from werkzeug.security import generate_password_hash

# --- Configuration ---
USER_DB_PATH = 'users.db'
INITIAL_USERS = [
    ("6701","1234"),
    ("6702","1234"),
    ("6703","1234"),
    ("6704","1234"),
    ("6705","1234"),
    ("6706","1234"),
    ("6707","1234"),
    ("6708","1234"),
    ("6709","1234"),
    ("6710","1234"),
    ("6711","1234"),
    ("6712","1234"),
    ("6713","1234"),
    ("6714","1234"),
    ("6715","1234"),
    ("6716","1234"),
    ("6717","1234"),
    ("6718","1234"),
    ("6719","1234"),
    ("6720","1234"),
    ("6721","1234"),
    ("6722","1234"),
    ("6723","1234"),
    ("6724","1234"),
    ("6725","1234"),
    ("6726","1234"),
    ("6727","1234"),
    ("6728","1234"),
    ("6729","1234"),
    ("6730","1234"),
    ("6731","1234"),
    ("6732","1234"),
    ("6733","1234"),
    ("6734","1234"),
    ("6735","1234"),
    ("6736","1234"),
    ("6737","1234"),
    ("6738","1234"),
    ("6739","1234"),
    ("6740","1234"),
    ("6741","1234"),
    ("6742","1234"),
    ("6743","1234"),
    ("6744","1234"),
    ("6745","1234"),
    ("6746","1234"),
    ("6747","1234"),
    ("6748","1234"),
    ("6749","1234"),
    ("6750","1234"),
    ("6751","1234"),
    ("6752","1234"),
    ("6753","1234"),
    ("6754","1234"),
    ("6755","1234"),
    ("6756","1234"),
    ("6757","1234"),
    ("6758","1234"),
    ("6759","1234"),
    ("6760","1234"),
    ("6761","1234"),
    ("6762","1234"),
    ("6763","1234"),
    ("6764","1234"),
    ("admin", "prodpass"),
    # Add more predefined testers here (e.g., ("tester2", "testpass"))
]

def init_user_db():
    """Creates the user database and populates it with hashed passwords."""
    conn = sqlite3.connect(USER_DB_PATH)
    cursor = conn.cursor()
    
    # Create Users Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    
    # Populate Users
    for username, raw_password in INITIAL_USERS:
        password_hash = generate_password_hash(raw_password)
        try:
            cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                           (username, password_hash))
            print(f"User added: {username}")
        except sqlite3.IntegrityError:
            pass # User already exists
            
    conn.commit()
    conn.close()

if __name__ == '__main__':
    print(f"Initializing user database at {USER_DB_PATH}...")
    init_user_db()
    print("User setup complete. You can now run 'python app.py'.")