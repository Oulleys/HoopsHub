import sqlite3

def check_tables():
    conn = sqlite3.connect("hoopshub_users.db")
    c = conn.cursor()

    # List all tables
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = c.fetchall()
    print("Tables:", tables)

    conn.close()

def check_users():
    conn = sqlite3.connect("hoopshub_users.db")
    c = conn.cursor()

    # Query data from users table
    c.execute("SELECT * FROM users;")
    users = c.fetchall()
    print("Users:", users)

    conn.close()

def check_bet_slips():
    conn = sqlite3.connect("hoopshub_users.db")
    c = conn.cursor()

    # Query data from bet_slips table
    c.execute("SELECT * FROM bet_slips;")
    bets = c.fetchall()
    print("Bet Slips:", bets)

    conn.close()

if __name__ == "__main__":
    check_tables()
    check_users()
    check_bet_slips()