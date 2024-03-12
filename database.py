import sqlite3

# creating the connection
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to {db_file}, sqlite version: {sqlite3.version}")
    except Exception as e:
        print(e)
    return conn

# table creation method
def create_table():
    try:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY,
        name TEXT NOT NULL, email text NOT NULL); """)
        print("table created")

    except Exception as e:
        print(e)


# adding a new user
def add_user(conn_user):
    sql = """ INSERT INTO users(name, email) VALUES(?,?) """
    curr = conn.cursor()
    cur.execute(sql, user)
    conn.commit()
    return cur.lastrowid

# query and display all users
def get_all_users(conn):
    cur = conn.cursor()
    



