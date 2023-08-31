import sqlite3

# connect to an sqlite3 database
conn = sqlite3.connect("books.db")

c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS books 
(id INTEGER PRIMARY KEY AUTOINCREMENT, 
title TEXT NOT NULL, 
author TEXT NOT NULL, 
published_date TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS book_ratings
(user_id INTEGER,
book_id INTEGER,
rating INTEGER, 
PRIMARY KEY (user_id, book_id))''')


# commit changes and close
conn.commit()
conn.close()



