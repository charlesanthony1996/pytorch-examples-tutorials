import flask
from flask import Flask, request, jsonify
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim


app = Flask(__name__)

@app.route("/")
def index():
    return "Welcome to the books api"

# create a new book
@app.route("/book", methods =['POST'])
def add_book():
    data = request.json
    title = data.get("title")
    author = data.get("author")
    published_date = data.get("published_date")

    conn = sqlite3.connect("books.db")
    c = conn.cursor()
    c.execute("INSERT INTO books (title, author, published_date) VALUES (?, ?, ?)", (title, author, published_date))
    conn.commit()
    conn.close()

    return jsonify({"message":"Book added"}), 201

@app.route("/delete_book", methods=["POST"])
def delete_book():
    data = request.json()
    book_id = data.get("id")

    conn = sqlite3.connect("books.db")
    c = conn.cursor()

    c.execute("DELETE FROM books WHERE id=?", (book_id, ))
    rows_deleted = c.rowcount

    conn.commit()
    conn.close()

    if rows_deleted == 0:
        return jsonify({"message": "No book found with the given ID"}), 404
    else:
        return jsonify({"message": "Book deleted"}), 200








# get all books
@app.route("/books", methods=["GET"])
def get_books():
    conn = sqlite3.connect("books.db")
    c = conn.cursor()
    c.execute("SELECT * FROM books")
    rows = c.fetchall()
    conn.close()


    books = []
    for row in rows:
        books.append({"id": row[0], "title": row[1], "author": row[2], "published_date": row[3]})
    
    return jsonify(books)


# recommendation system
class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_books, n_factors = 50):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.book_factors = nn.Embedding(n_books, n_factors)

    def forward(self, user, book):
        return (self.user_factors(user) * self.book_factors(book)).sum(dim =1)


# dummy data -> when the app becomes real you should have real data here
n_users , n_books = 1000, 1000
x_users = torch.randint(0, n_users, size = (100,))
x_books = torch.randint(0, n_books, size= (100,))
y_ratings = torch.randint(1, 6, size = (100,)).float()

# initialize the model and optimizer
model = RecommenderNet(n_users, n_books)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for i in range(10):
    optimizer.zero_grad()
    predictions = model(x_users, x_books)
    loss = nn.MSELoss()(predictions, y_ratings)
    loss.backward()
    optimizer.step()


@app.route("/recommend", methods=["GET"])
def recommend_books():
    user_id = int(request.args.get("user_id"))
    books_ids = torch.arange(n_books)

    # use the model to predict ratings
    with torch.no_grad():
        predicted_ratings = model(torch.tensor([user_id] * n_books), book_ids).numpy()

    # get the top 5 book recommendations
    top_books = book_ids.numpy()[predicted_rating.argsort()[5:]]

    return jsonify({"recommended_books": top_books.tolist()})








# run the app
if __name__ == "__main__":
    app.run(debug=True)




