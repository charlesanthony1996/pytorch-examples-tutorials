import flask
from flask import Flask, request, jsonify
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim

from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin


app = Flask(__name__)
app.secret_key = "secret"

# initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)

# user class for demostration purposes

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username
    
    def is_active(self):
        return True

    def is_authenticated(self):
        return True

    def is_anonymous(self):
        return True

    def get_id(self):
        return str(self.id)

    


# example user data
users = {1: User(1, "Charles"), 2: User(2, "Jane")}

# add user_loader callback
@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))

# route to login (for demonstration, hard-coded)
@app.route("/login", methods=["GET"])
def login():
    user = users.get(1)
    login_user(user)
    return "Logged in"


# route to logout
@app.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    return "Logged out"


# go to the users profile
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == "POST":
        new_name = request.form.get("username")
        current_user.username = new_name
    return f"Hello, {current_user.username}. <form method='post'> <input name='username'> </input> <input type='submit'></input> </form>"


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




