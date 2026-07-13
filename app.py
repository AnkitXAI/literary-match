import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.urandom(24)

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["literary_match"]
users_col = db["users"]
books_col = db["books"]

# Check if the database has been populated by the ETL script
def verify_db_population():
    if books_col.count_documents({}) == 0:
        print("\n[WARNING] MongoDB books collection is empty.")
        print("Please run 'python ingest_data.py' to populate the database with real Goodreads data first.\n")

# Run the sanity check immediately when app.py loads
verify_db_population()

@app.before_request
def check_login():
    allowed_routes = ['login', 'register', 'static']
    if 'user' not in session and request.endpoint not in allowed_routes and request.endpoint:
        return redirect(url_for('login'))

# --- Recommendation Engine Helper ---
def get_recommendations(book_title, num_recs=5):
    """Computes TF-IDF & Cosine Similarity on descriptions of the real Goodreads books."""
    books = list(books_col.find())
    if not books:
        return []
    
    df = pd.DataFrame(books)
    
    if book_title not in df['title'].values:
        return []
    
    # Compute TF-IDF Matrix on real book descriptions
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    
    # Compute Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get index of selected book
    idx = df[df['title'] == book_title].index[0]
    
    # Sort and rank similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the book itself
    sim_scores = [s for s in sim_scores if s[0] != idx]
    
    # Return top N matches
    top_indices = [i[0] for i in sim_scores[:num_recs]]
    recommended_books = df.iloc[top_indices].to_dict(orient='records')
    
    for b in recommended_books:
        b['_id'] = str(b['_id'])
    
    return recommended_books

# --- Web Routes ---

@app.route('/')
def index():
    # Show the "Top 50 Books" out of the real Goodreads records based on rating
    top_50 = list(books_col.find().sort("rating", -1).limit(50))
    for book in top_50:
        book['_id'] = str(book['_id'])
    return render_template('index.html', books=top_50, username=session.get('user'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        if users_col.find_one({"username": username}):
            flash("Username already exists.", "error")
            return redirect(url_for('register'))
            
        hashed_pw = generate_password_hash(password)
        users_col.insert_one({"username": username, "password": hashed_pw})
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        user = users_col.find_one({"username": username})
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.", "error")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    results = list(books_col.find({"title": {"$regex": query, "$options": "i"}}))
    for book in results:
        book['_id'] = str(book['_id'])
    return render_template('index.html', books=results, query=query, username=session.get('user'))

@app.route('/recommend/<book_id>')
def recommend(book_id):
    try:
        book = books_col.find_one({"_id": ObjectId(book_id)})
        if not book:
            flash("Book not found.", "error")
            return redirect(url_for('index'))
            
        book['_id'] = str(book['_id'])
        recommendations = get_recommendations(book['title'], num_recs=5)
        return render_template('recommendations.html', book=book, recommendations=recommendations, username=session.get('user'))
    except Exception as e:
        flash("An error occurred loading recommendations.", "error")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)