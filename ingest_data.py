import ast
import os
import pandas as pd
from pymongo import MongoClient

def run_etl_pipeline():
    print("--- Starting ETL Pipeline ---")
    
    # 1. Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["literary_match"]
    books_col = db["books"]
    
    # 2. Extract: Read the local Goodreads Extended Dataset
        # This dataset contains real book descriptions, genres, and ratings
        # url = "https://raw.githubusercontent.com/malcolmosh/goodbooks-10k/master/books_enriched.csv"
    file_path = "books_enriched.csv"
    print(f"Reading dataset from local file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found in the current directory.")
        print("Please place your downloaded 'books_enriched.csv' in the same folder as this script.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # 3. Transform: Clean and Format the Data
    print("Transforming and cleansing dataset...")
    
    # Drop rows that are missing critical fields required for recommendations
    df = df.dropna(subset=['title', 'authors', 'description', 'average_rating', 'genres'])
    
    # Clean up the string format of authors (e.g., "['J.K. Rowling', 'Mary GrandPré']" -> "J.K. Rowling")
    def clean_author(author_str):
        if isinstance(author_str, str):
            # Strip brackets and take the primary author
            clean_str = author_str.strip("[]'\"").split(',')[0]
            return clean_str.replace("'", "").replace('"', '').strip()
        return "Unknown Author"

    df['author'] = df['authors'].apply(clean_author)
    
    # Clean up and extract the main genre from the stringified genre list
    def extract_main_genre(genre_str):
        if not isinstance(genre_str, str) or genre_str == '[]':
            return "General"
        try:
            # Safely evaluate string representation of lists
            genre_list = ast.literal_eval(genre_str)
            if isinstance(genre_list, list) and len(genre_list) > 0:
                return genre_list[0].capitalize()
        except Exception:
            pass
        return "General"

    df['genre'] = df['genres'].apply(extract_main_genre)
    
    # Rename rating column for our application schema
    df = df.rename(columns={'average_rating': 'rating'})
    
    # Select only relevant columns
    processed_df = df[['title', 'author', 'genre', 'rating', 'description']].copy()
    
    # Optimize dataset size: Take the top 1,500 highest-rated books.
    # This guarantees high-quality metadata and fast TF-IDF matrix operations on standard hardware.
    processed_df = processed_df.sort_values(by='rating', ascending=False).head(1500)
    
    # Convert dataframe to dictionary format for MongoDB
    books_data = processed_df.to_dict(orient='records')
    
    # 4. Load: Populate MongoDB
    print("Clearing any existing data in the books collection...")
    books_col.delete_many({})
    
    print(f"Loading {len(books_data)} processed records into MongoDB...")
    books_col.insert_many(books_data)
    
    print("--- ETL Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    run_etl_pipeline()