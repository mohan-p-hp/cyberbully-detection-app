# backend/train_model.py (Advanced Version)

import pandas as pd
import numpy as np
import re
import html
import ast
import joblib
import os
import warnings
from nltk.corpus import stopwords
import emoji

# ML/DL Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
from sentence_transformers import SentenceTransformer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Basic Cleaning & Feature Engineering (Phases 3 & 4) ---
# (Functions: clean_text, TextStats, load_and_prepare_data, make_preprocessor)
def clean_text(s): # ... (same as before)
    if not isinstance(s, str): return ""
    # 1. Unescape HTML
    s = html.unescape(s)
    # 2. NEW: Convert emojis to their text descriptions
    s = emoji.demojize(s, delimiters=(":", ":"))
    # 3. Remove non-printable characters
    s = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', s)
    # 4. Replace URLs
    s = re.sub(r'https?://\S+|www\.\S+', ' URL ', s)
    # 5. Convert to lowercase and clean up spaces
    s = s.lower().strip()
    return re.sub(r'\s+', ' ', s)

class TextStats(BaseEstimator, TransformerMixin): # ... (same as before)
    def fit(self, X, y=None): return self
    def transform(self, X):
        texts = pd.Series(X).fillna("").astype(str)
        return np.vstack([
            texts.str.split().str.len().values,
            texts.str.len().values,
            texts.apply(lambda t: sum(1 for c in t if c.isupper()) / max(1, len(t))).values
        ]).T

def load_and_prepare_data(filepath="data/Cyberbullying.csv"): # ... (same as before)
    df = pd.read_csv(filepath)
    if 'Unnamed: 0' in df.columns: df = df.drop('Unnamed: 0', axis=1)
    df['label'] = df['annotation'].apply(lambda s: ast.literal_eval(s)['label'][0] if isinstance(s, str) else None)
    df.rename(columns={'content': 'text'}, inplace=True)
    final_df = df[['text', 'label']].copy(); final_df.dropna(inplace=True)
    final_df['text'] = final_df['text'].apply(clean_text)
    print("Data loaded and prepared successfully.")
    return final_df

def make_tfidf_preprocessor():
    return ColumnTransformer(
        [('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=3), 'text')],
        remainder='drop'
    )

# --- NEW: Function to create BERT Embeddings ---
def create_bert_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Uses a pre-trained SentenceTransformer model to convert a list of texts
    into high-quality sentence embeddings.
    """
    print(f"\n--- Generating BERT embeddings using '{model_name}' ---")
    print("This may take a few minutes and will download the model on first run...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    print("--- Embeddings generated successfully! ---")
    return embeddings

# --- Main Training and Selection Function ---
def train_and_select_model(output_path='backend/models/cyber_model.pkl'):
    df = load_and_prepare_data()
    X_text = df['text'].tolist() # Input for BERT
    X_df = pd.DataFrame({'text': df['text']}) # Input for TF-IDF
    y = df['label'].values
    
    # 1. Create BERT Embeddings
    X_bert = create_bert_embeddings(X_text)
    
    # 2. Split data for both feature sets
    X_tfidf_train, X_tfidf_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
    X_bert_train, X_bert_test, _, _ = train_test_split(X_bert, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Define all models to be tested
    tfidf_preprocessor = make_tfidf_preprocessor()
    models = {
        'Logistic Regression (TF-IDF)': Pipeline([('pre', tfidf_preprocessor), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
        'Linear SVC (TF-IDF)': Pipeline([('pre', tfidf_preprocessor), ('clf', LinearSVC(class_weight='balanced', dual=False))]),
        'Passive Aggressive (TF-IDF)': Pipeline([('pre', tfidf_preprocessor), ('clf', PassiveAggressiveClassifier(class_weight='balanced'))]),
        'LightGBM (TF-IDF)': Pipeline([('pre', tfidf_preprocessor), ('clf', lgb.LGBMClassifier(class_weight='balanced'))]),
        'Logistic Regression (BERT)': LogisticRegression(max_iter=1000, class_weight='balanced') # BERT features are already numbers
    }
    
    best_model, best_accuracy, best_name = None, 0.0, ""

    print("\n--- Starting Model Training and Selection ---")
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Use the correct feature set for the current model
        if "(BERT)" in name:
            model.fit(X_bert_train, y_train)
            preds = model.predict(X_bert_test)
        else: # For TF-IDF models
            model.fit(X_tfidf_train, y_train)
            preds = model.predict(X_tfidf_test)
            
        acc = accuracy_score(y_test, preds)
        print(f"  Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        if acc > best_accuracy:
            best_accuracy, best_model, best_name = acc, model, name
            print(f"*** New best model found: {name} with accuracy {acc:.4f} ***")

    # Save the overall best model (could be BERT or TF-IDF based)
    if best_model:
        # NOTE: For a BERT model, we save the simple classifier, not a full pipeline.
        # The API will need to be updated to generate embeddings for new text.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(best_model, output_path)
        print(f"\n--- Best overall model '{best_name}' saved to {output_path} ---")

if __name__ == "__main__":
    train_and_select_model()