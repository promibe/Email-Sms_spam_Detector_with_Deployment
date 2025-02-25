# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from catboost import CatBoost
from xgboost import XGBClassifier
from sklearn.svm import SVC
import streamlit as st
import pickle
import spacy  # Using SpaCy instead of NLTK
from spacy.lang.en.stop_words import STOP_WORDS  # Importing stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Set page configuration
st.set_page_config(page_title="Spam Message Detector", layout="centered")

# Function for text preprocessing
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Process the text with SpaCy
    doc = nlp(text)

    # Remove stopwords, punctuation, and non-alphanumeric tokens
    filtered_tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in STOP_WORDS]

    # Return processed text as a tokenized list
    return " ".join(filtered_tokens)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer (e.g., TfidfVectorizer or CountVectorizer)
with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

def create_GUI():
    # Header section
    st.title("Spam Messages Classifier")
    st.write("This classifies all spam messages.")

    # Input text to classify
    st.header("Input Email Text/SMS Text")
    text = st.text_input("Enter text")

    if st.button("Classify Text"):
        # Using the transform_text function to transform the inputted text
        transformed_text = transform_text(text)

        # Vectorize the transformed text
        vectorized_input = tfidf.transform([transformed_text])  # Fixed input format

        # Predict using the trained model
        result = model.predict(vectorized_input)[0]

        if result == 0:
            st.success("Spam Message")
        else:
            st.success("Not Spam Message")

if __name__ == "__main__":
    create_GUI()
