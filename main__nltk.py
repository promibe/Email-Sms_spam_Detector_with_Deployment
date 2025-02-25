#importing the necessary libraries
#libraries for dataframe preprocessing and visualization
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
#prepreocession libraries
import nltk
#importing libraries for stopwords, stem
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stopwords.words('english')
import string
from sklearn.feature_extraction.text import TfidfVectorizer


# set page configuration
st.set_page_config(page_title='Spam Message Detector', layout='centered')

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')



#function for text preprocessiong
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters (punctuation & special symbols)
    words = [word for word in words if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    string_punctuation = list(string.punctuation)
    filtered_words = [word for word in words if word not in stop_words and word not in string.punctuation]

    # Stemming
    stemmer = nltk.PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # Return processed text as a tokenized list
    return " ".join(stemmed_words)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load vectorizer (e.g., TfidfVectorizer or CountVectorizer)
with open("vectorizer.pkl", "rb") as file:
    tfidf = pickle.load(file)

def create_GUI():
    #header section
    st.title("Spam messages Classifier")
    st.write("This classify's all spam messages")

    #input text to classify
    st.header("Input Email Text/SMS Text")

    text = st.text_input("Enter text")

    # using the transform text function to transform the inputed text
    transformed_text = transform_text(text)

    #vectorize the transformed text
    vectorized_input = tfidf.transform(transform_text)

    if st.button("Classify Text"):
        result = model.predict(vectorized_input[0])

        if result == 0:
            st.success("Spam Message")
        else:
            st.success("Not Spam Message")

if __name__ == "__main__":
    create_GUI()





