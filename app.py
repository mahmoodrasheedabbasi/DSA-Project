import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_message(message):
    # Convert the message to lowercase
    message = message.lower()
    
    # Tokenize the message
    words = nltk.word_tokenize(message)
    
    # Initialize an empty list to store valid words
    filtered_words = []
    
    # Remove non-alphanumeric characters and stopwords
    for word in words:
        if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
            # Apply stemming to the word
            stemmed_word = ps.stem(word)
            filtered_words.append(stemmed_word)
    
    # Join the filtered words into a single string
    return " ".join(filtered_words)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("SMS Spam Classifier")

sms = st.text_area("Enter the message")

if st.button("Predict"):

    transformed_sms = transform_message(sms)

    vector_input = tfidf.transform([transformed_sms])

    result = int(model.predict(vector_input))
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")