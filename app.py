import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


# Function to preprocess the input text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text
    y = []

    # Removing alphanumeric tokens
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Removing stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Performing stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  # Join the list into a single string with spaces


# Load the saved models (TF-IDF Vectorizer and the classification model)
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Setting up the main title and description
st.title("📧 Email/SMS Spam Classifier")
st.write("""
### Enter a message to determine whether it's Spam or Not Spam. 
This classifier uses **natural language processing (NLP)** techniques to preprocess and predict based on your input.
""")

# Input text field for user to enter the message
st.write("#### Message Input:")
input_sms = st.text_area("Type or paste your message here", height=150)

# Add a button to trigger the classification
if st.button("🔍 Classify Message"):
    if input_sms.strip():  # Ensure there's text in the input
        ## 1. Preprocess the input text
        with st.spinner('Processing...'):
            transformed_sms = transform_text(input_sms)

        ## 2. Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])

        ## 3. Predict the label (Spam or Not Spam)
        result = model.predict(vector_input)[0]

        ## 4. Display the result with appropriate color and message
        if result == 1:
            st.success("🔴 This message is classified as **Spam**.")
        else:
            st.success("🟢 This message is classified as **Not Spam**.")
    else:
        st.warning("Please enter a valid message to classify.")

# Adding a footer with a reference to your classifier and author
st.markdown("""
---
Developed using **Streamlit** and **NLP techniques**.  
Author: **Aditya Yadav**
""")
