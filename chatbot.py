import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issue for NLTK data download
ssl._create_default_https_context = ssl._create_unverified_context

# NLTK data path and download
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file with error handling
def load_intents(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("Intents file not found. Please check the file path.")
        return []

intents = load_intents("C:\\Users\\Mansi\\Chatbot using NPL\\intents.json")

# Vectorizer and classifier setup
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess data for model training
def preprocess_data(intents):
    tags, patterns = [], []
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(pattern)
    return patterns, tags

patterns, tags = preprocess_data(intents)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function to get responses
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    predicted_tag = clf.predict(input_text)[0]

    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

# Function to save chat history
def save_conversation(user_input, response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([user_input, response, timestamp])

# Display chat history
def display_chat_history():
    try:
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
    except FileNotFoundError:
        st.warning("No conversation history found yet.")

# Main chatbot interface
def main():
    st.title("Intents-Based Chatbot using NLP")

    # Sidebar Menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! Type a message to start chatting.")

        # Initialize chat log file if it doesn't exist
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("You:")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120)

            save_conversation(user_input, response)

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting! Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        display_chat_history()

    elif choice == "About":
        st.write("About This Project")
        st.write("""
        This chatbot is built using NLP and Logistic Regression to understand user intents.
        It leverages Streamlit for a simple and user-friendly interface.
        
        Key Features:
        - Uses NLP techniques to predict user intent.
        - Stores chat history for future reference.
        - Provides clear guidance for users on how to use the chatbot.
        
        Enjoy chatting!
        """)

if __name__ == '__main__':
    main()
