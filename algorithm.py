import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
import json
from difflib import get_close_matches
import requests
from bs4 import BeautifulSoup
import time

# Load training data function
def load_data():
    try:
        with open("training_data.json", "r") as f:
            data = json.load(f)
        
        X_train = [entry["input"] for entry in data]
        y_train = [entry["output"] for entry in data]
        return X_train, y_train
    except FileNotFoundError:
        print("No training data found. Please add a valid training_data.json file.")
        return None, None
    except json.JSONDecodeError:
        print("Error reading training_data.json. Ensure it is properly formatted.")
        return None, None

# Fetch online information
def fetch_online_info(query):
    try:
        search_url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        snippets = soup.find_all("span")
        
        for snippet in snippets:
            text = snippet.get_text()
            if len(text) > 50:
                return text
        return "I couldn't find enough information online."
    except Exception as e:
        return f"Error fetching online info: {str(e)}"

# Find best match for user input
def find_best_match(user_input, X_train, y_train):
    matches = get_close_matches(user_input, X_train, n=1, cutoff=0.5)
    if matches:
        match_index = X_train.index(matches[0])
        print("AI internal reasoning: Found a similar query in training data.")
        return y_train[match_index]
    else:
        print("AI internal reasoning: No close match found. Fetching online information...")
        return fetch_online_info(user_input)

# Train the model function
def train_model():
    X_train, y_train = load_data()
    if X_train is None:
        return
    
    # Convert text data to numerical format
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    
    # Encode output labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(set(y_train)), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Training model...")
    model.fit(np.array(X_train_tfidf), np.array(y_train_encoded), epochs=10, batch_size=32, validation_split=0.2)
    model.save("chat_model.h5")
    print("Model trained and saved successfully.")

# Chat function
def chat():
    X_train, y_train = load_data()
    if X_train is None:
        return
    try:
        while True:
            user_input = input("You: ")
            if user_input.endswith("\""):
                user_input = user_input.strip("\"")
                print("AI is thinking...")
                time.sleep(1)  # Simulate a delay for thinking
                print("AI internal reasoning: Processing your input...")
                response = find_best_match(user_input, X_train, y_train)
                print("AI:", response)
            elif user_input.lower() == "exit":
                break
    except KeyboardInterrupt:
        print("\nExiting chat mode.")

# Main menu
def main():
    while True:
        print("1. Train AI")
        print("2. Chat with AI")
        choice = input("Choose an option: ")
        if choice == "1":
            train_model()
        elif choice == "2":
            chat()
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main()
