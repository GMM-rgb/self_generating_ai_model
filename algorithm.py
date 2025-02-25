import sys
import time
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import json
import requests
from bs4 import BeautifulSoup

# Load training data function: returns the full list of training entries.
def load_data():
    try:
        with open("training_data.json", "r") as f:
            training_data = json.load(f)
        return training_data
    except FileNotFoundError:
        print("No training data found. Please add a valid training_data.json file.")
        return None
    except json.JSONDecodeError:
        print("Error reading training_data.json. Ensure it is properly formatted.")
        return None

# Fetch online information (if no confident match is found)
def fetch_online_info(query):
    try:
        print("AI internal reasoning: Searching online for additional information...")
        search_url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        snippets = soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")
        for snippet in snippets:
            text = snippet.get_text()
            if len(text) > 50:
                return text
        return "I'm not finding much on that topic. Could you please elaborate?"
    except Exception as e:
        return f"Sorry, I ran into an error fetching information: {str(e)}"

# Find best match for user input using the trained model and display thinking process
def find_best_match(user_input, training_data):
    # Load model, vectorizer, label encoder
    model = load_model('chat_model.h5')
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    print("AI internal reasoning: Breaking down your input...")
    # Tokenize the input
    tokens = user_input.split()
    print(f"Tokens identified: {tokens}")

    # Convert input to TF-IDF features
    print("AI internal reasoning: Transforming input into numerical features using TF-IDF...")
    user_input_tfidf = vectorizer.transform([user_input]).toarray()

    # Make prediction
    print("AI internal reasoning: Predicting the most suitable response...")
    probs = model.predict(user_input_tfidf)
    predicted_class_index = np.argmax(probs, axis=1)
    predicted_class = label_encoder.inverse_transform(predicted_class_index)
    response = predicted_class[0]

    # Display probabilities for top responses
    print("AI internal reasoning: Evaluating possible responses and their confidence levels...")
    top_indices = probs[0].argsort()[-3:][::-1]
    for idx in top_indices:
        class_label = label_encoder.inverse_transform([idx])[0]
        probability = probs[0][idx]
        print(f" - '{class_label}': {probability:.2%} confidence")

    # If the highest probability is below a threshold, consider fetching online info
    if probs[0][predicted_class_index[0]] < 0.60:
        print("AI internal reasoning: Confidence is low. Seeking additional information...")
        response = fetch_online_info(user_input)
    else:
        # Append any additional definitions from training data if relevant
        best_match_input = training_data[predicted_class_index[0]]["input"]
        additional_defs = get_additional_definitions(user_input, training_data, best_match_input)
        if additional_defs:
            response += " " + " ".join(additional_defs)
    return response

# Get additional definitions from training entries that might be relevant.
def get_additional_definitions(user_input, training_data, best_match_input):
    definitions = []
    # Check other entries (exclude the best match) for keywords in common.
    for entry in training_data:
        if entry.get("input") == best_match_input:
            continue
        if "definition" in entry:
            # Check if any word from the entry's input appears in the user query.
            for word in entry["input"].split():
                if word.lower() in user_input.lower():
                    definitions.append(entry["definition"])
                    break
    return definitions

# Train the model function using the training data from JSON.
def train_model():
    training_data = load_data()
    if training_data is None:
        return

    # Prepare data for training the model
    X_train = [entry["input"] for entry in training_data]
    y_train = [entry["output"] for entry in training_data]

    # Convert text data to numerical format
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()

    # Encode output labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Save the vectorizer and label encoder for later use
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Build the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(set(y_train)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Training the AI model...")
    model.fit(np.array(X_train_tfidf), np.array(y_train_encoded), epochs=50, batch_size=8, validation_split=0.2)
    model.save("chat_model.h5")
    print("Model trained and saved successfully.")

# Chat function; can run in interactive mode or process a single command-line message.
def chat(interactive=True, initial_message=None):
    training_data = load_data()
    if training_data is None:
        return

    # If a message is provided from the command-line, process it once.
    if not interactive and initial_message is not None:
        print("You:", initial_message)
        print("AI is thinking...")
        response = find_best_match(initial_message, training_data)
        print("AI:", response)
        return

    # Otherwise, run in interactive mode.
    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            print("\nAI is thinking...")
            response = find_best_match(user_input, training_data)
            print("\nAI:", response)
    except KeyboardInterrupt:
        print("\nExiting chat mode.")

# Main menu for interactive mode.
def main():
    while True:
        print("\n===== AI Chatbot Menu =====")
        print("1. Train AI")
        print("2. Chat with AI")
        print("3. Exit")
        choice = input("Choose an option: ")
        if choice == "1":
            train_model()
        elif choice == "2":
            chat()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    # If command-line arguments are provided, treat them as a single message to process.
    if len(sys.argv) > 1:
        initial_message = " ".join(sys.argv[1:])
        chat(interactive=False, initial_message=initial_message)
    else:
        main()
