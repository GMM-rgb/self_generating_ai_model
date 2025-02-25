import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
import json
from difflib import get_close_matches

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

# Find best match for user input
def find_best_match(user_input, X_train, y_train):
    matches = get_close_matches(user_input, X_train, n=1, cutoff=0.5)
    if matches:
        match_index = X_train.index(matches[0])
        return y_train[match_index]
    else:
        return "I'm not sure how to respond to that yet. Can you teach me?"

# Train the model function
def train_model():
    X_train, y_train = load_data()
    if X_train is None:
        return
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(len(X_train[0]),)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(len(y_train[0]), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Training model...")
    model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_split=0.2)
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
