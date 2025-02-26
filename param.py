import sys
import time
import numpy as np
import pickle
import json
import requests
from bs4 import BeautifulSoup

# Import for Generative Model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

import warnings
warnings.filterwarnings("ignore")

# Load training data function
def load_data():
    data = []
    # Load original training data
    try:
        with open("training_data.json", "r") as f:
            training_data = json.load(f)
            data.extend(training_data)
    except FileNotFoundError:
        print("No original training data found. Please add a valid training_data.json file.")
    except json.JSONDecodeError:
        print("Error reading training_data.json. Ensure it is properly formatted.")
    
    # Load corrected training data
    try:
        with open("training_data_corrected.json", "r") as f:
            corrected_data = json.load(f)
            data.extend(corrected_data)
    except FileNotFoundError:
        print("No corrected training data found. If this is the first run, this is expected.")
    except json.JSONDecodeError:
        print("Error reading training_data_corrected.json. Ensure it is properly formatted.")

    if not data:
        print("No training data available.")
        return None
    else:
        return data

# Save updated training data
def save_training_data(training_data):
    with open("training_data.json", "w") as f:
        json.dump(training_data, f, indent=4)
    print("Training data updated with corrections.")

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

# Initialize the generative model
def initialize_generative_model():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        return tokenizer, model
    except Exception as e:
        print(f"Error initializing the generative model: {str(e)}")
        return None, None

# Generate a response using the generative model
def generate_response_gpt2(user_input, tokenizer, model):
    print("AI internal reasoning: Generating response using GPT-2...")
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, do_sample=True, top_p=0.95, top_k=60)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return the generated text after the user input
    response = text[len(user_input):].strip()
    return response

# Get additional definitions from training entries that might be relevant
def get_additional_definitions(user_input, training_data, best_match_input):
    definitions = []
    for entry in training_data:
        if entry.get("input") == best_match_input:
            continue
        if "definition" in entry:
            for word in entry["input"].split():
                if word.lower() in user_input.lower():
                    definitions.append(entry["definition"])
                    break
    return definitions

# Find best match for user input using the trained model
# Uses GPT-2 Model for telling it what to do, for better accuracy and future data for the model itself, which is not the gpt2 model, it's the model gpt2 is training.
def find_best_match(user_input, training_data, tokenizer, gpt2_model):
    # Load model, vectorizer, label encoder
    model = load_model('chat_model.h5', compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    print("AI internal reasoning: Breaking down your input...")
    # Tokenize the input
    tokens = user_input.split()
    print(f"Tokens identified: {tokens}")

    print("AI internal reasoning: Transforming input into numerical features using TF-IDF...")
    user_input_tfidf = vectorizer.transform([user_input]).toarray()

    print("AI internal reasoning: Predicting the most suitable response...")
    probs = model.predict(user_input_tfidf)
    predicted_class_index = np.argmax(probs, axis=1)
    predicted_response = label_encoder.inverse_transform(predicted_class_index)[0]
    confidence = probs[0][predicted_class_index[0]]

    print("AI internal reasoning: Evaluating possible responses and their confidence levels...")
    top_indices = probs[0].argsort()[-3:][::-1]
    for idx in top_indices:
        class_label = label_encoder.inverse_transform([idx])[0]
        probability = probs[0][idx]
        print(f" - '{class_label}': {probability:.2%} confidence")

    # Adjust confidence threshold based on input length
    if len(tokens) <= 2:
        confidence_threshold = 0.10  # Lower threshold for short inputs
    else:
        confidence_threshold = 0.60  # Original threshold for longer inputs

    if confidence < confidence_threshold:
        print("AI internal reasoning: Confidence is low. Generating a response...")
        response = generate_response_gpt2(user_input, tokenizer, gpt2_model)
    else:
        response = predicted_response
        # Append any additional definitions if relevant
        additional_defs = get_additional_definitions(user_input, training_data, predicted_response)
        if additional_defs:
            response += " " + " ".join(additional_defs)

    return response

# Train the model function with trial phase and correction mechanism
def train_model():
    training_data = load_data()
    if training_data is None:
        return

    # Prepare data
    X_train = [entry["input"] for entry in training_data]
    y_train = [entry["output"] for entry in training_data]

    # Vectorization
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3))
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()

    # Label Encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Build the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Training the AI model...")
    model.fit(np.array(X_train_tfidf), np.array(y_train_encoded), epochs=100, batch_size=8, validation_split=0.2)
    model.save("chat_model.h5")
    print("Model trained and saved successfully.")

    # Trial Phase: Testing the model's responses
    print("\nStarting trial phase to evaluate the model's responses...")
    score = 0
    corrections_needed = False
    for i, input_text in enumerate(X_train):
        print(f"\nTest {i+1}:")
        print(f"Input: {input_text}")

        # Transform input
        input_tfidf = vectorizer.transform([input_text]).toarray()

        # Predict
        probs = model.predict(input_tfidf)
        predicted_class_index = np.argmax(probs, axis=1)
        predicted_response = label_encoder.inverse_transform(predicted_class_index)[0]

        print(f"Expected Output: {y_train[i]}")
        print(f"Model's Output: {predicted_response}")

        if predicted_response == y_train[i]:
            print("Result: Correct (+1 point)")
            score += 1
        else:
            print("Result: Incorrect (-1 point)")
            score -= 1
            corrections_needed = True
            # Correct the response in the training data
            training_data[i]["output"] = predicted_response  # Applying the predicted response

    print(f"\nTrial phase completed. Total Score: {score}/{len(X_train)}")
    if corrections_needed:
        print("Corrections were made. Updating training data...")
        save_training_data(training_data)
        print("Please retrain the model to apply corrections.")
    else:
        print("The model performed well in the trial phase. No corrections needed.")

# Chat function; can run in interactive mode or process a single command-line message
def chat(interactive=True, initial_message=None):
    training_data = load_data()
    if training_data is None:
        return

    # Initialize the generative model
    tokenizer, gpt2_model = initialize_generative_model()
    if tokenizer is None or gpt2_model is None:
        print("Generative model not available. Chatbot will function without generative capabilities.")

    if not interactive and initial_message is not None:
        print("You:", initial_message)
        print("AI is thinking...")
        response = find_best_match(initial_message, training_data, tokenizer, gpt2_model)
        print("AI:", response)
        return

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            print("\nAI is thinking...")
            response = find_best_match(user_input, training_data, tokenizer, gpt2_model)
            print("\nAI:", response)
    except KeyboardInterrupt:
        print("\nExiting chat mode.")

# Main menu for interactive mode
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
    if len(sys.argv) > 1:
        initial_message = " ".join(sys.argv[1:])
        chat(interactive=False, initial_message=initial_message)
    else:
        main()
