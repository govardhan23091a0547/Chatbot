import nltk
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

intents = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "Good morning"], "responses": ["Hello! How can I help you today?"]},
        {"tag": "hours", "patterns": ["What are your hours?", "When are you open?"], "responses": ["We're open every day from 9am to 9pm."]},
        {"tag": "location", "patterns": ["Where are you located?", "What is your address?"], "responses": ["We are located at 123 Main Street, Cityville."]},
        {"tag": "goodbye", "patterns": ["Bye", "Goodbye", "See you later"], "responses": ["Goodbye! Have a great day."]}
    ]
}

all_patterns = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        all_patterns.append(pattern)
        tags.append(intent['tag'])

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(all_patterns)  # Fitting must happen before transform

def predict_tag(user_input):
    user_input_processed = [user_input]
    user_vec = vectorizer.transform(user_input_processed)
    similarity = cosine_similarity(user_vec, X)
    idx = similarity.argmax()
    return tags[idx]

def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def chat():
    print("Customer Support Chatbot (type 'quit' to exit)")
    while True:
        user_input = input("You: ").lower()
        if user_input == 'quit':
            print("Chatbot: Goodbye!")
            break
        tag = predict_tag(user_input)
        response = get_response(tag)
        print(f"Chatbot: {response}")

chat()