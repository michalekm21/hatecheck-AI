import joblib

# Load the saved model and vectorizer
model = joblib.load('hate_speech_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def classify_text(text):
    """Classify text as hateful or non-hateful."""
    text_tfidf = vectorizer.transform([text])  # Transform the input text
    prediction = model.predict(text_tfidf)[0]  # Predict label
    return "Hateful" if prediction == 1 else "Non-Hateful"


if __name__ == "__main__":
    print("Hate Speech Classifier CLI")
    print("Type your text below (or type 'exit' to quit):")
    
    while True:
        user_input = input(">> ")
        if user_input.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        
        result = classify_text(user_input)
        print(f"Prediction: {result}")
