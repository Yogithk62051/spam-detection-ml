import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1) Create dataset
data = {
    "message": [
        "Win a free iPhone now",
        "Congratulations you won cash prize",
        "Call now to claim your reward",
        "Free offer just for you",
        "Hello, how are you?",
        "Let's meet tomorrow",
        "Are you coming to office today?",
        "Please send the report by evening",
        "Dinner at 8 pm?",
        "Can we talk later?"
    ],
    "label": [
        "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham", "ham"
    ]
}

# 2) Convert into DataFrame
df = pd.DataFrame(data)

# 3) Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])

# 4) Train the model
model = MultinomialNB()
model.fit(X, df["label"])

# 5) Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and Vectorizer saved successfully!")

