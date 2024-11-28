import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load the dataset
url = "hf://datasets/Paul/hatecheck/test.csv"
df = pd.read_csv(url)

# Preprocess data
df = df.dropna(subset=["test_case", "label_gold"])  # Drop missing values
df['label_gold'] = df['label_gold'].map({'hateful': 0, 'non-hateful': 1})  # Map labels to binary
# print(df.head())
# print(df['label_gold'].value_counts())  # Check label distribution


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['test_case'], df['label_gold'], test_size=0.2, random_state=42
)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'hate_speech_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
