# hatecheck-AI
Sure! This Python script is a complete example of training a machine learning model to classify text into two categories: "hateful" and "non-hateful." Here’s an educational breakdown of its key components:

---

### **1. Importing Libraries**
- **`pandas`**: Used for data manipulation and analysis.
- **`scikit-learn` modules**:
  - `train_test_split`: Splits the dataset into training and test subsets.
  - `TfidfVectorizer`: Converts text data into numerical features based on the TF-IDF method.
  - `LogisticRegression`: A simple and effective classification algorithm.
  - `classification_report`: Provides a summary of the model’s performance.
- **`joblib`**: Used to save the trained model and vectorizer for future use.

---

### **2. Loading the Dataset**
```python
url = "hf://datasets/Paul/hatecheck/test.csv"
df = pd.read_csv(url)
```
This line loads the dataset into a Pandas DataFrame. The dataset is hosted remotely (possibly on Hugging Face datasets) and contains two important columns:
- `test_case`: The text to classify.
- `label_gold`: The label indicating whether the text is hateful or non-hateful.

---

### **3. Preprocessing Data**
```python
df = df.dropna(subset=["test_case", "label_gold"])  
df['label_gold'] = df['label_gold'].map({'hateful': 0, 'non-hateful': 1})
```
- **Remove Missing Values**: Ensures there are no empty rows in critical columns.
- **Label Encoding**: Converts categorical labels (`hateful`, `non-hateful`) into numeric values (`0`, `1`). This is necessary because machine learning models require numerical input.

---

### **4. Splitting the Dataset**
```python
X_train, X_test, y_train, y_test = train_test_split(
    df['test_case'], df['label_gold'], test_size=0.2, random_state=42
)
```
- **Input (X)**: `df['test_case']` contains the text data.
- **Output (y)**: `df['label_gold']` contains the labels.
- **Train-Test Split**:
  - `test_size=0.2`: 20% of the data is used for testing.
  - `random_state=42`: Ensures reproducibility by setting a fixed random seed.

---

### **5. Text Vectorization**
```python
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```
- **TF-IDF**: Transforms text into numerical vectors based on term frequency and inverse document frequency.
  - Words occurring frequently in one document but rarely across others are given higher importance.
  - `max_features=5000`: Limits the number of unique words to 5000.
- **Fit and Transform**:
  - `fit_transform` (training set): Learns the vocabulary and transforms the text into TF-IDF features.
  - `transform` (test set): Uses the learned vocabulary to transform the test data.

---

### **6. Training the Model**
```python
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
```
- **Logistic Regression**: A popular and efficient linear model for binary classification.
- **Training**: The `fit` method trains the model using the transformed training data (`X_train_tfidf`) and corresponding labels (`y_train`).

---

### **7. Evaluating the Model**
```python
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))
```
- **Prediction**: `model.predict` generates predictions for the test set.
- **Classification Report**: Shows:
  - **Precision**: How many predicted positives are actual positives.
  - **Recall**: How many actual positives are correctly identified.
  - **F1-score**: Harmonic mean of precision and recall, providing a balanced measure.
  - **Support**: Number of true instances for each label in the test data.

---

### **8. Saving the Model and Vectorizer**
```python
joblib.dump(model, 'hate_speech_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
```
- **Model Persistence**: The trained model and vectorizer are saved to disk for later use.
  - `hate_speech_model.pkl`: Contains the trained Logistic Regression model.
  - `tfidf_vectorizer.pkl`: Contains the fitted TF-IDF vectorizer.
- This allows deployment or reuse of the model without retraining.

---

### **Key Takeaways**
1. **Preprocessing**: Data cleaning and label encoding are crucial for effective model training.
2. **Feature Extraction**: TF-IDF is a powerful technique to convert raw text into numerical data suitable for machine learning.
3. **Model Training**: Logistic Regression is a simple yet effective method for binary classification.
4. **Evaluation**: Use metrics like precision, recall, and F1-score to assess performance.
5. **Deployment**: Save trained components for scalability and reuse.

This script demonstrates a common end-to-end workflow in Natural Language Processing (NLP) projects.