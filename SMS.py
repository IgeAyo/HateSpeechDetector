import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os
nltk.download('stopwords')  # Download stopwords package
nltk.download('punkt')  # Download tokenizer package

# Define path
data_path = r"C:\Users\igeay\Downloads\naijahate.csv"


# Read data into a DataFrame
df = pd.read_csv(data_path)


# Separate data based on "dataset" column values
df_stratified = df[df["dataset"] == "stratified"]
df_al = df[df["dataset"] == "al"]
df_eval = df[df["dataset"] == "eval"]
df_random = df[df["dataset"] == "random"]


# Print some rows from each DataFrame to inspect
print("First few rows of df_stratified:")
print(df_stratified.head(), "\n")


print("First few rows of df_al:")
print(df_al.head(), "\n")


print("First few rows of df_eval:")
print(df_eval.head(), "\n")


print("First few rows of df_random:")
print(df_random.head(), "\n")


# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text):
   """
   Cleans text data by lowercasing, removing punctuation, stop words, and lemmatizing.
   """
   text = text.lower()
   allowed_chars = "@$%&"
   table = str.maketrans('', '', ''.join(char for char in string.punctuation if char not in allowed_chars))
   text = text.translate(table)
   tokens = TweetTokenizer().tokenize(text)
   stop_words = set(stopwords.words('english'))
   tokens = [word for word in tokens if word not in stop_words]
   lemmatizer = WordNetLemmatizer()
   tokens = [lemmatizer.lemmatize(word) for word in tokens]
   cleaned_text = ' '.join(tokens)
   return cleaned_text


# Clean text data in each DataFrame
for df in [df_stratified, df_al, df_eval, df_random]:
   df['cleaned_text'] = df['text'].apply(clean_text)


# Define label encoder
encoder = LabelEncoder()


# Encode labels in each DataFrame
for df in [df_stratified, df_al, df_eval, df_random]:
   df["encoded_label"] = encoder.fit_transform(df["class"])


# Combine stratified and al data
df_combined = pd.concat([df_stratified, df_al])


# Handle NaN values in the target variable
df_combined = df_combined.dropna(subset=['class'])


# Check class distribution
print("Class distribution in df_combined:")
print(df_combined['class'].value_counts())


# Handle classes with too few samples (less than 2)
min_samples = 2  # Minimum samples required per class
class_counts = df_combined['class'].value_counts()


# Classes with less than min_samples samples
classes_to_handle = class_counts[class_counts < min_samples].index


# Option: Remove classes with too few samples
df_combined = df_combined[~df_combined['class'].isin(classes_to_handle)]


# Option: Alternatively, merge them into another class or handle as you see fit


# Re-check class distribution
print("Updated class distribution in df_combined:")
print(df_combined['class'].value_counts())


# Ensure no NaNs in target variable before splitting
df_combined = df_combined.dropna(subset=['cleaned_text', 'class'])


# Stratified split for training and validation (maintain class distribution)
test_size = 0.2  # Proportion for test set (evaluation set)


X_train_val, X_test, y_train_val, y_test = train_test_split(df_combined['cleaned_text'],  # Features
                                                           df_combined['class'],  # Target variable
                                                           test_size=test_size,
                                                           stratify=df_combined['class'],
                                                           random_state=42)


# Further split training/validation sets
validation_size = 0.2  # 20% for validation set


X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size, random_state=42)


# Print the first few rows of df_stratified to inspect it
print("First few rows of df_stratified:")
print(df_stratified.head())


# Check for NaN valuess in 'cleaned_text' and 'class' columns of df_stratified
print(f"NaNs in df_stratified['cleaned_text']: {df_stratified['cleaned_text'].isnull().sum()}")
print(f"NaNs in df_stratified['class']: {df_stratified['class'].isnull().sum()}")


# Ensure no NaNs in target variable before fitting
df_stratified.dropna(subset=['class'], inplace=True)


# Check for NaNs in 'cleaned_text' and 'class' columns of df_stratified
print(f"NaNs in df_stratified['cleaned_text']: {df_stratified['cleaned_text'].isnull().sum()}")
print(f"NaNs in df_stratified['class']: {df_stratified['class'].isnull().sum()}")


# Define a pipeline with TF-IDF vectorizer and Random Forest
pipeline_rf = Pipeline([
   ('tfidf', TfidfVectorizer(tokenizer=TweetTokenizer().tokenize, ngram_range=(1, 2), max_features=5000, stop_words='english')),
   ('rf', RandomForestClassifier())
])


# Define the hyperparameter grid for tuning
param_grid_rf = {
   'rf__n_estimators': [100, 200, 300],
   'rf__max_depth': [None, 10, 20],
   'rf__min_samples_split': [2, 5, 10],
   'rf__min_samples_leaf': [1, 2, 4],
   'rf__class_weight': ['balanced', 'balanced_subsample']
}


# Use GridSearchCV to find the best hyperparameters
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)


# Train the model with hyperparameter tuning on the stratified set
grid_search_rf.fit(X_train, y_train)  # Use X_train and y_train here


# Save the best model
model_filename = 'best_model_rf.joblib'
model_path = os.path.join(os.path.dirname(__file__), model_filename)
joblib.dump(grid_search_rf.best_estimator_, model_path)


# Output the best hyperparameters
print("Best Hyperparameters for Random Forest:")
print(grid_search_rf.best_params_)


# Ensure no NaNs in true labels (df_eval['class']) before generating the report
df_eval.dropna(subset=['class'], inplace=True)


# Evaluate the best model on the eval set
best_model_rf = grid_search_rf.best_estimator_
eval_pred_rf = best_model_rf.predict(X_val)  # Use X_val for evaluation


# Check for NaNs in predictions
eval_pred_rf = pd.Series(eval_pred_rf).fillna("unknown")


# Print evaluation metrics
print("Evaluation Classification Report for Random Forest:")
print(classification_report(y_val, eval_pred_rf))


print("Evaluation Confusion Matrix for Random Forest:")
print(confusion_matrix(y_val, eval_pred_rf))


print("Evaluation Accuracy Score for Random Forest:")
print(accuracy_score(y_val, eval_pred_rf))