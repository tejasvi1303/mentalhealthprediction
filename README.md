This project builds a machine learning-based web application for detecting mental health conditions from textual statements. The app supports multiple input methods (CSV, image with text, direct text) and classifies them into categories like Depression, Anxiety, Stress, etc.
The app is built using Streamlit for the frontend and NLP + ML (TF-IDF + SVM/Naive Bayes) for backend processing.
Features
 Upload CSV files with mental health statements for EDA + training

 Image-to-text support via OCR (pytesseract)

 Manual text input classification

 Visualizations (bar plots, histograms, violin plots, KDE, word clouds, heatmaps)

 Preprocessing options: NLTK, spaCy, or Combined

 Model training with TF-IDF + SMOTE + SVM

 Prediction with confidence scores

 Entity extraction for mental health terms

Dataset
The dataset should be in CSV format with the following columns:
Column Name	Description
statement	A text describing the person's experience
status	Mental health label (e.g., Depression, Normal, Anxiety, etc.)

Installation
1) Clone the repository:
git clone https://github.com/your-username/mental-health-classifier.git
cd mental-health-classifier

2) Install dependencies:
pip install -r requirements.txt

3) Download required NLTK & spaCy resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# spaCy
python -m spacy download en_core_web_sm

Running the App
To launch the Streamlit web app:
streamlit run mentalHealth.py
You will be presented with three tabs:
Upload CSV – upload dataset, view EDA, train model
Image Input – upload image with text (like mental health tweet/sentence)
Text Input – manually enter a sentence and classify

Model Pipeline:
1) Preprocessing
2)Lowercasing, punctuation removal
3)Lemmatization using NLTK and/or spaCy
4)Optional multiprocessing for speed
5)Vectorization
6)TF-IDF with unigram & bigram features
7)Max 3000–500000 features
8)Balancing
9)SMOTE to handle class imbalance
Training

Support Vector Machine (SVM)

Bernoulli Naive Bayes (as alternative)

Evaluation

Classification report

Accuracy, precision, recall, F1-score

Visualizations:
Distribution of mental health categories
Statement length histograms (before/after outlier removal)
Boxplots & violin plots across categories
Word clouds per label
Word frequency heatmaps
