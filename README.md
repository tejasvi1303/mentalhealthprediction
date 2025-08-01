# Mental Health Text Classifier

A machine learning–powered web application that detects mental health conditions from textual input. The system supports multiple input modes (CSV bulk upload, image with embedded text via OCR, and direct text entry) and classifies statements into categories such as **Depression**, **Anxiety**, **Stress**, **Normal**, etc. Built with **Streamlit** for the frontend and a hybrid **NLP + ML pipeline** (TF-IDF, SMOTE, SVM / Bernoulli Naive Bayes) for backend processing.

## 🚀 Features

- **Multi-modal input**
  - CSV upload for bulk data exploration and training
  - Image-to-text extraction via OCR (`pytesseract`)
  - Manual single-statement classification
- **Flexible preprocessing**
  - Lowercasing, punctuation removal
  - Lemmatization via NLTK, spaCy, or both combined
  - Optional multiprocessing for speed
- **Rich feature engineering**
  - TF-IDF vectorization with unigram & bigram support
  - Configurable feature dimensionality (e.g., 3,000–500,000)
  - Class balancing using **SMOTE**
- **Modeling**
  - Primary: Support Vector Machine (SVM)
  - Alternative: Bernoulli Naive Bayes
- **Prediction**
  - Confidence score output
  - Entity extraction of mental-health-related terms
- **Exploratory Data Analysis (EDA) & Visualizations**
  - Category distributions
  - Statement length histograms (with outlier handling)
  - Boxplots & violin plots per label
  - Word clouds per class
  - Word frequency heatmaps

## 🗂 Dataset Format

Input dataset must be a CSV with the following columns:

| Column Name | Description                                      |
|-------------|--------------------------------------------------|
| `statement`  | Text describing the person's thoughts/feelings    |
| `status`     | Mental health label (e.g., Depression, Anxiety)  |

Example row:
`csv
statement,status
"I have trouble sleeping and feel anxious all the time",Anxiety


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

health

📊 Evaluation Metrics
Accuracy

Precision, Recall, F1-score (per class)

Classification report

Visual diagnostic plots:

Label distribution

Statement length analysis

Word clouds

Heatmaps and violin/box plots

🧪 Example Usage
Single text classification
Enter a textual statement like:

"I feel hopeless and can't concentrate on anything."

The system will output:

Predicted label: Depression

Confidence: 0.87

Key extracted terms: ["hopeless", "concentrate"]

Bulk training
Upload a CSV with labeled statements, explore class balance, train the model, and visualize how features distribute across labels.

🧩 Configurable Options
Choice of lemmatizer: NLTK, spaCy, or both

TF-IDF n-gram range

Feature count cap

SMOTE toggle

Choice of classifier (SVM or Naive Bayes)

📦 Dependencies
Key libraries used:

streamlit

scikit-learn

imblearn (for SMOTE)

NLTK

spaCy

pytesseract

matplotlib, seaborn

wordcloud

pandas, numpy

(See requirements.txt for full list)

🔍 Potential Applications
Mental health screening tools

Early warning systems in counseling platforms

Research dashboards for psychology / behavioral analysis

Integration into telehealth or support chatbots

🛠️ Development & Contribution
Fork the repo

Create a feature branch

Commit changes with clear messages

Open a pull request describing your updates


