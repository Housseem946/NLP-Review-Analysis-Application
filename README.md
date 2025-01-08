# NLP Review Analysis and Prediction Application

This repository contains an NLP-based web application built with **Streamlit**. The app provides functionalities such as sentiment analysis, star rating prediction, topic modeling, and interactive summaries using advanced machine learning and NLP models.

---

## Features

### 1. **Data Cleaning and Preprocessing**
- Cleaning and standardizing review text:
  - Removal of punctuation and special characters.
  - Lowercasing and tokenization.
  - Spelling correction using `TextBlob` and custom rules.
- Translation of missing English reviews from French using `TextBlob`.
- Generated additional features:
  - Review length.
  - Sentiment score (polarity).
- Dataset preparation for supervised tasks:
  - Star rating classification (1-5 stars).
  - Sentiment analysis (positive, neutral, negative).

### 2. **Supervised Learning**
- **Star Rating Prediction**:
  - Used a **Random Forest Classifier** trained on `TF-IDF` features.
  - Predicted ratings based on review content.
- **Sentiment Analysis**:
  - Classifies reviews into positive, neutral, or negative sentiments.
  - Random Forest trained with labeled data for accuracy.

### 3. **Unsupervised Learning**
- **Topic Modeling**:
  - Implemented Latent Dirichlet Allocation (LDA) for topic extraction.
  - Visualized topics using `pyLDAvis`.

### 4. **Embedding and Similarity**
- Trained **Word2Vec** embeddings to find semantically similar words.
- Performed semantic search using cosine similarity.

### 5. **Explainable AI**
- **SHAP**:
  - Explained Random Forest predictions with SHAP values.
  - Displayed key features influencing predictions.

### 6. **Interactive Streamlit Application**
- **Key Functionalities**:
  - **Rating Prediction**:
    - Predict the number of stars (1-5) for user-entered reviews.
  - **Sentiment Prediction**:
    - Determine if the sentiment is positive, neutral, or negative.
  - **Summarization**:
    - Generate summaries for user reviews using pre-trained models.
  - **Explain Predictions**:
    - Visualize features influencing star rating predictions.
  - **RAG-Based Summary**:
    - Retrieve and summarize similar reviews for a given input.
  - **Question Answering**:
    - Answer dataset-related questions using a QA pipeline.
- **Visualizations**:
  - Topic distributions and SHAP explanations.

---

## Installation and Usage

### Prerequisites
- Python 3.10 or higher
- Required libraries (install via `pip`):

1. Clone the repository:

```bash
git clone https://github.com/Housseem946/NLP-Review-Analysis-Application.git
cd NLP-Review-Analysis-Application
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and go to http://localhost:8501.


## Project Directory Structure

The project directory contains the following files and folders:

1. deep_learning_model/
Purpose: Contains the trained deep learning model for predicting star ratings based on reviews.
Content:
assets/: Metadata files required by TensorFlow.
variables/: Model weights and configurations.
saved_model.pb: Main TensorFlow SavedModel format file.
tf_model.h5: Optional H5 format file for deep learning model.

3. app.py
Purpose: Main Streamlit application script to run the interactive web app.
Key Features:
User review analysis, rating predictions, and summarization.
Includes functionalities for question answering and explainability.

5. cleaned_reviews.csv
Purpose: The cleaned and preprocessed dataset used for model training and evaluation.
Content:
Processed reviews with additional features like review length and sentiment scores.
Translated reviews (avis_eng_corrected), ratings (note), and sentiments.

7. Data_cleaning_and_analysis.py
Purpose: Python script for data cleaning, preprocessing, and exploratory analysis.
Key Functions:
Cleaning raw reviews, handling missing values, and correcting text.
Generating additional features and preparing the dataset for modeling.

9. random_forest_model.pkl
Purpose: Trained Random Forest Classifier for predicting star ratings.
Format: Saved using joblib.

11. rf_sentiment_model.pkl
Purpose: Random Forest model trained for sentiment analysis (positive, neutral, negative).
Format: Saved using joblib.

13. tfidf_vectorizer.pkl
Purpose: TF-IDF vectorizer used for transforming text into numerical features for the star rating prediction model.
Format: Saved using joblib.

15. tfidf_vectorizer_sentiment.pkl
Purpose: TF-IDF vectorizer used for transforming text into numerical features for the sentiment analysis model.
Format: Saved using joblib.

## Application Interface

![WhatsApp Image 2025-01-08 à 23 50 20_b9585f1c](https://github.com/user-attachments/assets/1aca5982-bc44-4ac2-ac28-ff8bca72f47b)

![WhatsApp Image 2025-01-08 à 23 50 21_25e7871b](https://github.com/user-attachments/assets/43d1c5da-2dcf-45f1-9a99-9825f296955a)

Here are example of the fonctionallities : 

![WhatsApp Image 2025-01-08 à 23 50 21_ab2eefe7](https://github.com/user-attachments/assets/071aaed2-5f01-4ccf-8b82-a69d54d1c555)

![WhatsApp Image 2025-01-08 à 23 50 21_e22642df](https://github.com/user-attachments/assets/4720b2e4-f8f5-4c5e-955f-f6d631047869)

![WhatsApp Image 2025-01-08 à 23 50 37_e0929f18](https://github.com/user-attachments/assets/e9d9cdb8-6765-4a3b-be9e-71f1e9fc9287)

![WhatsApp Image 2025-01-08 à 23 50 51_a3bf51a2](https://github.com/user-attachments/assets/05bc2cba-23c0-47fc-9d6c-5a6d825dcb15)

![WhatsApp Image 2025-01-08 à 23 51 04_e425f430](https://github.com/user-attachments/assets/8734c7be-354d-435f-9d85-2424fba90148)

![WhatsApp Image 2025-01-08 à 23 51 27_aa266a57](https://github.com/user-attachments/assets/e7f9c44e-64a1-4ed8-9945-b5860d8e2c09)

![WhatsApp Image 2025-01-08 à 23 57 13_8ecccf75](https://github.com/user-attachments/assets/b2f86b40-2b4b-4877-b679-9802c51eb0a4)












