# NLP Review Analysis Application

## Overview

This project demonstrates a comprehensive end-to-end Natural Language Processing (NLP) pipeline, including supervised and unsupervised learning techniques, designed for analyzing customer reviews. 
The final deliverable is a Streamlit application that predicts review sentiments, categorizes reviews into themes, and provides additional insights like semantic similarity and topic modeling.


## Features

### Data Cleaning

* Tokenization and Preprocessing:

Removed stopwords, punctuation, and unnecessary characters.

Applied tokenization to break text into words.

* Highlighting Frequent Words and n-grams:

Extracted the most common unigrams and bigrams from the text data.

* Spelling Correction:

Used TextBlob for correcting typos in reviews.

* Translation Cleanup:

Cleaned and verified translated reviews for consistency.

### Topic Modeling

* Latent Dirichlet Allocation (LDA):

Modeled 5 topics based on the reviews using TF-IDF vectorization.

* Interpretation:

Extracted and displayed the top 10 words for each topic.

* Saved Results:

Topics and their associated keywords were saved for further analysis.

### Embedding for Similarity Analysis

* Word2Vec Training:

Trained Word2Vec on tokenized reviews to learn word embeddings.

* Cosine Similarity:

Measured semantic similarity between sample words.

* Visualization:

Used t-SNE for visualizing word embeddings in a 2D space.

* Semantic Search:

Implemented a feature to find the most similar words to a given query.

### Supervised Learning for Sentiment Analysis

* Classical Models:

Trained and evaluated Random Forest and SVM classifiers on TF-IDF features.

* Deep Learning Models:

Implemented an LSTM-based sentiment analysis model with embedding layers.

* Pre-trained Models:

Used Hugging Face's nlptown/bert-base-multilingual-uncased-sentiment for multilingual sentiment classification.

* Comparison:

Compared the performance of classical and deep learning models.

### Results Interpretation

Error Analysis: Visualized confusion matrices for classical models.

Feature Importance: Identified top features influencing Random Forest predictions.

Embedding Visualization: Reduced dimensions of LSTM embeddings using PCA.

### Streamlit Application

Real-Time Prediction: Predicts star ratings and main subjects of reviews.

Explanation: Provides insights into model predictions.

Information Retrieval: Placeholder for Retrieval-Augmented Generation (RAG) and Question Answering (QA).