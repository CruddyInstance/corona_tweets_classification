# Text Classification using DistilBERT on COVID-19 Tweets

This repository showcases a text classification project that utilizes the DistilBERT model to analyze sentiment in a dataset of COVID-19 related tweets.

## Dataset Overview

The dataset contains tweets related to COVID-19, and it comprises the following columns:

- **Location:** The geographical location of the tweet author.
- **Tweet At:** The timestamp indicating when the tweet was posted.
- **Original Tweet:** The text content of the tweet.
- **Label:** The sentiment label assigned to the tweet.

Manual tagging has been performed to categorize the sentiment of each tweet, ensuring data accuracy and relevance. To respect privacy, the dataset utilizes codes instead of real names and usernames.

## Project Highlights

- **Data Exploration:** The dataset is loaded and explored using the powerful data manipulation libraries of Python, such as Pandas.

- **Text Preprocessing:** The raw text data is cleaned, preprocessed, and prepared for further analysis. This step involves tokenization, handling missing values, and potentially stemming or lemmatization.

- **DistilBERT Integration:** The project employs PyTorch and the DistilBERT model for sequence classification. DistilBERT's pre-trained embeddings capture contextual information from the text.

- **Model Training:** The dataset is divided into training and testing sets. The DistilBERT model is fine-tuned on the training data, adjusting its parameters to suit the specific sentiment classification task.

- **Model Evaluation:** The trained model is evaluated on the testing set using metrics such as accuracy, precision, recall, and F1-score to quantify its performance.

- **Inference:** The trained model is utilized to predict the sentiment of new or unseen tweets, enabling real-world application.

## DistilBERT and PyTorch

DistilBERT, developed by Hugging Face, is a highly efficient version of BERT designed for faster training and reduced memory consumption. PyTorch, a widely used deep learning framework, is employed for model implementation and training.

## Final Accuracy

The model achieved an impressive accuracy of 84% on the test dataset, demonstrating its effectiveness in sentiment classification.

## Acknowledgements

The dataset originates from Kaggle and has been meticulously tagged for sentiment analysis.

