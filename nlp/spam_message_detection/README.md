# Using NLP to detect spam text messages with Python


In this project we will be working with UCI's SMS Spam Collection Data Set.
[UCI datasets](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

The data contains a collection of more than 5 thousand SMS phone messages. You can check out the readme file for more info.
The data is at **data\smsspamcollection**

With the help of these instances of ham and spam that have been tagged, we'll train a machine learning model to automatically distinguish between the two. We will then be able to categorize random unlabeled communications as spam or ham using a trained model.

## Objectives
### Analyse the data and get below insights
- Perform Expolatory data analysis and try ti identify the features for Feature engineering. 
- Text pre-processing: clean the text by removing punctuation, stopwords.
- Text Normalization
- Vectorization using SciKit Learn's CountVectorizer
- Using Bag of words model and calcualting the sparsity of the Sparse matrix.
- Calcualting the TF-IDF using scikit-learn's TfidfTransformer.
- Using Naive Bayes classifier algorithm for Model traning.
- Performing train test split and using classification reports to evaluate out model.

## Tech Stack
- **Python modlues**: pandas, numpy, sklearn.nltk and matplotlib, and downloading the corpus for stopwords.
- **SK Learn Modlues**: TfidfTransformer, train_test_split, metrics. 
- **Plots**: plot(), histogram
