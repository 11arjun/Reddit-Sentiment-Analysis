import numpy as np
import ssl
import pandas as pd
from keras import Model
from keras.src.layers import Lambda
from keras.src.optimizers import Adam
from nltk.corpus import stopwords
import emoji
import nltk
import contractions
import re
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from transformers import RobertaTokenizer
from transformers import TFRobertaModel
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.metrics import classification_report, precision_score, recall_score
import torch

# Let's load the labeled data
Datas = pd.read_csv('labeledSentiments.csv', delimiter='\t', header=None, names=['cleanText', 'sentiment'])
Data = Datas.sample(frac=0.001, random_state=42)
# Ensuring all values in cleanText are strings
Data['cleanText'] = Data['cleanText'].astype(str)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fitting  the label encoder and transforming the sentiment labels to numeric values
Data['sentiment'] = label_encoder.fit_transform(Data['sentiment'])

# Initializing the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Function to tokenize the text
def tokenize_text(text):
    return tokenizer(text, truncation=True, padding='max_length', max_length=512)

# Apply the tokenizer to your cleanText column
Data['tokenized'] = Data['cleanText'].apply(tokenize_text)
# Extract input IDs and attention masks from the tokenized output
Data['input_ids'] = Data['tokenized'].apply(lambda x: x['input_ids'])
Data['attention_mask'] = Data['tokenized'].apply(lambda x: x['attention_mask'])
    

