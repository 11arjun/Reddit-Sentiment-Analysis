import re
import ssl

import contractions
import nltk
import pandas as pd
import emoji
import numpy as np
from nltk.corpus import stopwords
from textblob import TextBlob
from transformers import RobertaTokenizer
#  Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
# Download the stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# Loading the data
Data = pd.read_csv('reddit_comments_latests.csv' , index_col=False)
Data.dropna(inplace=True)
# Taking .1 % datas from the set
# Data = Datas.sample(frac=0.001, random_state=42)
# Let's check for the missing values now
missing_values = Data.isnull().sum()
# Since There are now missing values moving ahead
def cleanText(dataClean):
    # Remove numbers
    if isinstance(dataClean, str):

        dataClean = re.sub(r'\d+', '', dataClean)
        # Convert emojis to text
        dataClean = emoji.demojize(dataClean)
        # Remove punctuations except for hashtags
        dataClean = re.sub(r'[^\w\s#]', '', dataClean)
        # Remove extra spaces
        dataClean = dataClean.strip()
        # Now lets use stop words to remove
        dataClean = ' ' .join([word for word in dataClean.split() if word.lower() not in stop_words])
        # Define a regex pattern to match web links (http, https, www)
        url_pattern = r'http[s]?://\S+|www\.\S+'
        # Define a regex pattern to match HTML links
        html_pattern = r'<a\s+(?:[^>]*?\s+)?href=(["\'])(.*?)\1'
        # Remove web links
        dataClean = re.sub(url_pattern, '', dataClean)
        # Remove HTML links
        dataClean = re.sub(html_pattern, '', dataClean)
        # Let' remove hashtags but keep the words
        dataClean = re.sub(r'#(\w+)', r'\1', dataClean)
        # Remove dashes
        dataClean = re.sub(r'-{1,}', '', dataClean)
        # Let's use contractions
        dataClean = contractions.fix(dataClean)
        # Let's correct the spelling using Textblob
        dataClean = str(TextBlob(dataClean).correct())
        return dataClean

# Let's apply cleanText Function
Data['cleanText'] = Data['Comment'].apply(cleanText)


# Now Let's apply textblob to get the sentiments.
def sentimentAnalysis(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"
# Let's Apply this in a dataset now.
Data['sentiment'] = Data['cleanText'].apply(sentimentAnalysis)
# Creating a DataFrame with desire columns
labeledData = Data[['cleanText', 'sentiment']]
labeledData.to_csv('labeledSentiments.csv' , index= False , sep='\t')


