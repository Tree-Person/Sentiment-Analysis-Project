import subprocess
import sys
import pandas as pd

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('gnews')
install('pandas')
install('transformers')
install('scipy')
install('torch')

from gnews import GNews
from datetime import datetime, timedelta
from gnews import GNews

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

google_news = GNews()

def find_highest_element(lst):
    max_ele = max(lst)
    for i , x in enumerate(lst):
        if x == max_ele:
            return max_ele , i

def data_results(df):
    df = df.reset_index().rename(columns={'index': 'Id'})
    df['Sentiment'] = df['title'].apply(sent)
    return df

def sent(text):
    encoded_example = tokenizer(text, return_tensors='pt')
    output = model(**encoded_example)
    scores = softmax(output[0][0].detach().numpy())
    results = find_highest_element(scores)
    if results[1] == [0]:
        return -(results[0])
    elif results[1] == [1]:
        return results[0]
    else:
        return results[0] + 1

def data(topic, i):
    google_news.period = '7d' 
    articles = google_news.get_news(topic)
    data_df = pd.DataFrame(articles)
    data_df['Sentiment'] = data_df['title'].apply(sent)
    data_df.rename(columns={'published date': f'week_{i}_date'}, inplace=True)
    data_df.reset_index(inplace=True)
    data_df.rename(columns={'index': 'Id'}, inplace=True)
    return data_df

def sentiment(topic):
    i = 1
    week_data = data(topic, i )
    week_data.rename(columns={'title': f'Article', 'Sentiment': f'Score', 'published date': f'Date'}, inplace=True)
    week_data.drop(['description', 'publisher', 'url'], axis=1, inplace=True)
    return week_data
