import feedparser
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
import os

import pandas as pd

from sqlalchemy import create_engine

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
  logger.info("Inside Handler")

  df = pd.read_csv("EQUITY_L.csv")
  nlp = spacy.load("en_core_web_sm")

  rss = "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"
  finbert_model = "ProsusAI/finbert"

  tokenizer = AutoTokenizer.from_pretrained(finbert_model)
  model = AutoModelForSequenceClassification.from_pretrained(finbert_model)
  sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

  feed = feedparser.parse(rss)
  articles = []

  for entry in feed.entries:
    article = {}
    pub_tm = datetime(*entry.published_parsed[:6])
    article['source'] = 'economictimes'
    article['title'] = entry.title
    article['description'] = entry.description
    article['publish_datetime'] = pub_tm
    articles.append(article)


  def analyze_sentiment(item):
      results = []
      out = sentiment_pipeline(item)[0]
      return out['label']

  for article in articles:
    news_text = article['title'] + ' '+ article['description']
    sentiment = analyze_sentiment(news_text)
    article['sentiment'] = sentiment
    symbol = 'General'
    doc = nlp(news_text)
    stock_names = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            stock_names.append(ent.text)
    print(stock_names)
    if len(stock_names) != 0:
      symbol = df.loc[
      (df['NAME OF COMPANY'].str.contains(stock_names[0], case=False, na=False)) |
      (df['SYMBOL'].str.contains(stock_names[0], case=False, na=False)),
      'SYMBOL'
      ]
      if not symbol.empty:
        symbol = symbol.iloc[0]
      else:
        symbol = 'General'
    article['symbol'] = symbol

  print(articles)
  df_final = pd.DataFrame(articles)
  db_user = os.environ.get("DB_USERNAME")
  db_password = os.environ.get("DB_PASSWORD")
  db_host = 'database-1.cs9ycq6ishdm.us-east-1.rds.amazonaws.com'
  db_port = '5432'
  db_name = 'capstone_project'

  # 3. Create a SQLAlchemy engine
  # The 'postgresql+psycopg2://' dialect specifies the use of psycopg2
  engine = create_engine(f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

  # 4. Write the DataFrame to a PostgreSQL table
  table_name = 'news_sentiment'

  df_final.to_sql(table_name, engine, if_exists='append', index=False)

  