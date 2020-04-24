# import libraries
import numpy as p
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
  df = pd.read_csv('emails.csv')

  # get the total rows & columsn
  #df.shape

  # first & tail row
  # df.head(5) 
  # df.tail(5)

  # get column names
  #df.columns

  # remove duplicates
  df.drop_duplicates(inplace = True)

  # show the number missing data 
  #df.isnull.sum()
  
  # processing the text 
  # df['text'].head().apply(clean_up_text)

  # convert a collection to tokens matrix
  print("Cleaning up text...")
  messages = CountVectorizer(analyzer=clean_up_text).fit_transform(df['text'])

  # split the data 80% training and 20 % testing
  print("Spliting training and testing datasets...")
  x_train, x_test, y_train, y_test = train_test_split(messages, df['spam'], test_size=0.20, random_state=0)

  # naive bayes classifier
  classifier = MultinomialNB().fit(x_train, y_train)

  # evaluate model on training datasets
  print("Evaluating datasets...")
  prediction = classifier.predict(x_train)

  classification_report(y_train, prediction)
  confusion_matrix(y_train, prediction)
  print('Accuracy score: ', accuracy_score(y_train, prediction))

  # Testing datasets
  # the prediction
  test_prediction = classifier.predict(x_test)
  print(test_prediction)
  # the actual values
  print(y_test.values)

  classification_report(y_test, test_prediction)
  confusion_matrix(y_test, test_prediction)
  print('Accuracy score: ', accuracy_score(y_test, test_prediction))

def clean_up_text(text):
  # remove punctuation
  without_punc = [char for char in text if char not in string.punctuation]
  without_punc = ''.join(without_punc)

  # remove stopwords
  clean_words = [word for word in without_punc.split() if word.lower not in stopwords.words('english')]

  # return text words  
  return clean_words

main()



