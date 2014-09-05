import numpy as np
import pandas as pa
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

import datetime
from pprint import pprint
import string

# http://bogdan-ivanov.com/entry/recipe-text-clustering-using-nltk-and-scikit-learn/
# Used to tokenize when constucting the vectorizer
def process_text(text, stem=True):
  puntuacion = string.punctuation + '…–«»“”¡¿´¨‘'
  remove_punct_map = dict.fromkeys(map(ord, puntuacion))
  text = text.translate(remove_punct_map)
  remove_digits_map = dict.fromkeys(map(ord, string.digits))
  text = text.translate(remove_digits_map)
  text = ' '.join(text.split())
  # obtain tokens 
  tokens = word_tokenize(text)
 
  if stem:
    stemmer = SnowballStemmer("spanish")
    tokens = [stemmer.stem(t) for t in tokens]
    #tokens = text.apply(stemmer.stem)

  return tokens

def train(model, dataset):
  #Test and Training
  X, y = dataset
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  #Text vectorization using text processing
  vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')
  #tmp = pa.DataFrame.as_matrix(X_train)

  #Learn vocabulary and idf, return term-document matrix.
  #Extracting features from the training dataset using a sparse vectorizer
  X_train_vectorized = vectorizer.fit_transform(X_train)
  #Transform documents to document-term matrix.
  #Extracting features from the test dataset using the same vectorizer
  X_test_vectorized = vectorizer.transform(X_test)

  resulting_model = model.fit(X_train_vectorized,y_train)
  #a.score(X_test_d,y_test)
  prediction = resulting_model.predict(X_test_vectorized)
  score = metrics.f1_score(y_test, prediction)
  print(vectorizer.get_feature_names())
  print(vectorizer.idf_)
  print(score)
  #print(prediction)

#Execute
raw_data = pa.read_csv("/home/alf/Dropbox/Master/AC/Ground Truth/Consolidated/GroundTruthTotal.csv")
dataset = DataFrameMapper(raw_data)

times = pa.to_datetime(raw_data.date +" "+raw_data.time)
earthquake_time = datetime.datetime(2010, 2, 27, 3,34,8)
time_gap = times - earthquake_time

#Add time_gap for the features
dataset.features['time_gap'] = time_gap

#Extract each feature set
columns_feature_1 = ['texto', 'time_gap', 'followers_count', 'friends_count']
columns_feature_2 = ['texto', 'followers_count', 'friends_count']
columns_feature_3 = ['texto', 'time_gap', 'friends_count']
columns_feature_4 = ['texto', 'time_gap', 'followers_count']

feature_1 = [dataset.features[columns_feature_1], dataset.features['value']]
feature_2 = [dataset.features[columns_feature_2], dataset.features['value']]
feature_3 = [dataset.features[columns_feature_3], dataset.features['value']]
feature_4 = [dataset.features[columns_feature_4]['texto'], dataset.features['value']]
feature_5 = [dataset.features['texto'], dataset.features['value']]

test_feature = [['esto es una prueba', 'una prueba dada por arreglo', 'me gusta este arreglo'], np.array([False, True, True], dtype=bool)]

train(svm.SVC(), test_feature)
train(svm.SVC(kernel='linear'), feature_5)
train(MultinomialNB(), feature_5)