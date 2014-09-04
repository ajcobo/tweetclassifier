import numpy as np
import pandas as pa
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn import svm
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

feature_1_data, feature_1_target = dataset.features[columns_feature_1], dataset.features['value']
feature_2_data, feature_2_target = dataset.features[columns_feature_2], dataset.features['value']
feature_3_data, feature_3_target = dataset.features[columns_feature_3], dataset.features['value']
feature_4_data, feature_4_target = dataset.features[columns_feature_4], dataset.features['value']
feature_5_data, feature_5_target = dataset.features['texto'], dataset.features['value']

#Test and Training
X_train, X_test, y_train, y_test = train_test_split(feature_5_data, feature_5_target, test_size=0.2)

#Text vectorization using text processing
vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')
#tmp = pa.DataFrame.as_matrix(X_train)

X_train = vectorizer.fit_transform(X_train)
y_train = vectorizer.fit_transform(y_train)



a = svm.SVC().fit(X_train,y_train)
a.score(X_test,y_test)