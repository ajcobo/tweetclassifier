import numpy as np
import pandas as pa
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split, LeavePOut
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils import check_arrays
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
import datetime
from pprint import pprint
import string

#Regexp
import re

#Plots
import matplotlib.pyplot as plt

#tokenizer http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py
emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

# The components of the tokenizer:
regex_strings = (
    # Phone numbers:
    r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""
    ,
    # Emoticons:
    emoticon_string
    ,    
    # HTML tags:
     r"""<[^>]+>"""
    ,
    # Twitter username:
    r"""(?:@[\w_]+)"""
    ,
    # Twitter hashtags:
    r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
    )

url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# http://bogdan-ivanov.com/entry/recipe-text-clustering-using-nltk-and-scikit-learn/
# Used to tokenize when constucting the vectorizer
def process_text(text, stem=True, remove_links=True, punctuation_exception='#@'):
    puntuacion = string.punctuation + '…–«»“”¡¿´¨‘'
    #Remove exceptions
    exceptions_table = str.maketrans('','',punctuation_exception)
    filtered_punctiation = puntuacion.translate(exceptions_table)
    # Remove links
    if remove_links:
        text = re.sub(url_regex, '', text)
    #Remove punctuation
    remove_punct_map = dict.fromkeys(map(ord, filtered_punctiation))
    text = text.translate(remove_punct_map)
    remove_digits_map = dict.fromkeys(map(ord, string.digits))
    text = text.translate(remove_digits_map)
    #Remove extra spaces
    text = ' '.join(text.split())
    # obtain tokens 
    tokens = word_re.findall(text)
 
    if stem:
        stemmer = SnowballStemmer("spanish")
        tokens = [stemmer.stem(t) for t in tokens]
        #tokens = text.apply(stemmer.stem)

    return tokens

def plot_roc(test, score):
    #ROC curve and ROC area
    fpr, tpr, _ = metrics.roc_curve(test, score)
    roc_auc = metrics.auc(fpr, tpr)

    # Plot of a ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def split_dataset(dataset):
    #Test and Training
    X, y = dataset
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    check_arrays(X,y)
    train_length = int(len(X)*0.8)
    return(X[:train_length], X[train_length:], y[:train_length], y[train_length:])

def train_text(model, dataset):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

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

    # Output
    # print(vectorizer.get_feature_names())
    # print(vectorizer.idf_)
    # print(len(vectorizer.idf_))
    # print(score)
    #print(prediction)
    prfs = metrics.classification_report(y_test, prediction)
    print(prfs)
    plot_roc(y_test, resulting_model.decision_function(X_test_vectorized))

def train_notext(model, dataset):
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    resulting_model = model.fit(X_train, y_train)
    
    prediction = resulting_model.predict(X_test)
    score = metrics.f1_score(y_test, prediction)

    # Output
    # print(vectorizer.get_feature_names())
    # print(vectorizer.idf_)
    # print(len(vectorizer.idf_))
    # print(score)
    #print(prediction)
    prfs = metrics.classification_report(y_test, prediction)
    print(prfs)
    plot_roc(y_test, resulting_model.decision_function(X_test))

#Execute
raw_data = pa.read_csv("/home/alf/Dropbox/Master/AC/Ground Truth/Consolidated/GroundTruthTotal.csv")
dataset = DataFrameMapper(raw_data)

times = pa.to_datetime(raw_data.date +" "+raw_data.time)
earthquake_time = datetime.datetime(2010, 2, 27, 3,34,8)
time_gap = times - earthquake_time

#order by date
dataset.features = dataset.features.sort(['date', 'time'])

#Add time_gap for the features
dataset.features['time_gap'] = time_gap

#Extract each feature set
columns_feature_1 = ['texto', 'time_gap', 'followers_count', 'friends_count']
columns_feature_2 = ['texto', 'followers_count', 'friends_count']
columns_feature_3 = ['texto', 'time_gap', 'friends_count']
columns_feature_4 = ['texto', 'time_gap', 'followers_count']
columns_feature_5 = ['time_gap', 'followers_count', 'friends_count']

feature_1 = [dataset.features[columns_feature_1], dataset.features['value']]
feature_2 = [dataset.features[columns_feature_2], dataset.features['value']]
feature_3 = [dataset.features[columns_feature_3], dataset.features['value']]
feature_4 = [dataset.features[columns_feature_4]['texto'], dataset.features['value']]
total_text = [dataset.features['texto'], dataset.features['value']]
total_notext = [dataset.features[columns_feature_5], dataset.features['value']]

test_feature = [['esto es una prueba', 'una prueba dada por arreglo', 'me gusta este arreglo'], np.array([False, True, True], dtype=bool)]

#train(svm.SVC(), test_feature)
train_text(svm.LinearSVC(), total_text)
train_text(MultinomialNB(), total_text)
train_notext(svm.LinearSVC(), total_notext)
train_text(MultinomialNB(), total_notext)