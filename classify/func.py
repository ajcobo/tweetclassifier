from plot import *
from util import *
import numpy as np
import pandas as pa
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import train_test_split, LeavePOut
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics, cross_validation, grid_search, preprocessing
from sklearn.utils import check_arrays
from sklearn.decomposition import PCA, TruncatedSVD
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
import datetime
from pprint import pprint
import string
import UnbalancedDataset as UD

#Regexp
import re

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

# TRAINING

def split_dataset(dataset, noiseset=None, noise_proportion= 0.0, noise_train=False, noise_test=False):
    #Test and Training
    X, y = dataset
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    check_arrays(X,y)
    train_length = int(len(X)*0.8)
    X_base_train, X_base_test, y_base_train, y_base_test = X[:train_length], X[train_length:], y[:train_length], y[train_length:]

    # Balance
    if noiseset:
        if noise_train:
            X_base_train, y_base_train, noiseset = join_datasets_by_proportion([X_base_train,y_base_train], noiseset, noise_proportion)
        if noise_test:
            X_base_test, y_base_test, noiseset = join_datasets_by_proportion([X_base_test,y_base_test], noiseset, noise_proportion)

    return(X_base_train, X_base_test, y_base_train, y_base_test)

def join_datasets_by_proportion(dataset, noiseset, noise_proportion):
    #Proportion of noise
    noise_max_index = int(len(dataset[0])*noise_proportion/(1-noise_proportion))
    noiseset = noiseset[0][:noise_max_index], noiseset[1][:noise_max_index]
    remaining_noise = noiseset[0][noise_max_index:], noiseset[1][noise_max_index:]

    #Combine datasets
    #noise_train_length = int(len(noiseset[0])*train_proportion)
    #dataset_train_length = int(len(dataset[0])*train_proportion)
    X = np.concatenate([dataset[0],noiseset[0]])
    y = np.concatenate([dataset[1],noiseset[1]])

    finalset = X, y, remaining_noise

    return finalset
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

def train_text(model, dataset, text, save):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    #Text vectorization using text processing
    vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')

    #Learn vocabulary and idf, return term-document matrix.
    #Extracting features from the training dataset using a sparse vectorizer
    X_train_vectorized = vectorizer.fit_transform(X_train)
    #Transform documents to document-term matrix.
    #Extracting features from the test dataset using the same vectorizer
    X_test_vectorized = vectorizer.transform(X_test)

    resulting_model = model.fit(X_train_vectorized,y_train)
    prediction = resulting_model.predict(X_test_vectorized)

    # Output
    print_report(X_test, y_test, resulting_model, prediction, text, save)

def train_text_pca(model, dataset, n_components = 100):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    #Text vectorization using text processing
    vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')

    #Learn vocabulary and idf, return term-document matrix.
    #Extracting features from the training dataset using a sparse vectorizer
    X_train_vectorized = vectorizer.fit_transform(X_train)
    pca = PCA(n_components)
    fit_model = pca.fit(X_train_vectorized)
    X_train_vectorized = fit_model.transform(X_train_vectorized)
    #Transform documents to document-term matrix.
    #Extracting features from the test dataset using the same vectorizer
    X_test_vectorized = vectorizer.transform(X_test).toarray()
    X_test_vectorized = fit_model.transform(X_test_vectorized)

    resulting_model = model.fit(X_train_vectorized,y_train)
    prediction = resulting_model.predict(X_test_vectorized)

    # Output
    print_report(X_test_vectorized, y_test, resulting_model, prediction)

def train_text_lsa(model, dataset, n_components = 100):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    #Text vectorization using text processing
    vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')

    #Learn vocabulary and idf, return term-document matrix.
    #SVD
    X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
    svd = TruncatedSVD(n_components)
    fit_model = svd.fit(X_train_vectorized)
    X_train_vectorized = fit_model.transform(X_train_vectorized)
    #Transform documents to document-term matrix.
    #Extracting features from the test dataset using the same vectorizer
    X_test_vectorized = vectorizer.transform(X_test).toarray()
    X_test_vectorized = fit_model.transform(X_test_vectorized)

    resulting_model = model.fit(X_train_vectorized,y_train)
    prediction = resulting_model.predict(X_test_vectorized)

    # Output
    print_report(X_test_vectorized, y_test, resulting_model, prediction)

def grid_search_lsa(model, dataset, parameters, description=""):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    #Text vectorization using text processing
    vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')

    text_clf = Pipeline([('extract', ColumnExtractor([...,0])),
                         ('vectorizer', vectorizer),
                         ('reduce_dim', TruncatedSVD()),
                         ('clf', model)])

    # n_jobs = -1 just in executable
    gs = grid_search.GridSearchCV(text_clf, parameters)

    resulting_model = gs.fit(X_train,y_train)
    prediction = resulting_model.predict(X_test)

    # Output
    print(resulting_model.best_params_)
    print(metrics.classification_report(y_test, prediction))
    print_report(X_test, y_test, resulting_model, prediction)

def grid_search_pca(model, dataset, parameters):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    #Text vectorization using text processing
    vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')

    text_clf = Pipeline([('vectorizer', vectorizer),
                         ('reduce_dim', PCA()),
                         ('clf', model)])

    # n_jobs = -1 just in executable
    gs = grid_search.GridSearchCV(text_clf, parameters)
    
    resulting_model = gs.fit(X_train,y_train)
    prediction = resulting_model.predict(X_test)

    # Output
    print(resulting_model.best_params_)
    print(metrics.classification_report(y_test, prediction))
    # print_report(X_test, y_test, resulting_model, prediction)

def grid_search_with_param(params):
    X_train, X_test, y_train, y_test = split_dataset(params.dataset, params.noiseset, params.noise_proportion, params.noise_train, params.noise_test)


    #balancing features
    X_train, y_train, resulting_pipeline = adjust_features(X_train, y_train, n_components=params.n_components)

    X_test = resulting_pipeline.transform(X_test)


    text_clf = model_pipeline(params.model)

    gs = grid_search.GridSearchCV(text_clf, params.parameters, cv=params.folds, n_jobs=params.n_jobs, verbose=params.verbose)
    resulting_model = gs.fit(X_train,y_train)
    prediction = resulting_model.predict(X_test)

    # Output
    print_report(X_test, y_test, resulting_model, prediction, params.text, params.save)

def train_fixed_param(params):
    X_train, X_test, y_train, y_test = split_dataset(params.dataset, params.noiseset, params.noise_proportion, params.noise_train, params.noise_test)

    text_clf = main_pipeline(params.model, params.n_components)

    #gs = grid_search.GridSearchCV(text_clf, parameters)
    resulting_model = text_clf.fit(X_train,y_train)
    prediction = resulting_model.predict(X_test)

    # Output
    #print(resulting_model.best_params_)
    #print(metrics.classification_report(y_test, prediction))
    print_report(X_test, y_test, resulting_model, prediction, params.text, params.save)

def train_text_fixed_param(model, dataset, text, n_components=100, save = False):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    text_clf = text_pipeline(model, n_components)

    #gs = grid_search.GridSearchCV(text_clf, parameters)
    resulting_model = text_clf.fit(X_train,y_train)
    prediction = resulting_model.predict(X_test)

    # Output
    #print(resulting_model.best_params_)
    #print(metrics.classification_report(y_test, prediction))
    print_single_report(X_test, y_test, resulting_model, prediction, text, save)

def train_notext(model, dataset):
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    resulting_model = model.fit(X_train, y_train)
    prediction = resulting_model.predict(X_test)

    # Output
    print_report(X_test, y_test, resulting_model, prediction, text)

# CROSS VALIDATION
def cross_val_train(model, dataset, nfolds, scoring, title = "", n_components=100, save = False):
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    #scores = cross_validation.cross_val_score(model, X_train, y_train, cv = folds, score_func=scoring)
    kf = cross_validation.KFold(len(X_train), nfolds)
    text_clf = main_pipeline(model, n_components=100)
    true_scores, false_scores, roc = custom_cross_val_score(text_clf, X_train, y_train, kf)
    #print(cross_validation.cross_val_score(model, X_train,y_train,cv=kf,scoring='accuracy'))
    result = np.array([('True',)+true_scores, ('False',)+false_scores], dtype=[('class', 'U5'), ('precision', 'float'), ('recall', 'float'), ('fscore', 'float')])
    print_cross_val_report(result, roc, title, save)

def custom_cross_val_score(model, X, y, kfolds):
    cm = np.zeros(len(np.unique(y)) ** 2)
    roc = 0
    substract_folds = 0
    for i, (train, test) in enumerate(kfolds):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        cm += metrics.confusion_matrix(y[test], y_pred).flatten()
        report = metrics.classification_report(y[test], y_pred)
        new_roc = 0
        unique = np.unique(y[test])
        # Just one label of False Values does not make sense
        if len(unique) == 1 and unique[0] is np.False_:
            substract_folds += 1
        else:
            new_roc = compute_roc(y[test], y_pred)
            roc += new_roc
        print(report)
        print("ROC: ", new_roc)
    return compute_measures(*cm / kfolds.n_folds), compute_negative_measures(*cm / kfolds.n_folds), roc / (kfolds.n_folds-substract_folds)

def compute_roc(test, score):
    fpr, tpr, _ = metrics.roc_curve(test, score)
    return metrics.auc(fpr, tpr)

def compute_measures(tp, fp, fn, tn):
    """Computes effectiveness measures given a confusion matrix."""
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = 2 * tp / (2 * tp + fp + fn)
    return precision, recall, fmeasure

def compute_negative_measures(tp, fp, fn, tn):
    precision = tn / (tn + fn)
    recall = tn / (tn + fp)
    fmeasure = 2 * tn / (2 * tn + fn + fp)
    return precision, recall, fmeasure

def fixed_pipeline(model, n_components):
    #Text vectorization using text processing
    vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')

    pipeline = Pipeline([('features', FeatureUnion([
                        ('text', Pipeline([
                            ('extract', ColumnExtractor([...,0])),
                            ('vectorize', vectorizer),
                            ('reduce_dim', TruncatedLDA(n_components = 100))
                        ])),
                        ('no_text', Pipeline([
                            ('extract', ColumnExtractor([...,slice(1,None,None)], datatype=np.float64))
                        ]))
                    ])),
                    ('classifier', model)])
    return pipeline

def main_pipeline(model, n_components=100, ratio=1):
    #Text vectorization using text processing
    vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')

    pipeline = Pipeline([('features', FeatureUnion([
                            ('text', Pipeline([
                                ('extract', ColumnExtractor([...,0])),
                                ('vectorize', vectorizer),
                                ('reduce_dim', TruncateLDA(num_topics = n_components))
                            ])),
                            ('no_text', Pipeline([
                                ('extract', ColumnExtractor([...,slice(1,None,None)], datatype=np.float64))
                            ]))
                        ])),
                        ('classifier', model)])
    return pipeline

def adjust_features(X, y, n_components=100):
    #Text vectorization using text processing
    vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')

    pipeline = FeatureUnion([
                            ('text', Pipeline([
                                ('extract', ColumnExtractor([...,0])),
                                ('vectorize', vectorizer),
                                ('reduce_dim', TruncateLDA(num_topics = n_components))
                            ])),
                            ('no_text', Pipeline([
                                ('extract', ColumnExtractor([...,slice(1,None,None)], datatype=np.float64))
                            ]))
                        ])
    result_pipeline = pipeline.fit(X,y)

    X = result_pipeline.transform(X)
    #raction of the number of minority samples to synthetically generate.
    # For 0.5
    ratio = (2*(len(y)-sum(y)) - len(y))/sum(y)

    balance_model = UD.bSMOTE1(ratio=ratio)

    X,y = balance_model.fit_transform(X,y)

    return X, y, result_pipeline


def model_pipeline(model):
    pipeline = Pipeline([('classifier', model)])
    return pipeline

def text_pipeline(model, n_components=100):
    #Text vectorization using text processing
    vectorizer = TfidfVectorizer(tokenizer=process_text, stop_words=stopwords.words('spanish'), min_df=1, lowercase=True, strip_accents='unicode')

    pipeline = Pipeline([('features', FeatureUnion([
                            ('text', Pipeline([
                                ('vectorize', vectorizer)
                            ]))
                        ])),
                        ('classifier', model)])
    return pipeline



