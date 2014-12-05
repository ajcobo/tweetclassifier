import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from gensim import matutils
from gensim.models import ldamulticore, ldamodel

class ColumnExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, columns=[]):
        self.columns = columns

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def transform(self, X, **transform_params):
        return X[self.columns]

    def fit(self, X, y=None, **fit_params):
        return self

class TruncateLDA(BaseEstimator, TransformerMixin):

    def __init__(self, num_topics=5):
        self.num_topics = num_topics

    def fit_transform(self, X, y=None):
        Xt = matutils.Sparse2Corpus(X, documents_columns=False)
        #model = ldamodel.LdaModel(Xt, num_topics=self.num_topics)
        model = ldamulticore.LdaMulticore(Xt, num_topics=self.num_topics)
        self.components_= model
        return self.get_doc_topic(Xt, model)


    def transform(self, X):
        t = matutils.Sparse2Corpus(X, documents_columns=False)
        inference = np.asarray(self.components_.inference(t)[0])
        return inference

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def get_doc_topic(self, corpus, model):
        raw_docs = [model.__getitem__(doc, eps=0) for doc in corpus]
        doc_topic = np.asarray(raw_docs)[:,:,1]
        return doc_topic