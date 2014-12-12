import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from gensim import matutils
from gensim.models import ldamulticore, ldamodel


class ColumnExtractor(BaseEstimator, TransformerMixin):
    datatype = None

    def __init__(self, columns=[], datatype=None):
        self.columns = columns
        self.datatype = datatype

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def transform(self, X, **transform_params):
        Xt = X[self.columns]
        if self.datatype:
            return Xt.astype(self.datatype)
        else:
            return Xt

    def fit(self, X, y=None, **fit_params):
        return self

class TruncateLDA(BaseEstimator, TransformerMixin):

    def __init__(self, num_topics=5):
        self.num_topics = num_topics

    def fit_transform(self, X, y=None):
        Xt = matutils.Sparse2Corpus(X, documents_columns=False)
        model = ldamodel.LdaModel(Xt, num_topics=self.num_topics)
        #model = ldamulticore.LdaMulticore(Xt, num_topics=self.num_topics)
        self.components_= model
        return self.get_doc_topic(Xt, model)


    def transform(self, X):
        t = matutils.Sparse2Corpus(X, documents_columns=False)
        inference = np.asarray(self.components_.inference(t)[0])
        return inference.astype(np.float64)

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def get_doc_topic(self, corpus, model):
        raw_docs = [model.__getitem__(doc, eps=0) for doc in corpus]
        doc_topic = np.asarray(raw_docs)[:,:,1]
        return doc_topic.astype(np.float64)

# class OverSample(BaseEstimator, TransformerMixin):
#     def __init__(self, k=5):
#         self.k = k
    
#     def fit_transform(self, X, y=None):
#         #balance set
#         needed_percentage_sampling = 0.5*100/(sum(y)/len(y))
#         #rounding needed
#         needed_percentage_sampling = round(needed_percentage_sampling/100)*100
#         X_minority, y_minority = 0
#         additional_samples = SMOTE([X_minority, y_minority], needed_percentage_sampling, self.k)

#         #join additional samples.
#         return X
#     def transform(self, X):
#         return X
    
#     def fit(self, X, y=None):
#         self.fit_transform(X)
#         return self

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

