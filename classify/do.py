from load import *
from clean import *
from func import *
from util import *

from sklearn import svm, ensemble, linear_model, naive_bayes

#Load Dataset
earthquake_time = datetime.datetime(2010, 2, 27, 3, 34, 8)
dataset =  extract_features_with_timegap("/home/alf/Dropbox/Master/AC/Ground Truth/Consolidated/GroundTruthTotal.csv", earthquake_time)

#Load Noise Dataset
base_first_date = pa.to_datetime(dataset.features["date"][0] +" "+dataset.features["time"][0])
noiseset = extract_noise_features_with_timegap("/home/alf/Dropbox/Master/AC/Noise/NoiseSet.csv",base_first_date, earthquake_time)

#Extract each feature set
columns_feature_1 = ['texto', 'time_gap', 'followers_count', 'friends_count']
columns_feature_2 = ['texto', 'followers_count', 'friends_count']
columns_feature_3 = ['texto', 'time_gap', 'friends_count']
columns_feature_4 = ['texto', 'time_gap', 'followers_count']
columns_feature_5 = ['time_gap', 'followers_count', 'friends_count']

columns_noise_feature_1 = ['text', 'time_gap', 'followers_count', 'friends_count']
columns_noise_feature_2 = ['text', 'followers_count', 'friends_count']
columns_noise_feature_3 = ['text', 'time_gap', 'friends_count']
columns_noise_feature_4 = ['text', 'time_gap', 'followers_count']
columns_noise_feature_5 = ['time_gap', 'followers_count', 'friends_count']

#does it work?
dataset.features['time_gap'] = dataset.features['time_gap'].astype('timedelta64[h]')
noiseset.features['time_gap'] = noiseset.features['time_gap'].astype('timedelta64[h]')

feature_1 = [dataset.features[columns_feature_1].values, dataset.features['value'].values]
feature_2 = [dataset.features[columns_feature_2].values, dataset.features['value'].values]
feature_3 = [dataset.features[columns_feature_3].values, dataset.features['value'].values]
feature_4 = [dataset.features[columns_feature_4].values, dataset.features['value'].values]
total_text = [dataset.features['texto'].values, dataset.features['value'].values]
total_notext = [dataset.features[columns_feature_5].values, dataset.features['value'].values]

noise_feature_1 = [noiseset.features[columns_noise_feature_1].values, noiseset.features['value'].values]
noise_feature_2 = [noiseset.features[columns_noise_feature_2].values, noiseset.features['value'].values]
noise_feature_3 = [noiseset.features[columns_noise_feature_3].values, noiseset.features['value'].values]
noise_feature_4 = [noiseset.features[columns_noise_feature_4].values, noiseset.features['value'].values]
noise_text = [noiseset.features['text'].values, noiseset.features['value'].values]

# Perform analysis

# #train(svm.SVC(), test_feature)
#train_text(svm.LinearSVC(), total_text)
# #train_text(MultinomialNB(), total_text)
# train_notext(ensemble.RandomForestClassifier(), total_notext)
# train_notext(svm.LinearSVC(), total_notext)
# train_text(ensemble.RandomForestClassifier(), total_text)
# #train_notext(MultinomialNB(), total_notext)

# Cross Validation
#cross_val_train(svm.LinearSVC(), total_notext, 5, metrics.classification_report)

#PCA
#train_text_pca(ensemble.RandomForestClassifier(), total_text, 100)
#train_text_pca(svm.LinearSVC(), total_text, 100)

#LSA
#train_text_lsa(ensemble.RandomForestClassifier(), total_text, 100)

#Grid Search
#parameters =  dict(features__text__reduce_dim__n_components=[100, 200])
#estimators = [('TruncatedSVD', TruncatedSVD()), ('main', total_notext)]
#parameters = dict(reduce_dim__n_components=[10, 20, 50, 100 ,200, 500])
#grid_search_lsa(svm.LinearSVC(), feature_1, parameters)
#grid_search_lsa(ensemble.RandomForestClassifier(), feature_1, parameters)
#train_fixed_param(ensemble.RandomForestClassifier(), feature_1)
#train_fixed_param(svm.SVC(kernel='linear'), feature_1, "SVM Linear")
#train_fixed_param(svm.SVC(kernel='rbf'), feature_1, "SVM RBF")
#train_fixed_param(svm.SVC(kernel='poly'), feature_1, "SVM Poly")
#train_fixed_param(svm.SVC(kernel='sigmoid'), feature_1, "SVM Sigmoid")

#Reg Log
#train_fixed_param(linear_model.LogisticRegression(), feature_1, "Logistic Regression")

# ## Reunion 16/10/14
# model = linear_model.LogisticRegression()
# train_fixed_param(model, feature_1, "Logistic Regression")
# # cross_val_train(model, feature_1, 10, metrics.classification_report)

# model = ensemble.RandomForestClassifier()
# train_fixed_param(model, feature_1, "Random Forest")
# # cross_val_train(model, feature_1, 10, metrics.classification_report)

# model = svm.SVC(kernel='linear')
# train_fixed_param(model, feature_1, "SVM Linear")
# # cross_val_train(model, feature_1, 10, metrics.classification_report)

# model = svm.SVC(kernel='poly')
# train_fixed_param(model, feature_1, "SVM Poly")
# # cross_val_train(model, feature_1, 10, metrics.classification_report)

# model = svm.SVC(kernel='sigmoid')
# train_fixed_param(model, feature_1, "SVM Sigmoid")
# # cross_val_train(model, feature_1, 10, metrics.classification_report)

# model = svm.SVC(kernel='rbf')
# train_fixed_param(model, feature_1, "SVM RBF")
# # cross_val_train(model, feature_1, 10, metrics.classification_report())

#Fin de semana 22/11/14
# save = True
# #n_components = [10,50,100,500,1000,2000]
# n_components = [10]
# folds = 10
# models = {
#           'Logistic Regression': linear_model.LogisticRegression(),
#           #'Random Forest': ensemble.RandomForestClassifier(),
#           #'SVM Sigmoid': svm.SVC(kernel='sigmoid', probability=True),
#           #'SVM RBF': svm.SVC(kernel='rbf', probability=True),
#           #'SVM Poly': svm.SVC(kernel='poly', verbose=False,degree=2, cache_size=2000),
#           #'SVM Linear': svm.SVC(kernel='linear', verbose=False, cache_size=2000, probability=True),
#           #'SVM Linear': svm.LinearSVC(dual=False),
#           #'MultinomialNB': naive_bayes.MultinomialNB(),
#           #'GaussianNB': naive_bayes.GaussianNB(),
#           #'BernoulliNB': naive_bayes.BernoulliNB(),
#           #'SGDClassifier': linear_model.SGDClassifier()
# }
# for n_component in n_components:
#   for title, model in models.items():
#     print("Working on "+title)
#     train_fixed_param(model, consolidated_feature_2_0_8, title+", Noise Proportion 0.8 "+str(n_component)+" dim", n_component, save)
#     train_fixed_param(model, consolidated_feature_2_0_6, title+", Noise Proportion 0.6 "+str(n_component)+" dim", n_component, save)
#     train_fixed_param(model, consolidated_feature_2_0_4, title+", Noise Proportion 0.4 "+str(n_component)+" dim", n_component, save)
#     train_fixed_param(model, consolidated_feature_2_0_2, title+", Noise Proportion 0.2 "+str(n_component)+" dim", n_component, save)
#     train_fixed_param(model, feature_2, title+" "+str(n_component)+" dim",n_component, save)
#     #cross_val_train(model, consolidated_feature_2_0_8, folds, metrics.classification_report, "Cross Validation "+title+", NP 0.8, "+str(folds)+" folds "+str(n_component)+" dim", n_component, save)
#     #cross_val_train(model, consolidated_feature_2_0_6, folds, metrics.classification_report, "Cross Validation "+title+", NP 0.6, "+str(folds)+" folds "+str(n_component)+" dim", n_component, save)
#     #cross_val_train(model, consolidated_feature_2_0_4, folds, metrics.classification_report, "Cross Validation "+title+", NP 0.4, "+str(folds)+" folds "+str(n_component)+" dim", n_component, save)
#     #cross_val_train(model, consolidated_feature_2_0_2, folds, metrics.classification_report, "Cross Validation "+title+", NP 0.2, "+str(folds)+" folds "+str(n_component)+" dim", n_component, save)
#     #cross_val_train(model, feature_2, folds, metrics.classification_report, "Cross Validation "+title+", "+str(folds)+" folds "+str(n_component)+" dim", n_component, save)

# Search

#n_components = [10,50,100,500,1000, 2000]

models = {
          # 'Logistic Regression': linear_model.LogisticRegression(),
          'Random Forest': ensemble.RandomForestClassifier(),
          #'SVM Sigmoid': svm.SVC(kernel='sigmoid', probability=True),
          #'SVM RBF': svm.SVC(kernel='rbf', probability=True),
          #'SVM Poly': svm.SVC(kernel='poly', verbose=False,degree=2, cache_size=2000),
          #'SVM Linear': svm.SVC(kernel='linear', verbose=False, cache_size=2000, probability=True),
          #'SVM Linear': svm.LinearSVC(dual=False),
          #'MultinomialNB': naive_bayes.MultinomialNB(),
          #'GaussianNB': naive_bayes.GaussianNB(),
          #'BernoulliNB': naive_bayes.BernoulliNB(),
          #'SGDClassifier': linear_model.SGDClassifier()
}
parameters =  {
        'Logistic Regression':[
            dict(
                classifier__C=[2**x for x in range(-5,15)],
                classifier__penalty=['l1', 'l2'],
                classifier__tol=[1e-02,1e-03,1e-04],
                classifier__fit_intercept=[True, False],
                classifier__intercept_scaling=[1,10,100,1000]
            )
        ],
        'Random Forest':[
            dict(
                classifier__criterion=["gini", "entropy"],
                classifier__max_features=['auto', 'sqrt', 'log2', None],
                classifier__max_depth=[5,6,7,8,9,10,None],
                classifier__min_samples_split=[1,2,3,4,5,6,7],
                classifier__min_samples_leaf=[1,2,3,4],
                #classifier__bootstrap=[True, False],
                classifier__oob_score=[True, False]
            )
        ],
        'SVM Sigmoid': [
            dict(
                classifier__C=[2**x for x in range(-5,15)],
                classifier__gamma=[2**x for x in range(-15,5)],
                #classifier__coef0=[min(1-min , 0),max(<x,y>)], not correct for tuning
                classifier__probability=[True],
                classifier__shrinking=[True, False],
                #classifier__dual=[False],
                classifier__tol=[1e-02,1e-03,1e-04],
            )
        ],
        'SVM RBF': [
            dict(
                classifier__C=[2**x for x in range(-5,15)],
                classifier__gamma=[2**x for x in range(-15,5)],
                #classifier__coef0=[min(1-min , 0),max(<x,y>)], not correct for tuning
                classifier__probability=[True],
                classifier__shrinking=[True, False],
                #classifier__dual=[False],
                classifier__tol=[1e-02,1e-03,1e-04],
            )
        ],
        'SVM Linear': [
            dict(
                classifier__C=[2**x for x in range(-5,15)],
                #classifier__gamma=[2**x for x in range(-15,5)],
                classifier__loss=['l1', 'l2'],
                classifier__penalty=['l2'],
                #Just L2, because l1 and l1 is ot permitted
                classifier__dual=[True],
                #classifier__dual=[False],
                classifier__tol=[1e-02,1e-03,1e-04],
                classifier__fit_intercept=[True, False],
                classifier__intercept_scaling=[1,5,10,50,100,500,1000]
            )
        ],
        'MultinomialNB': [
            dict(
                classifier__alpha=[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                #classifier__alpha=[0.01, 0.05],
                classifier__fit_prior =[True, False]
            )
        ],
        #'GaussianNB': [],
        'BernoulliNB': [
            dict(
                classifier__alpha=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                classifier__binarize=[True, False],
                classifier__fit_prior =[True, False]
            )
        ],
        'SGDClassifier': [
            dict(
                 classifier__loss=['hinge', 'log', 'modified_huber', 'huber', 'epsilon_insensitive'],
                 classifier__penalty=['l1', 'l2', 'elasticnet'],
                 classifier__alpha=[0.01, 0.1, 0.5, 1],
                 classifier__l1_ratio=[0.01, 0.1, 0.5, 1],
                 classifier__fit_intercept=[True, False],
                 classifier__n_iter=[4,5,6],
                 classifier__shuffle=[True, False],
                 classifier__epsilon=[1e-01, 1e-02,1e-03,1e-04],
                 #classifier__learning_rate default is optimal
                 #classifier__eta0
                 classifier__power_t=[0.01, 0.1, 0.5, 1]
            )
        ]
}

base_params = {
    'save': True,
    'folds': 5,
    'n_jobs': -1,
    'dataset': feature_2,
    'noiseset': noise_feature_2,
    'noise_train': True,
    'noise_test': False,
    'verbose': 2,
    'parameters': parameters,
    
}
noise_proportions=[#0.0,
                   #0.2, 
                   #0.4,
                   #0.6, 
                   0.8
                   ]
n_components= [#10,
               #50,
               #100,
               #500,
               #1000,
               2000
               ]
base_params = dotdict(base_params)

for title, model in models.items():
    base_params['model']=model
    for noise_proportion in noise_proportions:
        for n_component in n_components:
            base_params['noise_proportion']=noise_proportion
            base_params['text']= title+" Grid Search Noise " + str(noise_proportion) + " LDA "+str(n_component)
            base_params['parameters']=parameters[title]
            base_params['title']=title
            base_params['noise']=noise_proportion
            base_params['n_component']=n_component
            grid_search_with_param(base_params)
