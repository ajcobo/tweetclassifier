from load import *
from clean import *
from func import *

from sklearn import svm, ensemble, linear_model

#Load Dataset
earthquake_time = datetime.datetime(2010, 2, 27, 3,34,8)
dataset = extract_features("/home/alf/Dropbox/Master/AC/Ground Truth/Consolidated/GroundTruthTotal.csv", earthquake_time)

#Extract each feature set
columns_feature_1 = ['texto', 'time_gap', 'followers_count', 'friends_count']
columns_feature_2 = ['texto', 'followers_count', 'friends_count']
columns_feature_3 = ['texto', 'time_gap', 'friends_count']
columns_feature_4 = ['texto', 'time_gap', 'followers_count']
columns_feature_5 = ['time_gap', 'followers_count', 'friends_count']

#does it work?
dataset.features['time_gap'] = dataset.features['time_gap'].astype('timedelta64[h]')

feature_1 = [dataset.features[columns_feature_1].values, dataset.features['value'].values]
feature_2 = [dataset.features[columns_feature_2].values, dataset.features['value'].values]
feature_3 = [dataset.features[columns_feature_3].values, dataset.features['value'].values]
feature_4 = [dataset.features[columns_feature_4].values, dataset.features['value'].values]
total_text = [dataset.features['texto'].values, dataset.features['value'].values]
total_notext = [dataset.features[columns_feature_5].values, dataset.features['value'].values]

test_feature = [['esto es una prueba', 'una prueba dada por arreglo', 'me gusta este arreglo'], np.array([False, True, True], dtype=bool)]

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
train_fixed_param(svm.SVC(kernel='rbf'), feature_1, "SVM RBF")
#train_fixed_param(svm.SVC(kernel='poly'), feature_1)
#train_fixed_param(svm.SVC(kernel='sigmoid'), feature_1)

#Reg Log
train_fixed_param(linear_model.LogisticRegression(), feature_1, "Logistic Regression")

