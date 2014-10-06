from load import *
from clean import *
from func import *

from sklearn import svm, ensemble

#Load Dataset
earthquake_time = datetime.datetime(2010, 2, 27, 3,34,8)
dataset = extract_features("/home/alf/Dropbox/Master/AC/Ground Truth/Consolidated/GroundTruthTotal.csv", earthquake_time)

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

# Perform analysis

# #train(svm.SVC(), test_feature)
# train_text(svm.LinearSVC(), total_text)
# #train_text(MultinomialNB(), total_text)
# train_notext(ensemble.RandomForestClassifier(), total_notext)
# train_notext(svm.LinearSVC(), total_notext)
# train_text(ensemble.RandomForestClassifier(), total_text)
# #train_notext(MultinomialNB(), total_notext)

# Cross Validation
#cross_val_train(svm.LinearSVC(), total_notext, 5, metrics.classification_report)

#PCA
train_text_pca(ensemble.RandomForestClassifier(), total_text, 50)

#LSA
train_text_pca(ensemble.RandomForestClassifier(), total_text, 100)