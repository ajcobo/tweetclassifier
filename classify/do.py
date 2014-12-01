from load import *
from clean import *
from func import *

from sklearn import svm, ensemble, linear_model

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

consolidated_feature_1_0_8 = join_datasets_by_proportion(dataset = feature_1, noiseset = noise_feature_1, noise_proportion = 0.8, train_proportion = 0.8)
consolidated_feature_2_0_8 = join_datasets_by_proportion(dataset = feature_2, noiseset = noise_feature_2, noise_proportion = 0.8, train_proportion = 0.8)
consolidated_feature_3_0_8 = join_datasets_by_proportion(dataset = feature_3, noiseset = noise_feature_3, noise_proportion = 0.8, train_proportion = 0.8)
consolidated_feature_4_0_8 = join_datasets_by_proportion(dataset = feature_4, noiseset = noise_feature_4, noise_proportion = 0.8, train_proportion = 0.8)


consolidated_feature_1_0_6 = join_datasets_by_proportion(dataset = feature_1, noiseset = noise_feature_1, noise_proportion = 0.6, train_proportion = 0.8)
consolidated_feature_2_0_6 = join_datasets_by_proportion(dataset = feature_2, noiseset = noise_feature_2, noise_proportion = 0.6, train_proportion = 0.8)
consolidated_feature_3_0_6 = join_datasets_by_proportion(dataset = feature_3, noiseset = noise_feature_3, noise_proportion = 0.6, train_proportion = 0.8)
consolidated_feature_4_0_6 = join_datasets_by_proportion(dataset = feature_4, noiseset = noise_feature_4, noise_proportion = 0.6, train_proportion = 0.8)

consolidated_feature_1_0_4 = join_datasets_by_proportion(dataset = feature_1, noiseset = noise_feature_1, noise_proportion = 0.4, train_proportion = 0.8)
consolidated_feature_2_0_4 = join_datasets_by_proportion(dataset = feature_2, noiseset = noise_feature_2, noise_proportion = 0.4, train_proportion = 0.8)
consolidated_feature_3_0_4 = join_datasets_by_proportion(dataset = feature_3, noiseset = noise_feature_3, noise_proportion = 0.4, train_proportion = 0.8)
consolidated_feature_4_0_4 = join_datasets_by_proportion(dataset = feature_4, noiseset = noise_feature_4, noise_proportion = 0.4, train_proportion = 0.8)

consolidated_feature_1_0_2 = join_datasets_by_proportion(dataset = feature_1, noiseset = noise_feature_1, noise_proportion = 0.2, train_proportion = 0.8)
consolidated_feature_2_0_2 = join_datasets_by_proportion(dataset = feature_2, noiseset = noise_feature_2, noise_proportion = 0.2, train_proportion = 0.8)
consolidated_feature_3_0_2 = join_datasets_by_proportion(dataset = feature_3, noiseset = noise_feature_3, noise_proportion = 0.2, train_proportion = 0.8)
consolidated_feature_4_0_2 = join_datasets_by_proportion(dataset = feature_4, noiseset = noise_feature_4, noise_proportion = 0.2, train_proportion = 0.8)
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
save = True
n_components = 100
folds = 10
models = {
          #'Logistic Regression': linear_model.LogisticRegression(),
          #'Random Forest': ensemble.RandomForestClassifier(),
          #'SVM Sigmoid': svm.SVR(kernel='sigmoid'),
          #'SVM RBF': svm.SVC(kernel='rbf'),
          # 'SVM Poly': svm.SVC(kernel='poly', verbose=True, cache_size=2000),
          # 'SVM Linear': svm.SVC(kernel='linear', verbose=True, cache_size=2000)

}
for title, model in models.items():
  print("Working on "+title)
  train_fixed_param(model, consolidated_feature_1_0_8, title+", Noise Proportion 0.8 "+str(n_components)+" dim", n_components, save)
  train_fixed_param(model, consolidated_feature_1_0_6, title+", Noise Proportion 0.6 "+str(n_components)+" dim", n_components, save)
  train_fixed_param(model, consolidated_feature_1_0_4, title+", Noise Proportion 0.4 "+str(n_components)+" dim", n_components, save)
  train_fixed_param(model, consolidated_feature_1_0_2, title+", Noise Proportion 0.2 "+str(n_components)+" dim", n_components, save)
  train_fixed_param(model, feature_1, title+" "+str(n_components)+" dim", n_components, save)
  cross_val_train(model, consolidated_feature_1_0_8, folds, metrics.classification_report, "Cross Validation "+title+", NP 0.8, "+str(folds)+" folds "+str(n_components)+" dim", n_components, save)
  cross_val_train(model, consolidated_feature_1_0_6, folds, metrics.classification_report, "Cross Validation "+title+", NP 0.6, "+str(folds)+" folds "+str(n_components)+" dim", n_components, save)
  cross_val_train(model, consolidated_feature_1_0_4, folds, metrics.classification_report, "Cross Validation "+title+", NP 0.4, "+str(folds)+" folds "+str(n_components)+" dim", n_components, save)
  cross_val_train(model, consolidated_feature_1_0_2, folds, metrics.classification_report, "Cross Validation "+title+", NP 0.2, "+str(folds)+" folds "+str(n_components)+" dim", n_components, save)
  cross_val_train(model, feature_1, folds, metrics.classification_report, "Cross Validation "+title+", "+str(folds)+" folds", n_components, save)

# #Naive Bayes Gaussian Multinomial Bernoulli
# model = GaussianNB()
# model = MultinomialNB
# model = BernoulliNB
# #SGD: Stochastic Gradient Descent
# model = SGDClassifier()
# #Grid Search best algorithms