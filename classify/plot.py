#Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from sklearn import metrics, cross_validation
from sklearn.externals import joblib
import csv

def print_report(X_test, y_test, resulting_model, prediction, text, save = False):
    print_results(X_test, y_test, resulting_model, prediction, text, save)
    if save:
        save_results(resulting_model,y_test,prediction, text)


def print_cross_val_report(result, roc, title = "", save = False):
    tabulated_result = tabulate(result, headers = result.dtype.names)
    roc_summary = "roc %0.6f" % roc
    if save:
        text_file = open(title+".txt", "w")
        text_file.write(tabulated_result+'\n')
        text_file.write("roc %0.6f" % roc)
        text_file.close()
    else:
        print(tabulated_result)
        print(roc_summary)

def get_roc_auc(test, score):
    #ROC curve and ROC area
    fpr, tpr, _ = metrics.roc_curve(test, score)
    roc_auc = metrics.auc(fpr, tpr)
    return(roc_auc)

def plot_roc(test, score, title="", save = False):
    # Just lazyness in not refactoring everything for the text_file
    fpr, tpr, _ = metrics.roc_curve(test, score)
    roc_auc = metrics.auc(fpr, tpr)
    print("roc %0.6f" % roc_auc)
    # Plot of a ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic ' + title)
    plt.legend(loc="lower right")
    if save:
        plt.savefig(title+'.png')
    else:
        plt.show()

def predict_scores(model, X_test):
    mlist=["RandomForestClassifier", "BernoulliNB", "MultinomialNB", "GaussianNB"]
    name = model.named_steps['classifier'].__class__.__name__
    subrequired= any([x for x in mlist if x == name])
    if hasattr(model.named_steps['classifier'], "decision_function"):
        if subrequired:
            return model.decision_function(X_test)[:,1]
        else:
            return model.decision_function(X_test).astype(np.float64)
    else:
        if subrequired:
            return model.predict_proba(X_test)[:,1]
        else:
            return model.predict_proba(X_test).astype(np.float64)

def predict_single_scores(model, X_test):
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        return model.predict_proba(X_test)[:,1]
    else:
        if (model.named_steps['classifier'].__class__.__name__ == 'RandomForestClassifier'):
            return model.decision_function(X_test)[:,1]
        else:
            return model.decision_function(X_test).astype(np.float64)


def grid_search_predict_scores(model, X_test):
    if hasattr(model.best_estimator_.named_steps, "decision_function"):
        if (model.best_estimator_.named_steps['classifier'].__class__.__name__== 'RandomForestClassifier'):
            return model.best_estimator_.decision_function(X_test)[:,1]
        else:
            return model.best_estimator_.decision_function(X_test).astype(np.float64)
    else:
        if (model.best_estimator_.named_steps['classifier'].__class__.__name__ == 'RandomForestClassifier'):
            return model.best_estimator_.predict_proba(X_test)[:,1]
        else:
            return model.best_estimator_.predict_proba(X_test)

def save_model(model, title):
    joblib.dump(model, title+'.pkl', compress=9)

def save_results(resulting_model, X_test, y_test, prediction, text):
    save_model(resulting_model.best_estimator_, text)
    predicted_scores = predict_scores(resulting_model.best_estimator_, X_test)
    roc_auc = get_roc_auc(y_test, predicted_scores)
    prf = metrics.precision_recall_fscore_support(y_test, prediction)
    acc = metrics.accuracy_score(y_test,prediction)

    with open('test.csv', 'w', newline='') as fp:
        a = csv.writer(fp, delimiter=',')
        params = []
        data = [['precision', prf[0][1]],
                ['recall', prf[1][1]],
                ['fscore', prf[2][1]],
                ['accuracy', acc],
                ['roc', roc_auc]]
        a.writerows(params)
        a.writerows(data)

        if (resulting_model.best_estimator_.named_steps['classifier'].__class__.__name__ == 'LogisticRegression'):
            a.writerow('Interpreted Weights')
            np.set_printoptions(formatter= {'float_kind':'{:14f}'.format})
            a.writerows(np.exp( resulting_model.best_estimator_.named_steps['classifier'].coef_))

def print_results(X_test, y_test, resulting_model, prediction, text, save):
    print(resulting_model.best_params_)
    print(text)
    print(metrics.classification_report(y_test, prediction))
    print('accuracy '+str(metrics.accuracy_score(y_test,prediction)))
    predicted_scores = predict_scores(resulting_model, X_test)
    plot_roc(y_test, predicted_scores, text, save)