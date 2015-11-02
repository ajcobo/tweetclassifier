#Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from sklearn import metrics, cross_validation
from sklearn.externals import joblib
import csv

def print_report(X_test, y_test, resulting_model, prediction, params):
    print_results(X_test, y_test, resulting_model, prediction, params)
    if params.save:
        save_results(resulting_model, X_test, y_test, prediction, params)


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

def plot_roc(test, score, params):
    # Just lazyness in not refactoring everything for the text_file
    fpr, tpr, _ = metrics.roc_curve(test, score)
    roc_auc = metrics.auc(fpr, tpr)
    print("roc %0.6f" % roc_auc)
    title = 'Receiver operating characteristic ' + params.text
    # Plot of a ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if params.save:
        plt.savefig(title+'.png')
        np.savetxt(title+'.csv',(fpr, tpr), delimiter=',')
    else:
        plt.show()

def plot_precision_recall(test, score, params):
    # Just lazyness in not refactoring everything for the text_file
    precision, recall, _ = metrics.precision_recall_curve(test, score)
    title = 'Precision-Recall ' + params.text
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    if params.save:
        plt.savefig(title+'.png')
        np.savetxt(title+'.csv',(precision, recall), delimiter=',')
    else:
        plt.show()

def predict_scores(model, X_test):
    mlist=["RandomForestClassifier", "BernoulliNB", "MultinomialNB", "GaussianNB"]
    name = model.named_steps['classifier'].__class__.__name__
    subrequired= any([x for x in mlist if x == name])
    if hasattr(model.named_steps['classifier'], "decision_function"):
        if subrequired:
            return model.named_steps['classifier'].decision_function(X_test)[:,1]
        else:
            return model.named_steps['classifier'].decision_function(X_test).astype(np.float64)
    else:
        if subrequired:
            return model.named_steps['classifier'].predict_proba(X_test)[:,1]
        else:
            return model.named_steps['classifier'].predict_proba(X_test).astype(np.float64)

def predict_single_scores(model, X_test):
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        return model.predict_proba(X_test)[:,1]
    else:
        if (model.__class__.__name__ == 'RandomForestClassifier'):
            return model.decision_function(X_test)[:,1]
        else:
            return model.decision_function(X_test).astype(np.float64)


def grid_search_predict_scores(model, X_test):
    if hasattr(model.best_estimator_.named_steps['classifier'], "decision_function"):
        if (model.best_estimator_.named_steps['classifier'].__class__.__name__== 'RandomForestClassifier'):
            return model.best_estimator_.named_steps['classifier'].decision_function(X_test)[:,1]
        else:
            return model.best_estimator_.named_steps['classifier'].decision_function(X_test).astype(np.float64)
    else:
        if (model.best_estimator_.__class__.__name__ == 'RandomForestClassifier'):
            return model.best_estimator_.named_steps['classifier'].predict_proba(X_test)[:,1]
        else:
            return model.best_estimator_.named_steps['classifier'].predict_proba(X_test)

def save_model(model, params):
    joblib.dump(model, params.text+'.pkl', compress=3)

def save_results(resulting_model, X_test, y_test, prediction, params):
    save_model(resulting_model.best_estimator_, params)
    predicted_scores = predict_scores(resulting_model.best_estimator_, X_test)
    roc_auc = get_roc_auc(y_test, predicted_scores)
    prf = metrics.precision_recall_fscore_support(y_test, prediction)
    acc = metrics.accuracy_score(y_test,prediction)

    with open(params.text+'.csv', 'w', newline='') as fp:
        writer = csv.writer(fp, delimiter=',')
        best_params = resulting_model.best_params_
        data = {'precision': prf[0][1],
                'recall': prf[1][1],
                'fscore': prf[2][1],
                'accuracy': acc,
                'roc': roc_auc}
        
        #remove what we dont want
        temp_params = dict(params)
        for key in ('dataset', 'noiseset', 'save'):
            if key in temp_params:
                del temp_params[key]
        for key, value in temp_params.items():
            writer.writerow([key, value])
        for key, value in best_params.items():
            writer.writerow([key, value])
        for key, value in data.items():
            writer.writerow([key, value])

        if (resulting_model.best_estimator_.__class__.__name__ == 'LogisticRegression'):
            writer.writerow(['Interpreted Weights', ''])
            np.set_printoptions(formatter= {'float_kind':'{:14f}'.format})
            writer.writerows(np.exp( resulting_model.best_estimator_.coef_))

def print_results(X_test, y_test, resulting_model, prediction, params):
    print(resulting_model.best_params_)
    print(params.text)
    print(metrics.classification_report(y_test, prediction))
    print('accuracy '+str(metrics.accuracy_score(y_test,prediction)))
    predicted_scores = predict_scores(resulting_model.best_estimator_, X_test)
    plot_roc(y_test, predicted_scores, params)
    plot_precision_recall(y_test, predicted_scores, params)
