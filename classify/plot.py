#Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from sklearn import metrics, cross_validation
from sklearn.externals import joblib

def print_single_report(X_test, y_test, resulting_model, prediction, text, save = False):
    print(text)
    print(metrics.classification_report(y_test, prediction))
    print('accuracy '+str(metrics.accuracy_score(y_test,prediction)))
    predicted_scores = predict_single_scores(resulting_model, X_test)
    plot_roc(y_test, predicted_scores, text, save)
    if save:
        save_model(resulting_model, text)
        text_file = open(text+".txt", "w")
        text_file.write(text+'\n')
        text_file.write(metrics.classification_report(y_test, prediction))
        accuracy = metrics.accuracy_score(y_test,prediction)
        text_file.write("accuracy %0.6f" % accuracy+'\n')
        roc_auc = get_roc_auc(y_test, predicted_scores)
        text_file.write("roc %0.6f" % roc_auc+'\n')
        if (resulting_model.__class__.__name__ == 'LogisticRegression'):
            text_file.write('Interpreted Weights\n')
            np.set_printoptions(formatter= {'float_kind':'{:14f}'.format})
            text_file.write(str(np.exp( resulting_model.coef_)))
        text_file.close()

def print_report(X_test, y_test, resulting_model, prediction, text, save = False):
    print(text)
    print(metrics.classification_report(y_test, prediction))
    print('accuracy '+str(metrics.accuracy_score(y_test,prediction)))
    predicted_scores = predict_scores(resulting_model, X_test)
    plot_roc(y_test, predicted_scores, text, save)
    if save:
        save_model(resulting_model, text)
        text_file = open(text+".txt", "w")
        text_file.write(text+'\n')
        text_file.write(metrics.classification_report(y_test, prediction))
        accuracy = metrics.accuracy_score(y_test,prediction)
        text_file.write("accuracy %0.6f" % accuracy+'\n')
        roc_auc = get_roc_auc(y_test, predicted_scores)
        text_file.write("roc %0.6f" % roc_auc+'\n')
        if (resulting_model.named_steps['classifier'].__class__.__name__ == 'LogisticRegression'):
            text_file.write('Interpreted Weights\n')
            np.set_printoptions(formatter= {'float_kind':'{:14f}'.format})
            text_file.write(str(np.exp( resulting_model.named_steps['classifier'].coef_)))
        text_file.close()

    #if hasattr(resulting_model.named_steps['classifier'], 'coef_'):
    #    print(resulting_model.named_steps['classifier'].coef_)

def print_grid_search_report(X_test, y_test, resulting_model, prediction, text, save = False):
    print(text)
    print(resulting_model.best_params_)
    print(metrics.classification_report(y_test, prediction))
    print(metrics.accuracy_score(y_test,prediction))
    predicted_scores = grid_search_predict_scores(resulting_model, X_test)
    plot_roc(y_test, predicted_scores, text, save)
    if save:
        save_model(resulting_model.best_estimator_, text)
        text_file = open(text+".txt", "w")
        text_file.write(text+'\n')
        text_file.write(str(resulting_model.best_params_)+'\n')
        text_file.write(metrics.classification_report(y_test, prediction))
        text_file.write(metrics.accuracy_score(y_test,prediction)+'\n')
        roc_auc = get_roc_auc(y_test, predicted_scores)
        text_file.write("roc %0.6f" % roc_auc+'\n')
        if (resulting_model.best_estimator_.__class__.__name__ == 'LogisticRegression'):
            text_file.write('Interpreted Weights\n')
            np.set_printoptions(formatter= {'float_kind':'{:14f}'.format})
            text_file.write(str(np.exp( resulting_model.best_estimator_.named_steps['classifier'].coef_)))
        text_file.close()

    #if hasattr(resulting_model.named_steps['classifier'], 'coef_'):
    #    print(resulting_model.named_steps['classifier'].coef_)

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
    if hasattr(model.named_steps['classifier'], "decision_function"):
        if (model.named_steps['classifier'].__class__.__name__ == 'RandomForestClassifier'):
            return model.decision_function(X_test)[:,1]
        else:
            return model.decision_function(X_test).astype('float')
    else:
        if (model.named_steps['classifier'].__class__.__name__ == 'RandomForestClassifier'):
            return model.predict_proba(X_test)[:,1]
        else:
            return model.predict_proba(X_test)

def predict_single_scores(model, X_test):
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        return model.predict_proba(X_test)[:,1]
    else:
        if (model.named_steps['classifier'].__class__.__name__ == 'RandomForestClassifier'):
            return model.decision_function(X_test)[:,1]
        else:
            return model.decision_function(X_test).astype('float')


def grid_search_predict_scores(model, X_test):
    if hasattr(model.best_estimator_.named_steps, "decision_function"):
        if (model.best_estimator_.named_steps['classifier'].__class__.__name__== 'RandomForestClassifier'):
            return model.best_estimator_.decision_function(X_test)[:,1]
        else:
            return model.best_estimator_.decision_function(X_test).astype('float')
    else:
        if (model.best_estimator_.named_steps['classifier'].__class__.__name__ == 'RandomForestClassifier'):
            return model.best_estimator_.predict_proba(X_test)[:,1]
        else:
            return model.best_estimator_.predict_proba(X_test)

def save_model(model, title):
    joblib.dump(model, title+'.pkl', compress=9)