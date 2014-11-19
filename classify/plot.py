#Plots
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn import metrics, cross_validation

def print_report(X_test, y_test, resulting_model, prediction, text):
    print(text)
    print(metrics.classification_report(y_test, prediction))
    plot_roc(y_test, predict_scores(resulting_model, X_test), text)

    #if hasattr(resulting_model.named_steps['classifier'], 'coef_'):
    #    print(resulting_model.named_steps['classifier'].coef_)

def print_cross_val_report(result, roc):
    print(tabulate(result, headers = result.dtype.names))
    print("roc %0.6f" % roc)

def plot_roc(test, score, title=""):
    #ROC curve and ROC area
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