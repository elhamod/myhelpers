import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report
import os    

def generate_classification_report(lbllist, predlist, numberOfSpecies, experimentName):
    classificationReport = classification_report(lbllist.cpu().numpy(), predlist.cpu().numpy(), labels = range(numberOfSpecies), digits=2)
    classification_report_file = open(os.path.join(experimentName,"classification_report.txt"),"w")
    classification_report_file.writelines(classificationReport) 
    classification_report_file.close() #to change file access modes 
    return classificationReport

def plot_confusion_matrix(cm,
                          target_names,
                          experimentName,
                          fileName,
                          printOutput=False,
                          cmap=None,
                          normalize=True,
):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(30, 30), dpi=300)
    plt.title("Confusion Matrix")

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.imshow(cm, cmap=cmap) # , interpolation='nearest'
    plt.colorbar()


    # Removing text for now. TODO: add a flag for this.
    # thresh = np.nanmax(cm) / 1.5 if normalize else np.nanmax(cm) / 2
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     if normalize:
    #         plt.text(j, i, "{:0.2f}".format(cm[i, j]),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")
    #     else:
    #         plt.text(j, i, "{:,}".format(cm[i, j]),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nAccuracy={:0.4f}'.format(accuracy))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(experimentName, fileName))
    if printOutput:
        plt.show()
    plt.close()
    return cm