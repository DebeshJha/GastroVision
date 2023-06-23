import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

###################################################
# Print metrics values 
###################################################

def print_metrics(metrics, num_steps):
    outputs=[]
    for k in metrics.keys():
        if k=='dice_coeff' or k=='dice' or k=='bce':
            outputs.append('{}:{:4f}'.format(k,metrics[k]/num_steps))
        else:
            outputs.append('{}:{:2f}'.format(k,metrics[k]))
    print('{}'.format(','.join(outputs)))
    
    
###################################################
# Plot training validation loss curve 
###################################################
    
def training_curve(epochs,lossesT,lossesV):
    plt.plot(epochs, lossesT,'c')
    plt.plot(epochs, lossesV,'m')
    plt.title("Training Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f'train_val_epoch_curve')
    plt.close()



############################################################
# Plot confusion matrix 
############################################################
def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues,
                            plt_size=[10,10]):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.rcParams['figure.figsize'] = plt_size
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    figure = plt.gcf()
    plt.savefig('conf.png')
    print("Finished confusion matrix drawing...")




