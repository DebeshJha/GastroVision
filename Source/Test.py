from collections import defaultdict
import numpy as np
import torch
from misc import plot_confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as mtc
from sklearn.metrics import confusion_matrix

def test_net(model,test_generator,device,criterion):
        num_steps=0
        test_loss=0
        correct=0
        test_metrics=defaultdict(float)
        all_labels_d = torch.tensor([], dtype=torch.long).to(device)
        all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
        all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)
        for image, labels in test_generator:
                #Transfer to GPU:
               
                image, labels = image.to(device,  dtype=torch.float32), labels.to(device)
                outputs = model(image)
               
                loss = criterion(outputs, labels)
                
                num_steps+=image.size(0)
                
                test_loss+=loss.item()*image.size(0)
             
                predicted_probability, predicted = torch.max(outputs, dim=1)                
        
                correct += (predicted == labels).sum()
                all_labels_d = torch.cat((all_labels_d, labels), 0)
                all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
                all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)
                
        y_true = all_labels_d.cpu()
        y_predicted = all_predictions_d.cpu()  # to('cpu')
        valset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')
        
        micro_precision=mtc.precision_score(y_true, y_predicted, average="micro")     
        micro_recall=mtc.recall_score(y_true, y_predicted, average="micro")
        micro_f1=mtc.f1_score(y_true, y_predicted, average="micro")  
        
        macro_precision=mtc.precision_score(y_true, y_predicted, average="macro")     
        macro_recall=mtc.recall_score(y_true, y_predicted, average="macro")
        macro_f1=mtc.f1_score(y_true, y_predicted, average="macro")  
        
        mcc=mtc.matthews_corrcoef(y_true, y_predicted)
        
        test_metrics['micro_precision']=micro_precision
        test_metrics['micro_recall']=micro_recall
        test_metrics['micro_f1']=micro_f1
        test_metrics['macro_precision']=macro_precision
        test_metrics['macro_recall']=macro_recall
        test_metrics['macro_f1']=macro_f1
        test_metrics['mcc']=mcc
        
        cm = confusion_matrix(y_true, y_predicted)  # confusion matrix



        print('Accuracy of the network on the %d test images: %f %%' % (num_steps, (100.0 * correct / num_steps)))

        print(cm)

        print("taking class names to plot CM")

        class_names = test_generator.dataset.classes #test_datasets.classes  # taking class names for plotting confusion matrix

        print("Generating confution matrix")

        plot_confusion_matrix(cm, classes=class_names, title='my confusion matrix')

    

        ##################################################################
        # classification report
        #################################################################
        print(classification_report(y_true, y_predicted, target_names=class_names))


        
                         
       
        return (test_loss/num_steps), test_metrics, num_steps
        
