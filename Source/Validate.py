from collections import defaultdict
import numpy as np
import torch
import csv
import sklearn.metrics as mtc
from sklearn.metrics import classification_report

def validate_net(model,validation_generator,device,criterion):
        num_steps=0
        val_loss=0
        correct=0
        val_metrics=defaultdict(float)
        all_labels_d = torch.tensor([], dtype=torch.long).to(device)
        all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
        all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)
        for image, labels in validation_generator:
                #Transfer to GPU:
               
                image, labels = image.to(device, dtype=torch.float32), labels.to(device)
                outputs = model(image)
                
                loss = criterion(outputs, labels)
                
                num_steps+=image.size(0)
                
                val_loss+=loss.item()*image.size(0)
        
                predicted_probability, predicted = torch.max(outputs, dim=1)                            
                
                correct += (predicted == labels).sum()
                all_labels_d = torch.cat((all_labels_d, labels), 0)
                all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
                all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)
                
        y_true = all_labels_d.cpu()
        y_predicted = all_predictions_d.cpu()  # to('cpu')
        valset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')
        
        
        #############################
        # Standard metrics 
        #############################
        
        micro_precision=mtc.precision_score(y_true, y_predicted, average="micro")     
        micro_recall=mtc.recall_score(y_true, y_predicted, average="micro")
        micro_f1=mtc.f1_score(y_true, y_predicted, average="micro")  
        
        macro_precision=mtc.precision_score(y_true, y_predicted, average="macro")     
        macro_recall=mtc.recall_score(y_true, y_predicted, average="macro")
        macro_f1=mtc.f1_score(y_true, y_predicted, average="macro")  
        
        mcc=mtc.matthews_corrcoef(y_true, y_predicted)
        
        
        
        val_metrics['micro_precision']=micro_precision
        val_metrics['micro_recall']=micro_recall
        val_metrics['micro_f1']=micro_f1
        val_metrics['macro_precision']=macro_precision
        val_metrics['macro_recall']=macro_recall
        val_metrics['macro_f1']=macro_f1
        val_metrics['mcc']=mcc
        
                         
       
        return (val_loss/num_steps), val_metrics, num_steps
        
                

