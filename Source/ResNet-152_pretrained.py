import torch
import torchvision
import time
from torch import nn
from torch import optim
from torch.utils import data
from Validate import validate_net
from Test import test_net
from misc import print_metrics, training_curve 
from PIL import Image
import os
import re
import argparse
from collections import defaultdict
import numpy as np
import logging
import csv
from torchvision import transforms, datasets, models
import sklearn.metrics as mtc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


###########################
# Checking if GPU is used
###########################

use_cuda=torch.cuda.is_available()
device=torch.device("cuda:0" if use_cuda else "cpu")

########################################
# Setting basic parameters for the model
########################################

def get_args():
    parser=argparse.ArgumentParser(description='Train the model on images and target labels',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e','--epochs', metavar='E', type=int, default=4, nargs='?', help='Number of epochs', dest='max_epochs')
    parser.add_argument('-b','--batch-size', metavar='B', type=int, default=3, nargs='?', help='Batch size', dest='batch_size')
    parser.add_argument('-l','--learning-rate', metavar='LR', type=float, default=0.004, nargs='?', help='Learning rate', dest='lr')
    
    return parser.parse_args()               
         

args=get_args()
batch_size=args.batch_size
max_epochs=args.max_epochs
lr=args.lr

train_root_dir=f"./GastroVision/train"
val_root_dir=f"./GastroVision/val"
test_root_dir=f"./GastroVision/test"
model_path=r'./checkpoints/'  # set path to the folder that will store model's checkpoints

n_classes=22  # number of classes used for training

global val_f1_max


try:
   if not os.path.exists(os.path.dirname(model_path)):
       os.makedirs(os.path.dirname(model_path))
except OSError as err:
   print(err)

print("Directory '% s' created" % model_path)
filename='results_e'+str(max_epochs)+'_'+'b'+str(batch_size)+'_'+'lr'+str(lr)+'_'+'resnet152'   #filename used for saving epoch-wise training details and test results 

####################################
# Training
####################################

trans={
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # ImageNet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}	

        
class train:
    def __init__(self):

        #Generators
        training_dataset= datasets.ImageFolder(train_root_dir,transform=trans['train'])
        validation_dataset= datasets.ImageFolder(val_root_dir,transform=trans['valid'])
        test_dataset= datasets.ImageFolder(test_root_dir,transform=trans['test'])
        
        self.training_generator=data.DataLoader(training_dataset,batch_size,shuffle=True) # ** unpacks a dictionary into keyword arguments
        self.validation_generator=data.DataLoader(validation_dataset,batch_size)
        self.test_generator=data.DataLoader(test_dataset,batch_size)
       
        print('Number of Training set images:{}'.format(len(training_dataset)))
        print('Number of Validation set images:{}'.format(len(validation_dataset)))
        print('Number of Test set images:{}'.format(len(test_dataset)))
        
    def train_net(self):
        
        #Initialize model
        model = torchvision.models.resnet152(weights=False).to(device)   # make weights=True if you want to download pre-trained weights
        
        
        model.load_state_dict(torch.load('./resnet152-394f9c45.pth',map_location='cuda'))   # provide a .pth path for already downloaded weights otherwise comment this line out 	        
        
        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = False
        
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
                      nn.Linear(n_inputs, n_classes),                   
                      nn.LogSoftmax(dim=1))
        
       
        model.to(device)
        optimizer=optim.Adam(model.parameters(), lr, weight_decay=1e-4)
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=4,verbose=True)
        criterion = nn.NLLLoss()
        val_f1_max=0.0
        epochs=[]
        lossesT=[]
        lossesV=[]

        for epoch in range(max_epochs):
            print('Epoch {}/{}'.format(epoch+1,max_epochs))
            print('-'*10)
            
            since=time.time()
            train_metrics=defaultdict(float)
            total_loss=0
            running_corrects=0
            num_steps=0
            
            all_labels_d = torch.tensor([], dtype=torch.long).to(device)
            all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
            all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)
            
            model.train()
            
            #Training
            for image, labels in self.training_generator:
                #Transfer to GPU:
                
                image, labels = image.to(device, dtype=torch.float32), labels.to(device)
                outputs = model(image)
                predicted_probability, predicted  = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
              
                num_steps+=image.size(0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss+=loss.item()*image.size(0)
           
                running_corrects += torch.sum(predicted == labels.data)
                all_labels_d = torch.cat((all_labels_d, labels), 0)
                all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
                all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)
                
                
            y_true = all_labels_d.cpu()
            y_predicted = all_predictions_d.cpu()  # to('cpu')
            valset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')
            
            
            #############################
            # Standard metrics 
            #############################
        
            train_micro_precision=mtc.precision_score(y_true, y_predicted, average="micro")     
            train_micro_recall=mtc.recall_score(y_true, y_predicted, average="micro")
            train_micro_f1=mtc.f1_score(y_true, y_predicted, average="micro")  
        
            train_macro_precision=mtc.precision_score(y_true, y_predicted, average="macro")     
            train_macro_recall=mtc.recall_score(y_true, y_predicted, average="macro")
            train_macro_f1=mtc.f1_score(y_true, y_predicted, average="macro")  
        
            train_mcc=mtc.matthews_corrcoef(y_true, y_predicted)
             
            
            train_metrics['loss']=total_loss/num_steps
        
            train_metrics['micro_precision']=train_micro_precision
            train_metrics['micro_recall']=train_micro_recall
            train_metrics['micro_f1']=train_micro_f1
            train_metrics['macro_precision']=train_macro_precision
            train_metrics['macro_recall']=train_macro_recall
            train_metrics['macro_f1']=train_macro_f1
            train_metrics['mcc']=train_mcc
            
            print('Training...')
            print('Train_loss:{:.3f}'.format(total_loss/num_steps))
           
            
            print_metrics(train_metrics,num_steps)

            ############################
            # Validation
            ############################
            
            model.eval()
            with torch.no_grad():
                val_loss, val_metrics, val_num_steps=validate_net(model,self.validation_generator,device,criterion)
                
            scheduler.step(val_loss)
            epochs.append(epoch)
            lossesT.append(total_loss/num_steps)
            lossesV.append(val_loss)
            
            print('.'*5)
            print('Validating...')
            print('val_loss:{:.3f}'.format(val_loss))
        
            print_metrics(val_metrics,val_num_steps)


            ##################################################################
            # Writing epoch-wise training and validation results to a csv file 
            ##################################################################

            key_name=['Epoch','Train_loss','Train_micro_precision','Train_micro_recall','Train_micro_f1','Train_macro_precision','Train_macro_recall','Train_macro_f1','Train_mcc','Val_loss','Val_micro_precision','Val_micro_recall','Val_micro_f1','Val_macro_precision','Val_macro_recall','Val_macro_f1','Val_mcc']
            train_list=[]
            train_list.append(epoch)

            try:

                with open(filename+str('.csv'), 'a',newline="") as f:
                    wr = csv.writer(f,delimiter=",")
                    if epoch==0:
                        wr.writerow(key_name)

                    for k, vl in train_metrics.items():
                        train_list.append(vl)

                    train_list.append(val_loss)

                    for k, vl in val_metrics.items():
                        train_list.append(vl)
                    zip(train_list)
                    wr.writerow(train_list)


            except IOError:
                print("I/O Error")

            
            ##############################
            # Saving best model 
            ##############################
            
            if val_metrics['micro_f1']>=val_f1_max:
                print('val micro f1 increased ({:.6f}-->{:.6f}).Saving model'.format(val_f1_max,val_metrics['micro_f1']))
                
                torch.save({'epoch':epoch+1,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 
                            'loss':val_loss},model_path+f'/C_{epoch+1}_{batch_size}.pth')
                best_model_path=model_path+f'/C_{epoch+1}_{batch_size}.pth'
               
                val_f1_max=val_metrics['micro_f1']
                

            print('-'*10)
       
        time_elapsed=time.time()-since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        training_curve(epochs,lossesT,lossesV)
        epochs.clear()
        lossesT.clear()
        lossesV.clear()
        

        ############################
        #         Test
        ############################
        test_list=[]
        print('Best model path:{}'.format(best_model_path))
        best_model=torchvision.models.resnet152(weights=False).to(device)
        
        n_inputs = best_model.fc.in_features
        best_model.fc = nn.Sequential(
                      nn.Linear(n_inputs, n_classes),                  
                      nn.LogSoftmax(dim=1))

 
        checkpoint=torch.load(best_model_path,map_location=device)   # loading best model
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model.to(device)
        best_model.eval()
        with torch.no_grad():
       	       test_loss, test_metrics, test_num_steps=test_net(best_model,self.test_generator,device,criterion)

        
        print_metrics(test_metrics,test_num_steps)
        test_list.append(test_loss)
     

        for k, vl in test_metrics.items():      
            test_list.append(vl)              # append metrics results in a list
  
  
  
        ##################################################################
        # Writing test results to a csv file 
        ##################################################################

        key_name=['Test_loss','Test_micro_precision','Test_micro_recall','Test_micro_f1','Test_macro_precision','Test_macro_recall','Test_macro_f1','Test_mcc']
        try:

                with open(filename+str('.csv'), 'a',newline="") as f:
                    wr = csv.writer(f,delimiter=",")
                    wr.writerow(key_name)
                    zip(test_list)
                    wr.writerow(test_list) 
                    wr.writerow("") 
        except IOError:
                print("I/O Error")  
        return val_metrics, test_metrics
        
        
                       
         
                
if __name__=="__main__":
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device: {device}')
    logging.info(f'''Starting training:
                 Epochs: {max_epochs}
                 Batch Size: {batch_size}
                 Learning Rate: {lr}''')
    t=train()
    t.train_net()
  




