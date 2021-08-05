import numpy as np
import pandas as pd
import torch
from math import log
import os
import torch.nn.functional as F 
import torch
from torch import nn,optim
from torchvision import transforms as T,datasets,models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import copy
import time
import json

"""
# Setting the seeds for reproducibility:
# Ref: https://pytorch.org/docs/stable/notes/randomness.html
# However, reproducibility in pytorch is not at all guaranteed. 
# Ref: https://stackoverflow.com/questions/56354461/reproducibility-and-performance-in-pytorch
# In fact, setting cudnn.deterministic as true will only gives us the same results on the CPU or the GPU on the same system when feeding the
  same inputs. But we still cannot guarantee same results on different machines. 
"""
def set_seed():
    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed()

start_time = time.strftime("%d%m%y_%H%M%S")

class EarlyStopping:
    # Ref: https://github.com/Bjarten/early-stopping-pytorch
    
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print, split="split"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.split = split
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'model_checkpoints/'+model.__class__.__name__+'_'+self.split+'_'+start_time+'.pt')
        self.val_loss_min = val_loss

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1) 
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)), preds

def get_metrics(labels, preds):
    fpr, tpr, thres = metrics.roc_curve(labels, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print("Classification Report: \n",metrics.classification_report(labels, preds))
    cm  = metrics.confusion_matrix(labels, preds)
    plt.figure(figsize=(12,8))
    sns.heatmap(cm, annot=True, fmt = '.4g', xticklabels = ['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'], 
               cmap='cubehelix')
    plt.xlabel('Predicted Label',fontsize=18, color = "blue")
    plt.ylabel('True Label',fontsize=18, color="red")
    plt.title('Confusion Matrix',fontsize=28)
    plt.show()
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (tp + fn)
    tnr_spec = tn / (fp + tn)
    accuracy = (np.array(preds) == np.array(labels)).sum() / len(preds)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*((precision*recall)/(precision+recall))
    print("Accuracy of the model is {:.2f}".format(accuracy))
    print("Recall (Sensitivity) of the model is {:.2f}".format(recall))
    print("Precision of the model is {:.2f}".format(precision))
    print("F1 Score of the model is {:.2f}".format(f1))
    print("False Negative Rate (Miss Rate) of the model is {:.2f}".format(fnr))
    print("True Negative Rate (Specificity) of the model is {:.2f}".format(tnr_spec))
    print("ROC-AUC of the model is {:.2f}".format(roc_auc))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return accuracy, precision, recall, roc_auc, fnr, tnr_spec
    
def get_plots(model, train_acc, val_acc, train_loss, val_loss):
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 8))
    t = f.suptitle('Performance of '+model.__class__.__name__, fontsize=28)

    minposs = val_loss.index(min(val_loss))

    ax1.plot(train_acc, label='Train Accuracy')
    ax1.plot(val_acc, label='Validation Accuracy')
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epochs')
    ax1.set_title('Accuracy on each epoch', fontsize=18)
    l1 = ax1.legend(loc="best")

    ax2.plot(train_loss, label='Train Loss')
    ax2.plot(val_loss, label='Validation Loss')
    ax2.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epochs')
    ax2.set_title('Loss on each epoch', fontsize=18)
    l2 = ax2.legend(loc="best")
    plt.show()
    
    return f
    
# Evaluating validation set   
@torch.no_grad()
def evaluate(model, val_loader, loss_fn):
    model.eval()
    outputs = [model.validation_step(batch, loss_fn) for batch in tqdm(val_loader)]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, patience, weight,loss_fn=nn.CrossEntropyLoss, opt_func=torch.optim.SGD, split="split"):
    # release all the GPU memory cache
    torch.cuda.empty_cache()
    
    # for saving the logs
    history = {}
    
    # initializing the optimizer, loss function and early stopper
    optimizer = opt_func(model.parameters(), lr)
    early_stopping = EarlyStopping(patience=patience, verbose=True, split=split)
    loss_fn = loss_fn(weight = weight) # weighted compute loss
    loss_fn_val = loss_fn
    # initialize best loss, which will be replaced with lower better loss
    best_loss = 1 
    
    # We originially divide the 5216 training images into 8 batches of 652 then it will take 8 iterations to complete 1 epoch.
    for epoch in range(epochs):
        
        # Training Phase 
        model.train() 
        
        train_outputs = []
        
        # forward + backward + optimize
        for batch in tqdm(train_loader):
             
            outputs = model.training_step(batch, loss_fn)
            
            # get the loss
            loss = outputs['train_loss']
            
            # get the train average loss and acc for each epoch
            train_outputs.append(outputs)
            # get the train average loss and acc for each epoch
            train_results = model.train_epoch_end(train_outputs)   
            
            # zero the parameter gradients
            optimizer.zero_grad() 
            
            # compute gradients
            loss.backward()
            
            # update weights
            optimizer.step()                                      # update weights 
        
        # Validation phase
        val_results = evaluate(model, val_loader, loss_fn_val)
        
        # Save best loss
        if val_results['val_loss'] < best_loss:
            best_loss = min(best_loss, val_results['val_loss'])
            best_model_wts = copy.deepcopy(model.state_dict())
            #torch.save(model.state_dict(), 'best_model.pt')
        
        # print results
        model.epoch_end(epoch, train_results, val_results)
        
        # save results to dictionary
        to_add = {'train_loss': train_results['train_loss'],
                  'train_acc': train_results['train_acc'],
                 'val_loss': val_results['val_loss'],
                  'val_acc': val_results['val_acc']}
        
        # update performance dictionary
        for key,val in to_add.items():
            if key in history:
                history[key].append(val)
            else:
                history[key] = [val]
        
        # early stop if the validation loss does not improve with patience
        early_stopping(val_results['val_loss'], model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # automatically load the best model for testing
    model.load_state_dict(best_model_wts)
    
    # save the logs
    history['model'] = model.__class__.__name__
    history['best_loss'] = best_loss
    history['epochs'] = epochs
    history['learning rate'] = lr
    
    
    with open('logs/'+model.__class__.__name__+'_'+split+start_time+'.json', 'w') as f:
        json.dump(history, f)
    
    return history, optimizer, best_loss

@torch.no_grad()
def test_predict(model, test_loader):
    model.eval()
    # perform testing for each batch
    outputs = [model.testing_step(batch) for batch in tqdm(test_loader)] 
    results = model.test_prediction(outputs)                          
    print('test_loss: {:.4f}, test_acc: {:.4f}'
          .format(results['test_loss'], results['test_acc']))
    
    return results['test_preds'], results['test_labels']


class train_test_val(nn.Module):
    
    # Ref: https://jovian.ai/learn/deep-learning-with-pytorch-zero-to-gans/lesson/lesson-4-image-classification-with-cnn
    
    # this is for loading the batch of train image and outputting its loss, accuracy & predictions
    def training_step(self, batch, loss_fn):
        images,labels = batch
        out = self(images)                                      # generate predictions
        loss = loss_fn(out, labels)      # Compute loss
        acc,preds = accuracy(out, labels)                       # calculate accuracy
        
        return {'train_loss': loss, 'train_acc':acc}
       
    # this is for computing the train average loss and acc for each epoch
    def train_epoch_end(self, outputs):
        batch_losses = [x['train_loss'] for x in outputs]       # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()           # combine losses
        batch_accs = [x['train_acc'] for x in outputs]          # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
        
        return {'train_loss': epoch_loss.item(), 'train_acc': epoch_acc.item()}
    
    # this is for loading the batch of val/test image and outputting its loss, accuracy, predictions & labels
    def validation_step(self, batch, loss_fn):
        images,labels = batch
        out = self(images)                                      # generate predictions
        loss = loss_fn(out, labels)                     # compute loss
        acc,preds = accuracy(out, labels)                       # calculate acc & get preds
        
        return {'val_loss': loss.detach(), 'val_acc':acc.detach(), 
                'preds':preds.detach(), 'labels':labels.detach()}
    # detach extracts only the needed number, or other numbers will crowd memory
    
    def testing_step(self, batch):
        images,labels = batch
        out = self(images)                                      # generate predictions
        loss = F.cross_entropy(out, labels)                     # compute loss
        acc,preds = accuracy(out, labels)                       # calculate acc & get preds
        
        return {'val_loss': loss.detach(), 'val_acc':acc.detach(), 
                'preds':preds.detach(), 'labels':labels.detach()}
    
    # this is for computing the validation average loss and acc for each epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]         # get all the batches loss
        epoch_loss = torch.stack(batch_losses).mean()           # combine losses
        batch_accs = [x['val_acc'] for x in outputs]            # get all the batches acc
        epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
        
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # this is for printing out the results after each epoch
    def epoch_end(self, epoch, train_result, val_result):
        print('Epoch [{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.
              format(epoch+1, train_result['train_loss'], train_result['train_acc'],
                     val_result['val_loss'], val_result['val_acc']))
    
    # this is for using on the test set, it outputs the average loss and acc, and outputs the predictions
    def test_prediction(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()           # combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()              # combine accuracies
        # combine predictions
        batch_preds = [pred for x in outputs for pred in x['preds'].tolist()] 
        # combine labels
        batch_labels = [lab for x in outputs for lab in x['labels'].tolist()]  
        
        return {'test_loss': epoch_loss.item(), 'test_acc': epoch_acc.item(),
                'test_preds': batch_preds, 'test_labels': batch_labels}      

# FINETUNING BASELINE MODEL
# class xray_cnnmodel(train_test_val):
#     def __init__(self):
#         super().__init__()
        
#         self.cnn_layers = nn.Sequential(
           
#             nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1, bias=False),
#             #nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             #nn.Dropout(0.2),
            
#             nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1, bias=False),
#             #nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             #nn.Dropout(0.2),

#             nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1, bias=False),
#             #nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             #nn.Dropout(0.2),
            
#             nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1, bias=False),
#             #nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             #nn.Dropout(0.2),

#             nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout(0.5),
            
#             # I have commented the below line because we can even use AdaptiveMaxPool and avoid the maths of diving the shapes as per 
#             # pooling, this means that my model can be used on any dataset, by just uncommenting this line. 
#             #nn.AdaptiveMaxPool2d(1), 
            
#             nn.Flatten(),
            
#             nn.Dropout(0.5),
            
#             nn.Linear(in_features=(512* 7 * 7), out_features= 1024),
#             nn.ReLU(),
            
#             nn.Linear(in_features=1024, out_features= 2))
          
#     def forward(self, xb):
#         return self.cnn_layers(xb)

# MY FINAL MODEL
class xray_cnnmodel(train_test_val):
    def __init__(self):
        super().__init__()
        
        self.cnn_layers = nn.Sequential( 
            #Input: -1 * 3 * 224 * 224            
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),   #Output: -1 * 32 * 224 * 224
            nn.ReLU(),   #Output: -1 * 32 * 224 * 224
            nn.MaxPool2d(kernel_size=2),   #Output: -1 * 32 * 112 * 112
        
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),   #Output: -1 * 64 * 112 * 112
            nn.ReLU(),   #Output: -1 * 64 * 112 * 112
            nn.MaxPool2d(kernel_size=2),   #Output: -1 * 64 * 56 * 56
        
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1, bias=False),   #Output: -1 * 128 * 56 * 56
            nn.ReLU(),   #Output: -1 * 128 * 56 * 56
            nn.MaxPool2d(kernel_size=2),   #Output: -1 * 128 * 28 * 28

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1, bias=False),   #Output: -1 * 256 * 28 * 28
            nn.BatchNorm2d(256),   #Output: -1 * 256 * 28 * 28
            nn.ReLU(),   #Output: -1 * 256 * 28 * 28
            nn.MaxPool2d(kernel_size=2),   #Output: -1 * 256 * 14 * 14
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1, bias=False),   #Output: -1 * 512 * 14 * 14
            nn.BatchNorm2d(512),   #Output: -1 * 512 * 14 * 14
            nn.ReLU(),   #Output: -1 * 512 * 14 * 14
            nn.MaxPool2d(kernel_size=2),   #Output: -1 * 512 * 7 * 7
            
            nn.Flatten(),   #Output: -1 * 25088
                                     
            nn.Dropout(0.5),   #Output: -1 * 25088
            nn.Linear(in_features=(512*7*7), out_features= 512),   #Output: -1 * 512
            nn.ReLU(),   #Output: -1 * 512
            
            nn.Dropout(0.3),   #Output: -1 * 512
            nn.Linear(in_features=(512), out_features= 1024),   #Output: -1 * 1024
            nn.ReLU(),   #Output: -1 * 1024
        
            nn.Dropout(0.2),   #Output: -1 * 1024
            nn.Linear(in_features=1024, out_features= 2))  #Output: -1 * 2
        
    def forward(self, xb):
        return self.cnn_layers(xb)

#Transfer Learning
class ResNet50(train_test_val):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Freeze training for all layers before classifier
        for param in self.network.fc.parameters():
            param.require_grad = False  
        num_features = self.network.fc.in_features # get number of in features of last layer
        self.network.fc = nn.Linear(num_features, 2) # replace model classifier
    
    def forward(self, xb):
        return self.network(xb)

    
    
