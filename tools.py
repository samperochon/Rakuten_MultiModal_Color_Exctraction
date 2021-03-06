import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from PIL import Image

from glob import glob 
import json
import os
import shutil
import time


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import f1_score
from sklearn import svm

from transformers import BertTokenizer, BertModel

# For the ViT
import timm

from imgaug import augmenters as iaa



from sklearn import svm
import numpy as np


class MixModel(torch.nn.Module):

    def init(self, relaxation_type):
        super(CustomModel, self).init()
        self.relaxation_type = relaxation_type

        self.Bert   =  BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        self.ViT    =  timm.create_model('vit_large_patch16_224', pretrained=True)
        self.ViT.model.head = nn.Identity()
        for param in self.Bert.parameters(): 
                param.requires_grad = False
        for param in self.ViT.parameters(): 
                param.requires_grad = False

        self.fc1 = torch.nn.Linear(1792, 700)
        self.fc2 = torch.nn.Linear(700, 200)
        self.fc3 = torch.nn.Linear(200, 19)
        self.BN =  nn.BatchNorm1d(num_features=700)
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, tokens_tensor , image ):
        text_features  = self.encoder.forward(input_ids=tokens_tensor,return_dict=True)
        text_features  = text_features['pooler_output'].squeeze(0)

        image_features = self.model.forward(image)

        features = torch.cat((text_features,image_features),1)

        features = F.relu(self.fc1(features))
        features = nn.Dropout(F.relu(self.BN(self.fc2(features))))
        logits   = self.fc3(features)

        return logits

    def relaxation(self):
        if self.relaxation_type=="soft":
            for name,param in self.Bert.named_parameters():
                if name.startswith('encoder.encoder.layer.11') or name.startswith('encoder.pooler.dense'):
                    param.requires_grad = True
            for name,param in self.ViT.named_parameters():
                if name.startswith('model.blocks.20') or name.startswith('model.blocks.21') or name.startswith('model.blocks.22') or name.startswith('model.blocks.23') :
                    param.requires_grad = True

        elif self.relaxation_type=="hard":
            for param in self.encoder.parameters(): 
                param.requires_grad = True
                

class CustomBertModel(torch.nn.Module):

    def __init__(self, dropout = 0.3, batchnorm = False):
        super(CustomBertModel, self).__init__()

        self.dropout = dropout
        self.batchnorm = batchnorm
        self.encoder   =  BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        for param in self.encoder.parameters(): 
                param.requires_grad = False
        self.fc1 = torch.nn.Linear(768, 450)
        self.fc2 = torch.nn.Linear(450, 200)
        self.fc3 = torch.nn.Linear(200, 19)
        if self.batchnorm:
            self.batchNorm = nn.BatchNorm1d(num_features=450)
        self.dropout = nn.Dropout(0.15)


    def forward(self, tokens_tensor):
        text_features  = self.encoder.forward(input_ids=tokens_tensor,return_dict=True)
        text_features  = text_features['pooler_output'].squeeze(0)
        if self.batchnorm:
            text_features = self.dropout(F.relu(self.batchNorm(self.fc1(text_features))))
        else:
            text_features = F.relu(self.dropout(self.fc1(text_features)))
        text_features = F.relu(self.dropout(self.fc2(text_features)))
        logits = self.fc3(text_features)

        return logits

    def relaxation(self,type_relax):
        if type_relax=="soft":
            for name,param in self.named_parameters():
                if name.startswith('encoder.encoder.layer.11') or name.startswith('encoder.pooler.dense'):
                    param.requires_grad = True
        elif type_relax=="hard":
            for param in self.encoder.parameters(): 
                param.requires_grad = True
    
    
    
    
class TextDataset(torch.utils.data.Dataset):

    def __init__(self, XTrain, Ytrain_label, item_caption , troncature, test=False):
        
        self.troncature=troncature
        self.test = test
        #Load pre-computed tensors
        if item_caption:
                self.text_name = XTrain['item_caption']
        else:
                self.text_name = XTrain['item_name']
        # self.text_caption = XTrain['item_caption']
        self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        self.labels = Ytrain_label if not test else None
        #torch.cat((Xtrain_item_name,Xtrain_item_caption),0)
    def __len__(self):
        return len(self.text_name)

    def __getitem__(self, idx):
        
        if str(self.text_name.iloc[idx])=='nan':
            tokenized_text_name=self.tokenizer.tokenize('何もない')
        elif (self.tokenizer.tokenize(self.text_name.iloc[idx])==[]):
            tokenized_text_name=self.tokenizer.tokenize('何もない')
        else:
            tokenized_text_name    = self.tokenizer.tokenize(self.text_name.iloc[idx])
        
            #tokenized_text_name    = self.tokenizer.tokenize(self.text_name.iloc[idx])
    #    tokenized_text_caption = self.tokenizer.tokenize(str(self.text_caption[idx])) #sometimes there is no caption so str() is required

        indexed_tokens_name    = self.tokenizer.convert_tokens_to_ids(tokenized_text_name)
      #  indexed_tokens_caption = self.tokenizer.convert_tokens_to_ids(tokenized_text_caption)
        
        tokens_tensor_name     = torch.tensor([indexed_tokens_name])
       # tokens_tensor_caption  = torch.tensor([indexed_tokens_caption])

        tokens_tensor_name    = tokens_tensor_name[0,:self.troncature] #to prevent tokens sequence longer than 512 tokens
      #  tokens_tensor_caption = tokens_tensor_caption[0,:412] #to prevent tokens sequence longer than 512 tokens

        #return  torch.cat((tokens_tensor_name,tokens_tensor_caption),0),self.labels[:,idx]
        if not self.test:
            return  tokens_tensor_name, self.labels[:,idx]
        else:
            return  tokens_tensor_name

def generate_batch(data_batch):
    tokens_batch = [item[0] for item in data_batch]
    labels_batch = [item[1] for item in data_batch]
    tokens_batch = pad_sequence(tokens_batch,batch_first=True, padding_value=1)
    labels_batch = pad_sequence(labels_batch,batch_first=True, padding_value=0) #just to have tensor instead of list

    return tokens_batch, labels_batch

def generate_batch_Test(data_batch):
    tokens_batch = [item for item in data_batch]
    tokens_batch = pad_sequence(tokens_batch,batch_first=True, padding_value=1)
    return tokens_batch

class MixedDataset(torch.utils.data.Dataset):
    
    def __init__(self, XTrain, Ytrain_label, item_caption , troncature, test=False,  X_df=None, Y_df=None, data_path=None, dataset_type='training', augment=True, model_type='inception', use_text=False):
        
        self.imageDataset = ImageDataset(X_df, Y_df, data_path, dataset_type=dataset_type, augment=augment, model_type=model_type, use_text=use_text)
        self.textDataset = TextDataset(XTrain, Ytrain_label, item_caption , troncature, test=test)
        
        
    def __len__(self):
        return len(self.textDataset)
    
    def __getitem__(self, index):
        
        image, label = self.imageDataset[index]
        text, _ = self.textDataset[index]
        
        return(text, image, label)

# Create Data Loader
class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, X_df, Y_df, data_path, dataset_type='training', augment=True, model_type='inception', use_text=False):
        
        # Datapath to retrieve each image tensor in the __getitem__ method
        self.data_path = data_path
        
        # DataFrame 
        self.X_df = X_df
        self.Y_df = Y_df
        
        if model_type in ['DenseNet', 'vit', 'resnet']:
            # Transformations for the Densenet121 and ViT
            if augment and dataset_type=='training':
                self.transformImage = transforms.Compose([ImgAugTransform(),
                                                            lambda x: Image.fromarray(x),
                                                            transforms.Resize((224,224)),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
                
            else:
                self.transformImage = transforms.Compose([transforms.Resize((224,224)),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            
      

        elif model_type=='inception':
            
            if augment and dataset_type=='training':
                self.transformImage = transforms.Compose([ImgAugTransform(),
                                                           lambda x: Image.fromarray(x),
                                                            transforms.Resize((299,299)),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
                
            else:
                self.transformImage = transforms.Compose([transforms.Resize((299,299)),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im
        
    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, index):
        
        try:
            # Load data point
            imPath = os.path.join('/content','images',self.X_df.iloc[index]["image_file_name"])
            image = self.loadImage(imPath)
            image = self.transformImage(image)

            #Load target
            colorTag = self.Y_df.iloc[index]["color_tags_num"]
        except RuntimeError:
            print('Error leading: {}'.format(imPath))
            index+=1
            # Load data point
            imPath = os.path.join('/content','images',self.X_df.iloc[index]["image_file_name"])
            image = self.loadImage(imPath)
            image = self.transformImage(image)

            #Load target
            colorTag = self.Y_df.iloc[index]["color_tags_num"]
        return  image, torch.tensor(colorTag) 
# Create Data Loader
class ImageDatasetTEST(torch.utils.data.Dataset):

    def __init__(self, X_df, data_path, model_type='inception'):
        
        # Datapath to retrieve each image tensor in the __getitem__ method
        self.data_path = data_path
        
        # DataFrame 
        self.X_df = X_df
        
        if model_type in ['DenseNet', 'vit']:
            # Transformations for the Densenet121 and ViT
            self.transformImage = transforms.Compose([transforms.Resize((224,224)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            
            
        elif model_type=='resnet':
            
            self.transformImage = transforms.Compose([transforms.Resize((224,224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])


        elif model_type=='inception':
            
            self.transformImage = transforms.Compose([transforms.Resize((299,299)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            

    def loadImage(self, path):
        try:
            im = Image.open(path).convert('RGB')
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im
        
    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, index):
        
        try:
            # Load data point
            imPath = os.path.join('/content','images',self.X_df.iloc[index]["image_file_name"])
            image = self.loadImage(imPath)
            image = self.transformImage(image)

        except RuntimeError:
            print('Error leading: {}'.format(imPath))
            index+=1
            # Load data point
            imPath = os.path.join('/content','images',self.X_df.iloc[index]["image_file_name"])
            image = self.loadImage(imPath)
            image = self.transformImage(image)

        return  image
    
 
    
   


class ImgAugTransform:
    """
      Uses imgaug library to perform some custom image augmentation
  """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(
                0.25,
                iaa.OneOf([
                    iaa.Dropout(p=(0, 0.1)),
                    iaa.CoarseDropout(0.1, size_percent=0.5),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.AverageBlur(k=(2, 7)),
                ])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


    
    
 
# Store important metrics
class Metric(object):
    """Computes and stores the average and current value"""
    def __init__(self, name=''):
        self.name=name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.values = []

    def update(self, val, n=1):
        self.values.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def merge(self, metric_to_merge):
        """
            Aims at merging the same metric (e.g loss, f1 score, etc.) computed during one epoch, into the agregated one. 
            ex: self.losses_experiment.merge(loss_epoch)
        """
        self.val = metric_to_merge.val
        self.sum += metric_to_merge.sum
        self.count += metric_to_merge.count
        self.values.extend(metric_to_merge.values)
        self.avg = self.sum/self.count
        
    def load(self, values):
        self.val = values[-1]
        self.sum = np.sum(values)
        self.values = values
        self.count = len(values)
        self.avg = self.sum/self.count

        
    def plot(self, savefig_path = None):
        plt.figure(figsize=(30,12))
        plt.plot(self.values)
        plt.title('Values of across epochs\nMean:{}'.format(self.avg), fontsize=20, weight='bold')
        plt.xlabel('Epochs', fontsize=18, weight='bold')
        plt.ylabel(self.name+' evolution', fontsize=18, weight='bold')
        if savefig_path is not None:
            plt.savefig(savefig_path, dpi=100, bbox_inches='tight')
            
            
def generateNumericalLabelsDataset(root_data, path_to_save='./YTrain.csv'):
    '''
        This function aims at adding the one hot encoded label columns to the Y dataset, and save the resulting new csv. 
    '''
    
    YTrain = pd.read_csv(os.path.join(root_data,'y_train_Q9n2dCu.csv'), index_col=0)
    YTrain['color_tags'] = YTrain['color_tags'].apply(lambda x: ast.literal_eval(x))

    # Encode the multiclass color labels using one hot encoding (Vector of 19 zeros with 1 when the color is present)
    idx2color={ 0: "Beige",1:"Black",2:"Blue",3:"Brown",4:"Burgundy",5:"Gold",6:"Green",7:"Grey",
                     8:"Khaki",9:"Multiple Colors",10:"Navy",11:"Orange",12:"Pink",
                     13:"Purple",14:"Red",15:"Silver",16:"Transparent",17:"White",18:"Yellow"}
    color2idx = {v: k for k, v in idx2color.items()}

    YTrain['color_tags_num'] = np.empty((len(YTrain), 0)).tolist()
    for idx, row in YTrain.iterrows():
        one_hot_encoding = 19*[0]
        for color in row['color_tags']:
            one_hot_encoding[color2idx[color]] = 1

        YTrain.loc[idx,'color_tags_num'] = np.array(one_hot_encoding)
        
    YTrain.to_csv(path_to_save)
    return YTrain


def displayPredictions(model,root_data, n=None):
    '''
        From a model, show n examples with prediction and label.
        Ex: 
        
        # Model creation
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        model.eval()
        model.classifier = nn.Linear(1024, 19) 
        displayPredictions(model, n=4)
    '''
    XTrain = pd.read_csv(os.path.join(root_data,'X_train_12tkObq.csv'), index_col=0)

    randomImages = np.random.choice(np.arange(len(XTrain)), size=n, replace = False)

    transformation = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # Encode the multiclass color labels using one hot encoding (Vector of 19 zeros with 1 when the color is present)
    idx2color={ 0: "Beige",1:"Black",2:"Blue",3:"Brown",4:"Burgundy",5:"Gold",6:"Green",7:"Grey",
                 8:"Khaki",9:"Multiple Colors",10:"Navy",11:"Orange",12:"Pink",
                 13:"Purple",14:"Red",15:"Silver",16:"Transparent",17:"White",18:"Yellow"}
    
    fig = plt.figure(figsize=(40,10*(n//2)))
    axes = []
    j=0
    for i in range(n):
        axes.append(fig.add_subplot(n//2,2,j+1))
        path = os.path.join('/content/images',XTrain.iloc[randomImages[i]]["image_file_name"])
        image_raw = Image.open(path).convert('RGB')
        image_tranformed = transformation(image_raw)
        pred = model(image_tranformed[None,:,:,:].cuda())
        pred = (torch.sigmoid(pred).detach().numpy()>.5).astype(int)
        axes[j].imshow(image_raw)
        labels_pred = [idx2color[idx] for idx in np.where(pred[0]==1)[0] if len(np.where(pred[0]==1)[0])>0]
        
        axes[j].set_title('Prediction: {}\nGT: {}'.format(labels_pred,YTrain.iloc[randomImages[i]]["color_tags"]), weight='bold', fontsize=22)
        axes[j].set_xticks([])
        axes[j].set_yticks([])

        j+=1
    plt.tight_layout()
    return

def lookExperiments(model_type='vit', experiment_num = None):

    param_dict = ['model_type', 'augment', 'batch_size', 'best_f1', 'betas', 'criterion', 'dropout', 'epochs', 'head_model', 'item_caption', 'lr', 'lr_decay_epoch', 'pos_weight', 'relaxation_type', 'split_train_val', 'troncature', 'use_text', 'weight_decay']


    if experiment_num is None:
        vit_exp_list = glob(os.path.join(ROOT_DATA, 'experiments',model_type,'*'))
    else:
        vit_exp_list = [os.path.join(ROOT_DATA, 'experiments',model_type,str(experiment_num))]
    print(vit_exp_list)
    for path in vit_exp_list:
        print(path)
        exp_num = os.path.basename(path)
        if exp_num=='0':
            continue
        path_json = os.path.join(path, 'experiment_log.json')

        print('\nLOOKING AT EXPERIMENT : {}'.format(exp_num))
        #try
        with open(path_json, 'r') as json_file:

            data = json.load(json_file)
                
            if 'losses_train_epoch' not in data.keys():
              print("Don't have loss for exp num: {}".format(exp_num))
              continue

            text=""
            count=0
            for param in param_dict:
              if param in data.keys():
                text+=param+': {} | '.format(data[param])
                count+=1
              if count%5==0:
                text+='\n'
            n_epoch = len(data['losses_train_epoch'])

            plt.figure(figsize=(12,5))
            plt.plot(data['losses_train_batch'], color = 'b', alpha = 0.5, label = 'Training loss (batch)')
            plt.scatter(int(len(data['losses_train_batch'])/n_epoch)*np.arange(n_epoch), data['losses_val_epoch'], s=50, color = 'r', label = 'Validation loss (epoch)')
            plt.scatter(int(len(data['losses_train_batch'])/n_epoch)*np.arange(n_epoch), data['losses_train_epoch'], s=50, color = 'g', label = 'Training loss (epoch)')
            plt.plot(int(len(data['losses_train_batch'])/n_epoch)*np.arange(n_epoch), data['losses_train_epoch'], color = 'g')
            plt.title('Evolution of the losses during training - Experiment : {}\n'.format(os.path.basename(path))+text, fontsize=17, weight='bold')
            plt.xlabel('Epochs', fontsize=18, weight='bold')
            plt.ylabel('Loss evolution', fontsize=18, weight='bold')
            plt.xticks(ticks = int(len(data['losses_train_batch'])/n_epoch)*np.arange(n_epoch), labels=[str(i) for i in np.arange(1,n_epoch+1)])
            _ = plt.legend()
            plt.show()

            plt.figure(figsize=(12,5))
            plt.plot(data['f1_train_batch'], color = 'b', alpha = 0.5, label = 'Training loss (batch)')
            plt.scatter(int(len(data['f1_train_batch'])/n_epoch)*np.arange(n_epoch), data['f1_val_epoch'], s=50, color = 'r', label = 'Validation F1-Score (epoch)')
            plt.scatter(int(len(data['f1_train_batch'])/n_epoch)*np.arange(n_epoch), data['f1_train_epoch'], s=50, color = 'g', label = 'Training F1-Score (epoch)')
            plt.plot(int(len(data['f1_train_batch'])/n_epoch)*np.arange(n_epoch), data['f1_train_epoch'], color = 'g')
            plt.title('Evolution of the F1 score during training - Experiment : {}\n'.format(os.path.basename(path))+text, fontsize=17, weight='bold')
            plt.xlabel('Epochs', fontsize=18, weight='bold')
            plt.ylabel('F1 evolution', fontsize=18, weight='bold')
            plt.xticks(ticks = int(len(data['losses_train_batch'])/n_epoch)*np.arange(n_epoch), labels=[str(i) for i in np.arange(1,n_epoch+1)])
            _ = plt.legend()
            plt.ylim((0,1))
            plt.show()


            print('Number of epoch: {}'.format(n_epoch))
        #except KeyError:
            #print(path)