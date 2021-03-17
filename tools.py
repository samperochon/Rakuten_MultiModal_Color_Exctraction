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

from transformers import BertTokenizer, BertModel

# For the ViT
import timm

from imgaug import augmenters as iaa



from sklearn import svm
import numpy as np



class StrongClassifier():
    
    def __init__(self,*argv):
        super(StrongClassifier, self).__init__()
        self.nbr_class=19

        self.strongsClass = []    # Our nbr_class SVMs
        self.weakClfs=[]          # Our weak classifier models
        
        self.weaksPredTrain=[]         # Their associated predictions on trainset (SAM)
        self.truePredTrain=[]          #   True labels of the predictions on trainset (SAM) list of 19 arrays of size len_train_dataset
        self.weaksPredVal=[]          # Their associated predictions on testset   (SAM)
        
        self.strongPred=[]        # Where we will store our true prediction of size 19*len_val_set (np.array)
        
        for _ in range(self.nbr_class):     #fill the strongsClass list with nbr_class different SVMs
            self.strongsClass.append(svm.SVC())
            
        for arg in argv:                    #fill the weakClfs list with nbr_class different SVMs
            self.weakClfs.append(arg)
    
    def generate_weaksPred(self):
        '''
        Generate the predictions of every weak classifier
        
        This direcrlty updates self.weaksPredTrain to be a list of 19 arrays of size [nbr_sample_trainset,_nbr_model]
        This direcrlty updates self.weaksPredVal to be a list of 19 arrays of size [nbr_sample_val,_nbr_model]

        Returns
        -------
        None

        '''
        for weak in self.weakClfs:
            
            print('generating')
            

    
    def fit_svm(self):
        for idx,labelwise_svm in enumerate(self.strongsClass):
            labelwise_svm.fit(self.weaksPred[idx],Y)
            #predictions=
            #return a np.array of size [len_dataset,19]
            
    #def generate_strond_preds(self):
        
        
        
        
        
        

class CustomBertModel(torch.nn.Module):

    def __init__(self):
        super(CustomBertModel, self).__init__()

        
        self.encoder   =  BertModel.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        for param in self.encoder.parameters(): 
                param.requires_grad = False
        self.fc1 = torch.nn.Linear(768, 450)
        self.fc2 = torch.nn.Linear(450, 200)
        self.fc3 = torch.nn.Linear(200, 19)


    def forward(self, tokens_tensor):
        text_features  = self.encoder.forward(input_ids=tokens_tensor,return_dict=True)
        text_features  = text_features['pooler_output'].squeeze(0)
        text_features = F.relu(self.fc1(text_features))
        text_features = F.relu(self.fc2(text_features))
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

    def __init__(self, XTrain, Ytrain_label, item_caption):
        #Load pre-computed tensors
        if item_caption:
                self.text_name = XTrain['item_caption']
        else:
                self.text_name = XTrain['item_name']
        # self.text_caption = XTrain['item_caption']
        self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-v2')
        self.labels = Ytrain_label
        #torch.cat((Xtrain_item_name,Xtrain_item_caption),0)
    def __len__(self):
        return len(self.text_name)

    def __getitem__(self, idx):
        
        if str(self.text_name[idx])=='nan':
            tokenized_text_name=self.tokenizer.tokenize('何もない')
        else:
            tokenized_text_name    = self.tokenizer.tokenize(self.text_name.iloc[idx])
        
            #tokenized_text_name    = self.tokenizer.tokenize(self.text_name.iloc[idx])
    #    tokenized_text_caption = self.tokenizer.tokenize(str(self.text_caption[idx])) #sometimes there is no caption so str() is required

        indexed_tokens_name    = self.tokenizer.convert_tokens_to_ids(tokenized_text_name)
      #  indexed_tokens_caption = self.tokenizer.convert_tokens_to_ids(tokenized_text_caption)
        
        tokens_tensor_name     = torch.tensor([indexed_tokens_name])
       # tokens_tensor_caption  = torch.tensor([indexed_tokens_caption])

        tokens_tensor_name    = tokens_tensor_name[0,:100] #to prevent tokens sequence longer than 512 tokens
      #  tokens_tensor_caption = tokens_tensor_caption[0,:412] #to prevent tokens sequence longer than 512 tokens

        #return  torch.cat((tokens_tensor_name,tokens_tensor_caption),0),self.labels[:,idx]
        return  tokens_tensor_name,self.labels[:,idx]

def generate_batch(data_batch):
    tokens_batch = [item[0] for item in data_batch]
    labels_batch = [item[1] for item in data_batch]
    tokens_batch = pad_sequence(tokens_batch,batch_first=True, padding_value=1)
    labels_batch = pad_sequence(labels_batch,batch_first=True, padding_value=0) #just to have tensor instead of list

    return tokens_batch, labels_batch




# Create Data Loader
class ImageTextDataset(torch.utils.data.Dataset):

    def __init__(self, X_df, Y_df, features, data_path, dataset_type='training', augment=True, model_type='inception'):
        
        # Datapath to retrieve each image tensor in the __getitem__ method
        self.data_path = data_path
        
        # DataFrame 
        self.X_df = X_df
        self.Y_df = Y_df
        
        # Text data
        self.features=features
        
        if model_type in ['DenseNet', 'vit']:
            # Transformations for the Densenet121 and ViT
            if augment and dataset_type=='training':
                self.transformImage = transforms.Compose([#ImgAugTransform(),
                                                            #lambda x: Image.fromarray(x),
                                                            transforms.Resize((224,224)),
                                                            #transforms.CenterCrop(224),
                                                            transforms.RandomHorizontalFlip(),
                                                            #transforms.RandomRotation(20, resample=Image.BILINEAR),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
                
            else:
                self.transformImage = transforms.Compose([transforms.Resize((224,224)),
                                                    #transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            
            
        elif model_type=='resnet':
            
            if augment and dataset_type=='training':
                self.transformImage = transforms.Compose([ImgAugTransform(),
                                                            lambda x: Image.fromarray(x),
                                                            transforms.Resize((224,224)),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomRotation(20, resample=Image.BILINEAR),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
                
                
            else:
                self.transformImage = transforms.Compose([transforms.Resize((224,224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])


        elif model_type=='inception':
            
            if augment and dataset_type=='training':
                self.transformImage = transforms.Compose([ImgAugTransform(),
                                                            lambda x: Image.fromarray(x),
                                                            transforms.Resize(299),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomRotation(20, resample=Image.BILINEAR),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
                
            else:
                self.transformImage = transforms.Compose([transforms.Resize(299),
                                                             transforms.CenterCrop(299),
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
        
        # Image, text features, labels 
        return  image, torch.tensor(colorTag), self.features[:,index]
    
    
    


# Create Data Loader
class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, X_df, Y_df, data_path, dataset_type='training', augment=True, model_type='inception', use_text=False):
        
        # Datapath to retrieve each image tensor in the __getitem__ method
        self.data_path = data_path
        
        # DataFrame 
        self.X_df = X_df
        self.Y_df = Y_df
        
        if model_type in ['DenseNet', 'vit']:
            # Transformations for the Densenet121 and ViT
            if augment and dataset_type=='training':
                self.transformImage = transforms.Compose([#ImgAugTransform(),
                                                            #lambda x: Image.fromarray(x),
                                                            transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.85, 1.15)),
                                                            transforms.RandomPerspective(distortion_scale=0.3, p=0.3, interpolation=2, fill=0),
                                                            #transforms.CenterCrop(224),
                                                            transforms.RandomHorizontalFlip(0.5),
                                                            #transforms.RandomRotation(20, resample=Image.BILINEAR),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
                
            else:
                self.transformImage = transforms.Compose([transforms.Resize((224,224)),
                                                          #transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            
            
        elif model_type=='resnet':
            
            if augment and dataset_type=='training':
                self.transformImage = transforms.Compose([ImgAugTransform(),
                                                            lambda x: Image.fromarray(x),
                                                            transforms.Resize((224,224)),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomRotation(20, resample=Image.BILINEAR),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
                
                
            else:
                self.transformImage = transforms.Compose([transforms.Resize((224,224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])


        elif model_type=='inception':
            
            if augment and dataset_type=='training':
                self.transformImage = transforms.Compose([ImgAugTransform(),
                                                            lambda x: Image.fromarray(x),
                                                            transforms.Resize(299),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.RandomRotation(20, resample=Image.BILINEAR),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
                
            else:
                self.transformImage = transforms.Compose([transforms.Resize(299),
                                                             transforms.CenterCrop(299),
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
            
            self.transformImage = transforms.Compose([transforms.Resize(299),
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



def displayPredictions(model, n=None):
    '''
        From a model, show n examples with prediction and label.
        Ex: 
        
        # Model creation
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
        model.eval()
        model.classifier = nn.Linear(1024, 19) 
        displayPredictions(model, n=4)
    '''
    
    randomImages = np.random.choice(np.arange(len(XTrain)), size=n, replace = False)

    transformation = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # Encode the multiclass color labels using one hot encoding (Vector of 19 zeros with 1 when the color is present)
    idx2color={ 0: "Beige",1:"Black",2:"Blue",3:"Brown",4:"Burgundy",5:"Gold",6:"Green",7:"Grey",
                 8:"Khaki",9:"Multiple Colors",10:"Navy",11:"Orange",12:"Pink",
                 13:"Purple",14:"Red",15:"Silver",16:"Transparent",17:"White",18:"Yellow"}
    if n is None:
        n=image.shape[0]
    
    assert n <= image.shape[0]
    
    fig = plt.figure(figsize=(40,10*(n//2)))
    axes = []
    j=0
    for i in range(n):
        axes.append(fig.add_subplot(n//2,2,j+1))
        path = os.path.join(root_data,'images',XTrain.iloc[randomImages[i]]["image_file_name"])
        image_raw = Image.open(path).convert('RGB')
        image_tranformed = transformation(image_raw)
        pred = model(image_tranformed[None,:,:,:])
        pred = (torch.sigmoid(pred).detach().numpy()>.5).astype(int)
        
        
        axes[j].imshow(image_raw)
        labels_pred = [idx2color[idx] for idx in np.where(pred[0]==1)[0] if len(np.where(pred[0]==1)[0])>0]
        
        axes[j].set_title('Prediction: {}\nGT: {}'.format(labels_pred,YTrain.iloc[randomImages[i]]["color_tags"]), weight='bold', fontsize=22)
        axes[j].set_xticks([])
        axes[j].set_yticks([])

        j+=1
    plt.tight_layout()
    return
