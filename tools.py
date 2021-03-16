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
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from sklearn.metrics import f1_score

# For the ViT
import timm

from imgaug import augmenters as iaa


# Create Data Loader
class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, X_df, Y_df, data_path, dataset_type='training', augment=True, model_type='inception'):
        
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
        return len(self.X_df)-1

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
            self.transformImage = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            
            
        elif model_type=='resnet':
            
            self.transformImage = transforms.Compose([transforms.Resize((224,224)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])


        elif model_type=='inception':
            
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
        return len(self.X_df)-1

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
    
    
class TextDataset(torch.utils.data.Dataset):

    def __init__(self,features,labels):
      #Load pre-computed tensors
      self.features=features
      self.labels=labels
        
    def __len__(self):
        return self.features.shape[1]

    def __getitem__(self, idx):

        return  self.features[:,idx],self.labels[:,idx]
        


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
