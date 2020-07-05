import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from albumentations import HorizontalFlip, VerticalFlip, GaussNoise, MedianBlur, RandomContrast, Rotate,  Compose, Normalize
from sklearn.model_selection import StratifiedKFold

class Melonama_Data(Dataset):

    def __init__(self, fold, mode = 'train'):
        super().__init__()

        self.fold = fold 
        self.mode = mode

        ## Train Data
        if self.mode == 'train':
                      
            self.df = pd.read_csv('train_kfold.csv')
            self.df.dropna(inplace = True)
            self.df.reset_index(inplace = True)

            ## For Training 
            self.train_img_ids = self.df[self.df.kfold!= self.fold].image_name.values
            self.train_targets  = self.df[self.df.kfold!= self.fold].target.values.reshape(-1,1)

            ## Image Path
            self.img_path = 'jpeg/Resized_Images/train'

            ## Augmentation for Data
            self.aug = Compose(
                            [HorizontalFlip(p=0.5),
                            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True),
                            VerticalFlip(p=0.5),
                            MedianBlur(p=0.5, blur_limit=5),
                            RandomContrast(p=0.5),
                            Rotate(p=0.5)
                            ])

        elif self.mode == 'val':

            self.df = pd.read_csv('train_kfold.csv')
            self.df.dropna(inplace = True)
            self.df.reset_index(inplace = True)

            ## For Validation
            self.val_img_ids = self.df[self.df.kfold == self.fold].image_name.values
            self.val_targets = self.df[self.df.kfold == self.fold].target.values.reshape(-1,1)

            ## Image Path
            self.img_path = 'jpeg/Resized_Images/train'

             ## Augmentation for Data
            self.aug = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True) ])

        ## Test Data
        elif csv_data == 'test.csv':

            self.df = pd.read_csv('test.csv')

            self.test_img_ids = self.df.image_name.values
            self.test_targets = self.df.target.values.reshape(-1,1)

            ## Image Path
            self.img_path = 'jpeg/Resized_Images/test'

            ## Augmentation for Data
            self.aug = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True) ])        


    def __len__(self):

        if self.mode == 'train':            
            return len(self.train_img_ids)

        elif self.mode == 'val':
            return len(self.val_img_ids)

        elif self.mode == 'test':
            return len(self.test_img_ids)

    def __getitem__(self, idx):

        if self.mode == 'train':

            ## Read the image -> Grayscale Image -> Apply Augmentation Style -> Tensor
            id = self.train_img_ids[idx]
            img  = cv2.imread(self.img_path + '/' + str(id) + '.jpg')
            img = self.aug(image = img)["image"]

            ## Tensors
            img = torch.tensor(img, dtype=torch.float32).view(-1,156,156)
            target_class = torch.tensor(self.train_targets[idx], dtype=torch.float32)

            return { 'image': img,
                            'class': target_class }


        elif self.mode == 'val':

            ## Read the image -> Grayscale Image -> Normalizattion-> Tensor
            id = self.val_img_ids[idx]
            img  = cv2.imread(self.img_path + '/' + str(id) + '.jpg')
            img = self.aug(image = img)["image"]

            
            ## Tensors
            img = torch.tensor(img, dtype=torch.float32).view(-1,156,156)
            target_class = torch.tensor(self.val_targets[idx], dtype=torch.float32)

            return { 'image': img,
                            'class': target_class }

        elif self.mode == 'test':

           ## Read the image -> Grayscale Image -> Normalizattion-> Tensor
            id = self.test_img_ids[idx]
            img  = cv2.imread(self.img_path + '/' + str(id) + '.jpg')
            img = self.aug(image = img)["image"]

            return {'image': img }
