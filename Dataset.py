import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from albumentations import HorizontalFlip, VerticalFlip, GaussNoise, MedianBlur, RandomContrast, Rotate,  Compose



class Melonama_Data(Dataset):

    def __init__(self, img_path = 'jpeg/train', mode = 'train', csv_data = 'train.csv'):
        super().__init__()

        ## Read and Preprocess the Data
        self.df = pd.read_csv(csv_data)

        if csv_data == 'train.csv':
            self.mode = 'train'
            self.target =  self.df.iloc[:,-1].values.reshape(-1,1)
            self.df.dropna(inplace = True)
        elif csv_data == 'test.csv':
            self.mode = 'test'



        self.input = self.df.iloc[:,:-1]

        self.img_ids = self.input.image_name.values
        ## Image Path
        self.img_path = img_path


        ## Augmentation for Data
        self.aug = Compose([HorizontalFlip(p=0.5),
                            VerticalFlip(p=0.5),
                            MedianBlur(p=0.5),
                            RandomContrast(p=0.3),
                            Rotate(p=0.3)
                            ])


    def __len__(self):

        return len(self.img_ids)

    def __getitem__(self, idx):

        id = self.img_ids[idx]
        img  = cv2.imread(self.img_path + '/' + str(id) + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img, dtype=torch.float32).view(-1,156,156)

        if self.mode == 'train':

            img = self.aug(image = img)["image"]
            target_class = torch.tensor(self.target[idx], dtype=torch.float32)

            return {
                'image': img,
                'class': target_class }

        elif self.mode == 'test':

            return {
                'image': img }
