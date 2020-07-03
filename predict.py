import torch
from Model import Resnet34, SemiUNet
import pandas as pd
from Dataset import Melonama_Data
from tqdm import tqdm 


test_file = pd.read_csv('test.csv')

tf = Melonama_Data(img_path = 'jpeg/Resized_Images/test', csv_data = 'test.csv')

model = Resnet34()

# DEVICE = torch.device('cuda:0')
print('Building and Loading the Model...')
model = model
model.load_state_dict(torch.load('weights/Resnet34/1_Acc_0.9823245406150818.pt'))
model.eval()

predictions = []
print('Starting the predictions...')
for i in tqdm(tf):
	inpt = i['image']
	inpt = inpt.view(1,3,156,156)
	predicted = model(inpt)
	predicted = torch.round(predicted)
	predictions.append(predicted[0])

dic = { 'image_name':  test_file.image_name.values,
         'target' : predictions
}

df = pd.DataFrame(dic)
df.to_csv('submission.csv')





