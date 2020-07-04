from tqdm import tqdm
import torch
from torch.optim import Adam
from Model import Resnet34, SemiUNet
from Dataset import Melonama_Data
from torch.utils.data import DataLoader
import torch.nn as nn

MODEL = Resnet34()
DEVICE = torch.device('cuda:0')
MODEL = MODEL.to(DEVICE)
LOSS_CRITERION = nn.BCELoss()
OPTIMIZER = Adam(MODEL.parameters(), lr = 1e-4)
WEIGHT_BASE_PATH = 'weights/Resnet34/'
BATCH_SIZE = 32
EPOCHS = 50

def train(fold = 0):

    train_data = Melonama_Data(fold = fold)
    val_data = Melonama_Data(mode = 'val',fold = fold)

    train_batches = DataLoader(DATA, batch_size=BATCH_SIZE, shuffle = True, num_workers=6)

   
    epoch_acc  = {'train_acc':0, 'val_acc':0}

    epoch_loss = {'train_loss':0, 'val_loss':0}
    
    correct = 0

    MODEL = MODEL.train()
    ## Looping through the Training Data
    for batch in tqdm(train_batches):

        images = batch['image'].to(DEVICE)
        target = batch['class'].to(DEVICE)

        OPTIMIZER.zero_grad()
        output = MODEL(images)
        loss = LOSS_CRITERION(output, target)
        loss.backward()
        OPTIMIZER.step()

        epoch_loss['train_loss'] += float(loss)
        output = (output>0.5).float()
        correct += (output == target).float().sum()


    epoch_acc['train_acc'] = float(correct/len(val_data))
    print('TRAINING_LOSS: {} TRAINING ACC: {}'.format(epoch_loss['train_loss'], epoch_acc['train_acc']))
   
    correct = 0

    ## Validation Data
    MODEL = MODEL.eval()
    for i in tqdm(range(len(val_data))):

       images = images.view(1,3,156,156).to(DEVICE)
        target = i['class'].to(DEVICE)       

        with torch.no_grad()
            output = MODEL(inpt)
        loss = LOSS_CRITERION(output, target)
        output = (output>0.5).float()
        epoch_loss['val_loss'] += float(loss)
        correct += (output == target).float().sum()

    epoch_acc['val_acc'] = float(correct/len(val_data))
    print('VAL_LOSS: {} VAL_ACC: {}'.format(epoch_loss['val_loss'], epoch_acc['val-loss']))

    print('Saving Model' +  WEIGHT_BASE_PATH + '{}_Acc:{}'.format(epoch+1, epoch_acc['train_acc']) + '.pt')
    torch.save(MODEL.state_dict(), WEIGHT_BASE_PATH + '{}_Acc:{}'.format(epoch+1, epoch_acc['train_acc']) + '.pt')



for i range(EPOCHS):
    train(fold = i%5)