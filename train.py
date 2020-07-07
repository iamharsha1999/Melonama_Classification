from tqdm import tqdm
import torch
from torch.optim import Adam
from Model import Resnet34, SemiUNet
from Dataset import Melonama_Data
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

MODEL = Resnet34()
DEVICE = torch.device('cuda:0')
MODEL = MODEL.to(DEVICE)
LOSS_CRITERION = nn.BCEWithLogitsLoss()
OPTIMIZER = Adam(MODEL.parameters(), lr = 1e-4)
WEIGHT_BASE_PATH = 'weights/Resnet34/'
BATCH_SIZE = 32
EPOCHS = 50


def train(epoch,fold = 0, MODEL = MODEL, DEVICE = DEVICE, OPTIMIZER = OPTIMIZER, BATCH_SIZE = BATCH_SIZE, LOSS_CRITERION = LOSS_CRITERION):

    train_data = Melonama_Data(fold = fold)
    val_data = Melonama_Data(mode = 'val',fold = fold)

    train_batches = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle = True, num_workers=6)

    ## Number of Batches and Final Batch Size
    nob = math.ceil(len(train_data)/BATCH_SIZE)
    fbs = len(train_data)%BATCH_SIZE

    ## Epoch Accuracy
    epoch_acc  = {'train_acc':0, 'val_acc':0}

    ## Epoch Loss
    epoch_loss = {'train_loss':0, 'val_loss':0}

    ## Epoch AUC Score
    epoch_roc_auc =  {'train_auc':0, 'val_auc':0}
    
    correct = 0

    train_preds = torch.zeros(len(train_data), 1)
    MODEL = MODEL.train()

    ## Looping through the Training Data
    train_targets = torch.zeros(len(train_data), 1)
    start_idx = 0
    for b_cnt, batch in enumerate(tqdm(train_batches)):

        images = batch['image'].to(DEVICE)
        target = batch['class'].to(DEVICE)

        OPTIMIZER.zero_grad()
        output = MODEL(images)
        loss = LOSS_CRITERION(output, target)
        loss.backward()
        OPTIMIZER.step()

        epoch_loss['train_loss'] += float(loss)
        output = (output>0.5).float()

        if b_cnt == nob-1:
          train_preds[start_idx:start_idx + fbs] = output.cpu()
          train_targets[start_idx:start_idx + fbs] = target.cpu()

        else:
          train_preds[start_idx:start_idx + BATCH_SIZE] = output.cpu()
          train_targets[start_idx:start_idx + BATCH_SIZE] = target.cpu()

          start_idx += BATCH_SIZE        


    epoch_acc['train_acc'] = accuracy_score(train_targets, train_preds)
    epoch_roc_auc['train_auc'] = roc_auc_score(train_targets, train_preds)
    print('TRAINING_LOSS: {} | TRAINING ACC: {} '.format(epoch_loss['train_loss'], epoch_acc['train_acc']))
   
    correct = 0

    val_preds = torch.zeros(len(val_data), 1)
    val_targets = torch.zeros(len(val_data), 1)
    ## Validation Data
    MODEL = MODEL.eval()
    for i in tqdm(range(len(val_data))):
        inpt = val_data[i]['image']
        inpt = inpt.view(1,3,156,156).to(DEVICE)
        target = val_data[i]['class'].to(DEVICE)       

        with torch.no_grad():
            output = MODEL(inpt)
        loss = LOSS_CRITERION(output, target)
        output = (output>0.5).float()
        epoch_loss['val_loss'] += float(loss)
        val_preds[i] = int(output[0].cpu())
        val_targets[i] = int(target.cpu())
        

    epoch_acc['val_acc'] = accuracy_score(val_targets, val_preds)
    epoch_roc_auc['val_auc'] = roc_auc_score(val_targets, val_preds)

    print('VAL_LOSS: {} | VAL_ACC: {}'.format(epoch_loss['val_loss'], epoch_acc['val_acc']))
    print('TRAIN_AUC: {} | VAL_AUC: {}'.format(epoch_roc_auc['train_auc'], epoch_roc_auc['val_auc']))

    print('Saving Model' +  WEIGHT_BASE_PATH + 'EPOCH:{}'.format(epoch+1) + '.pt')
    torch.save(MODEL.state_dict(), WEIGHT_BASE_PATH + 'EPOCH:{}'.format(epoch+1) + '.pt')

    ## Update the Learning Rate
    SCHEDULER.step(epoch_roc_auc['val_auc'])

for i in range(EPOCHS):
    print('EPOCH: {}'.format(i+1))
    train(fold = i%5, epoch = i+1)