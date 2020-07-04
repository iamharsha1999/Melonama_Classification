import pandas as pd 
from sklearn.model_selection import  StratifiedKFold


df = pd.read_csv('train.csv')

df['kfold'] = -1            
df.reset_index(inplace = True)

img_ids  = df.image_name.values
target = df.iloc[:,-1].values

kf =  StratifiedKFold(n_splits=5)

for idx, (trn_idx, val_idx) in enumerate(kf.split(X=img_ids,  y = target)):
    df.loc[val_idx, 'kfold'] = idx

df.to_csv('train_kfold.csv') 