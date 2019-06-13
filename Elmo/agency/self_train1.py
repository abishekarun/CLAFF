import pandas as pd
labeled_train = "https://raw.githubusercontent.com/kj2013/claff-happydb/master/data/TRAIN/labeled_10k.csv"
unlabeled_train = "https://raw.githubusercontent.com/kj2013/claff-happydb/master/data/TRAIN/unlabeled_70k.csv"
unlabeled_test = "https://raw.githubusercontent.com/kj2013/claff-happydb/master/data/TEST/unlabeled_17k.csv"
labeled_data = pd.read_table(labeled_train,sep=',',index_col=False, error_bad_lines=False) 
test = pd.read_table(unlabeled_train,sep=',',index_col=False, error_bad_lines=False) 
path ='/home/ch13b009/ENV/elmo/agency_test1/'

import numpy as np
from sklearn.model_selection import train_test_split
train, valid = train_test_split(labeled_data, test_size=0.2, random_state=0, stratify=labeled_data['agency'])

train = train[['hmid','moment','agency']]
valid = valid[['hmid','moment','agency']]
test = test[['hmid','moment']]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train['agency'] = le.fit_transform(train['agency'])
valid['agency'] = le.transform(valid['agency'])

train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

import torch 
import torch.nn as nn
import torchvision.datasets as datasets
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter,OrderedDict
import random
import math

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import spacy
import re

# tokenizer function using spacy
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
def tokenizer(s): 
    sample = [tweet_clean(y).split() for y in s]
    character_ids = batch_to_ids(sample)
    embeddings = elmo(character_ids)
    return embeddings['elmo_representations'][1] 

def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()

class CustomDataset(Dataset):
  def __init__(self,x,y):
    super(CustomDataset,self).__init__()
    self.x=x
    self.y=y
  def __getitem__(self,ids):
    df = self.x.iloc[ids]
    label = self.y.iloc[ids]
    sample = {'tweet': df, 'label': label}
    return sample
  def __len__(self):
    return len(self.x)

data=CustomDataset(train.moment,train.agency)
valid_data = CustomDataset(valid.moment,valid.agency)
data_iterator = torch.utils.data.DataLoader(data,batch_size=64,shuffle=False)
valid_iterator = torch.utils.data.DataLoader(valid_data,batch_size=len(valid),shuffle=False)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
 
class LSTM(nn.Module):
    def __init__(self, hidden_dim, n_lstm_layers,emb_dim, dropout_p):
        super(LSTM,self).__init__() # don't forget to call this!
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=n_lstm_layers)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1=nn.Linear(hidden_dim,64)
        self.predictor=nn.Linear(64,1)
 
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
  
    def forward(self, seq):
        output, (final_hidden_state, final_cell_state) = self.encoder(seq)
        output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
        attn_output = self.attention_net(output, final_hidden_state)
        attn_output= self.dropout(attn_output)
        fc_output= self.fc1(attn_output)
        fc_output= self.dropout(fc_output)
        preds=self.predictor(fc_output)
        return torch.sigmoid(preds)

embed_size = 1024
nh = 512
drop=0.5
n_lstm_layers=1
model = LSTM(nh,n_lstm_layers,embed_size,drop)
model.to(device)

from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from torch.optim.lr_scheduler import StepLR
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_func = nn.BCELoss()
epochs = 20
batch_size = 64 
scheduler = StepLR(opt, step_size=5, gamma=0.1)
text_file = open(path+"self_train_results.txt", "w")

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    iterations=0
    scheduler.step()
    model.train() # turn on training mode
    for batch in data_iterator: # thanks to our wrapper, we can intuitively iterate over our data!
        iterations=iterations+1
        x = tokenizer(batch['tweet'])
        y=batch['label']
        x=torch.FloatTensor(x)
        x=torch.transpose(x,0,1)
        opt.zero_grad()
        x=x.to(device)
        y=y.to(device)
        predicted = model(x)
        predicted=predicted.view(-1)
        loss = loss_func(predicted,y.float())
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(train)
 
    # calculate the validation loss for this epoch
    val_loss = 0.0
    list_scores=[]
    model.eval() # turn on evaluation mode
    for batch in valid_iterator:
        x = tokenizer(batch['tweet'])
        y=batch['label']
        x=torch.FloatTensor(x)
        x=torch.transpose(x,0,1)
        x=x.to(device)
        y=y.to(device)
        preds = model(x)
        loss = loss_func(preds,y.float())
        val_loss += loss.item() * x.size(0)
        preds = [int(x>0.5) for x in preds]
        temp=accuracy_score(preds,y)
        auc= roc_auc_score(preds,y)
        f1 = f1_score(preds,y,average = 'binary')
        list_scores.append(temp)
    
    val_loss = val_loss / len(valid)
    torch.save(model.state_dict(), path + 'eval'+str(epoch)+'.pt')
    text_file.write("\nEpoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}\n, Validation AUC: {:.4f}\n Validation F1 score: {:.4f}\n ".format(epoch,epoch_loss,val_loss,temp,auc,f1))
text_file.close()

ind= np.argmax(list_scores)
model.load_state_dict(torch.load(path + 'eval'+str(ind+1)+'.pt'))
model.eval()
test_data = CustomDataset(test.moment,test.hmid)
test_iterator = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=False)

test_ypreds = []
test_npreds = []
unsure_preds=[]
for batch in test_iterator:
    x = tokenizer(batch['tweet'])
    y=batch['label']
    x=torch.FloatTensor(x)
    x=torch.transpose(x,0,1)
    x=x.to(device)
    y=y.to(device)
    preds = model(x)
    preds=preds.cpu().detach().numpy()
    unsure_preds.append(np.extract( (preds>=0.3) & (preds<=0.7),y))
    test_ypreds.append(np.extract(preds>0.9,y))
    test_npreds.append(np.extract(preds<0.1,y))

import itertools
pred_yes = list(itertools.chain.from_iterable(test_ypreds))
pred_no = list(itertools.chain.from_iterable(test_npreds))

addon_yes = test.loc[test['hmid'].isin(pred_yes)]
addon_yes['agency'] = 1

addon_no = test.loc[test['hmid'].isin(pred_no)]
addon_no['agency'] = 0

test = test.loc[~test['hmid'].isin(pred_yes)]
test = test.loc[~test['hmid'].isin(pred_no)]

train = pd.concat([train,addon_yes,addon_no])

train.to_csv(path+"train_data2.csv",index=False)
test.to_csv(path+"test_data2.csv",index=False)
valid.to_csv(path+"valid_data.csv",index=False)