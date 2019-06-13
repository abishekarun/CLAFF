import pandas as pd
path = "/home/ch13b009/ENV/cove/test/"
labeled_train = "https://raw.githubusercontent.com/kj2013/claff-happydb/master/data/TRAIN/labeled_10k.csv"
unlabeled_train = "https://raw.githubusercontent.com/kj2013/claff-happydb/master/data/TRAIN/unlabeled_70k.csv"
unlabeled_test = "https://raw.githubusercontent.com/kj2013/claff-happydb/master/data/TEST/unlabeled_17k.csv"
labeled_data = pd.read_table(labeled_train,sep=',',index_col=False, error_bad_lines=False,encoding = "ISO-8859-1") 
unlabeled_data = pd.read_table(unlabeled_test,sep=',',index_col=False, error_bad_lines=False,encoding = "ISO-8859-1")

import numpy as np
from sklearn.model_selection import train_test_split
train, valid = train_test_split(labeled_data, test_size=0.2, random_state=0, stratify=labeled_data['agency'])

train = train[['hmid','moment','agency']]
valid = valid[['hmid','moment','agency']]

train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)
unlabeled_data.reset_index(drop=True, inplace=True)

train.to_csv(path+'train_data.csv',index=False,sep=',')
valid.to_csv(path+'valid_data.csv',index=False,sep=',')
unlabeled_data.to_csv(path+'test_data.csv',index=False,sep=',')

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter,OrderedDict
from keras.utils.vis_utils import *
import random
import math
from cove import MTLSTM

from torchtext import vocab
# specify the path to the localy saved vectors
vec = vocab.Vectors('glove.840B.300d.txt','/home/ch13b009/ENV/cove/glove/')
# vec = vocab.GloVe(name='6B', dim=200)

import torchtext
import spacy
import re
from torchtext import data

# tokenizer function using spacy
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
def tokenizer(s): 
    return [w.text.lower() for w in nlp(tweet_clean(s))]

def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()

def tokenizer(s): 
    return [w.text.lower() for w in nlp(tweet_clean(s))]

# define the columns that we want to process and how to process
TEXT = data.Field(sequential=True, 
                 tokenize=tokenizer, 
                 include_lengths=True, 
                 use_vocab=True)
LABEL = data.Field(sequential=False, 
                   use_vocab=True,
                   pad_token=None, 
                   unk_token=None)

NUM = data.Field(sequential=False, 
                   use_vocab=False,
                   pad_token=None, 
                   unk_token=None)

train_val_fields = [
    ("hmid", NUM),
    ('moment', TEXT), # process it as text
    ('agency', LABEL) # process it as label
]


trainds, valds = data.TabularDataset.splits(path=path, 
                                            format='csv', 
                                            train='train_data.csv', 
                                            validation='valid_data.csv', 
                                            fields=train_val_fields, 
                                            skip_header=True)

tst_datafields = [("hmid", NUM), # we won't be needing the id, so we pass in None as the field
                  ('moment', TEXT)] # process it as text
tst = data.TabularDataset(
           path=path + "test_data.csv", # the file path
           format='csv',
           skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
           fields=tst_datafields)

# build the vocabulary using train and validation dataset and assign the vectors
TEXT.build_vocab(trainds,valds, max_size=100000,vectors=vec)
# build vocab for labels
LABEL.build_vocab(trainds)

outputs_last_layer_cove = MTLSTM(n_vocab=len(TEXT.vocab), vectors=TEXT.vocab.vectors)
outputs_both_layer_cove = MTLSTM(n_vocab=len(TEXT.vocab), vectors=TEXT.vocab.vectors, layer0=True)
outputs_both_layer_cove_with_glove = MTLSTM(n_vocab=len(TEXT.vocab), vectors=TEXT.vocab.vectors, layer0=True, residual_embeddings=True)


traindl, valdl = data.BucketIterator.splits(datasets=(trainds, valds), # specify train and validation Tabulardataset
                                            batch_sizes=(64,len(valid)),  # batch size of train and validation
                                            sort_key=lambda x: len(x.moment), # on what attribute the text should be sorted
                                            device=None, # -1 mean cpu and 0 or None mean gpu
                                            sort_within_batch=True, 
                                            repeat=False)
test_iter = data.Iterator(tst, batch_size=64, device=None, sort=False, sort_within_batch=False, repeat=False)

class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X,y)

train_batch_it = BatchGenerator(traindl, 'moment', 'agency')
valid_batch_it = BatchGenerator(valdl, 'moment', 'agency')
test_batch_it = BatchGenerator(test_iter,'moment','hmid')

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

embed_size = 1500
nh = 1024
drop=0.75
n_lstm_layers=1
model = LSTM(nh,n_lstm_layers,embed_size,drop)

from tqdm import tqdm
from cove import MTLSTM
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from torch.optim.lr_scheduler import StepLR
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_func = nn.BCELoss()
epochs = 20
batch_size = 64 
list_scores=[]
f=open(path+'self_train_results.txt',mode='w+')

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    iterations=0
    model.train() # turn on training mode
    for x,y in tqdm(train_batch_it,disable=True): # thanks to our wrapper, we can intuitively iterate over our data!
        iterations=iterations+1
        opt.zero_grad()
        (x,lengths)=x
        x=x.transpose(1,0)
        batch = (x,lengths)
        x = outputs_both_layer_cove_with_glove(*batch)
        x=x.transpose(1,0)
        predicted = model(x)
        predicted=predicted.view(-1)
        loss = loss_func(predicted,y.float())
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(trainds)
 
    # calculate the validation loss for this epoch
    val_loss = 0.0
    list_scores=[]
    model.eval() # turn on evaluation mode
    for x,y in tqdm(valid_batch_it,disable=True):
        (x,lengths)=x
        x=x.transpose(1,0)
        batch = (x,lengths)
        x = outputs_both_layer_cove_with_glove(*batch)
        x=x.transpose(1,0)
        preds = model(x)
        loss = loss_func(preds,y.float())
        val_loss += loss.item() * x.size(0)
        preds = [int(x>0.5) for x in preds]
        temp=accuracy_score(preds,y)
        auc = roc_auc_score(preds,y)
        f1=f1_score(preds,y,average = 'binary')
        list_scores.append(temp)
    val_loss /= len(valds)
    torch.save(model.state_dict(), 'eval'+str(epoch)+'.pt')
    f.write('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f},Validation AUC: {:.4f},Validation F1 score: {:.4f}\n'.format(epoch, epoch_loss, val_loss,temp,auc,f1))
print('Successfull')
f.close()    

ind= np.argmax(list_scores)
model.load_state_dict(torch.load(path + 'eval'+str(ind+1)+'.pt'))
model.eval()

test_ypreds = []
test_npreds = []
unsure_preds=[]
for x,y in tqdm(test_batch_it,disable=True):
    (x,lengths)=x
    x=x.transpose(1,0)
    batch = (x,lengths)
    x = outputs_both_layer_cove_with_glove(*batch)
    x=x.transpose(1,0)
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
valid.to_csv(path+"valid_data2.csv",index=False)