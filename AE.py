from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from load_data import TabularDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import random
from sklearn.svm import OneClassSVM
from scipy import stats

random.seed(30)
np.random.seed(30)
"""
dataset = TabularDataset(data=Normalized_training_df,output_col='label')
testset = TabularDataset(data=Normalized_testing_df,output_col='label')

trainloader = DataLoader(dataset, batchsize, shuffle=True, num_workers=1)
trainloader_all = DataLoader(dataset, len(Normalized_training_df), shuffle=True, num_workers=1)
testloader = DataLoader(testset, len(Normalized_testing_df), shuffle=False, num_workers=1)
"""

batchsize = ????
dim = ???
latent_dim = int(np.round(1+np.sqrt(dim)))
first_layer = ???
second_layer = ???

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.fc1 = nn.Linear(dim, first_layer)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc12 = nn.Linear(first_layer, second_layer)
        nn.init.xavier_uniform_(self.fc12.weight)
        self.fc2 = nn.Linear(second_layer, latent_dim) #bottleneck
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(latent_dim, second_layer)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc32 = nn.Linear(second_layer, first_layer)
        nn.init.xavier_uniform_(self.fc32.weight)
        self.fc4 = nn.Linear(first_layer, dim)
        nn.init.xavier_uniform_(self.fc4.weight)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc12(h1))
        return self.fc2(h2)

    def decode(self, x):
        h3 = F.relu(self.fc3(x))
        h4 = F.relu(self.fc32(h3))
        return torch.sigmoid(self.fc4(h4))

    def forward(self, x):
        bottleneck = self.encode(x.view(-1, dim))
        return bottleneck, self.decode(F.relu(bottleneck))


torch.manual_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE().to(device)
torch.manual_seed(2)
optimizer = optim.Adadelta(model.parameters())
criterion = nn.MSELoss()

log_interval = 10
training_losses=[]
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (y, cont_x, cat_x) in enumerate(trainloader):
        data = cont_x
        data = data.to(device)
        optimizer.zero_grad()
        btlneck, recon_batch= model(data)
        loss = criterion(recon_batch,data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average train loss: {:.4f}'.format(
          epoch, train_loss / len(trainloader.dataset)))
    training_losses.append(train_loss / len(trainloader))


t0 = time.time()
for epoch in range(1, 200 + 1):
    train(epoch)

t1 = time.time()
print('training time: {} seconds'.format(t1 - t0))
torch.save(model.state_dict(), 'AE.pth')

model = AE()
#model.load_state_dict(torch.load('AE.pth'))

def select_data_online_ocsvm(ocsvm_df,ocsvm,df_for_online):
    support_vec = ocsvm_df.loc[ocsvm.support_]
    new_ocsvm_df = pd.concat([support_vec,df_for_online]).reset_index(drop=True)
    return new_ocsvm_df

def select_scores(tr_scores,per = 85):
    iqr = stats.iqr(tr_scores)
    lq = 0
    uq = np.percentile(tr_scores,per) + 0*iqr
    qr_arr = np.where((tr_scores>lq) & (tr_scores<uq))

    return tr_scores[qr_arr]

def ocsvm_online(nu_oc=0.01,minp=50,testing_arr=test_arr,off_scores=scores,obatch=1000,mwbatch=1000,thres=85):
    normal_scores = normal_data_dist
    ocsvm = OneClassSVM(kernel='rbf', nu=0.0001, gamma='auto')
    temp_df = Normalized_testing_df[:obatch].copy()
    indices = temp_df[temp_df['label'] == 0].index
    ae_min = np.percentile(off_scores[:obatch],minp)
    print('ae_min={}'.format(ae_min))
    ocsvm_df = (pd.DataFrame(testing_arr[indices])).reset_index(drop=True)
    ocsvm.fit(ocsvm_df)
    testing_df = pd.DataFrame(testing_arr[obatch:]).reset_index(drop=True)

    df_for_online = pd.DataFrame()
    ocsvm_scores = []
    scores_for_ae_min = []
    medval_list = []
    acp_rej=[]
    prediction = []
    batch_pred = []

    for i in testing_df.index:
        flow = testing_df.loc[i]
        ae_score = off_scores[i+obatch]
        scores_for_ae_min.append(ae_score)

        if ae_score < ae_min:
            df_for_online = df_for_online.append(flow).reset_index(drop=True)
        if ae_score < ae_min:
            lbl = 1
            ocsvm_sc = 0
        else:
            lbl = ocsvm.predict([flow])[0]
            ocsvm_sc = -ocsvm.score_samples([flow])[0]+1

        prediction.append(lbl)
        batch_pred.append(lbl)
        ocsvm_scores.append(ocsvm_sc)

        nbr_new_data = len(ocsvm.support_)+len(df_for_online)###
        if nbr_new_data>(obatch-1):
            ocsvm_df = select_data_online_ocsvm(ocsvm_df,ocsvm,df_for_online)
            ocsvm = OneClassSVM(kernel='rbf', nu=nu_oc, gamma='auto')
            ocsvm.fit(ocsvm_df)
            df_for_online = pd.DataFrame()

        if len(scores_for_ae_min)==mwbatch:
            batch_scores2 = Select_scores(np.array(scores_for_ae_min),per=thres)
            stat, pval = ???
            medval_list.append(np.percentile(scores_for_ae_min, minp))
            if pval>0.05:
                ae_min = np.percentile(scores_for_ae_min, 50)
                acp_rej.append('Accept')
            else:
                acp_rej.append('Reject')

            scores_for_ae_min = []
            batch_pred = []

    return ocsvm_scores
