import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
from model import SpADANN
import time
from sklearn.metrics import f1_score

def train_step(model, criterion, opt, train_dataloader, test_dataloader, epoch, num_epochs, device):
    start = time.time()
    model.train()
    alpha = float(epoch) / num_epochs
    tot_loss = []
    tot_loss_source = []
    tot_loss_target = []
    tot_loss_da = []
    tot_result = []
    for i, sample in  enumerate(train_dataloader):#
        x_train_s, y_train_s, x_train_t = sample
        opt.zero_grad()
        x_train_s = x_train_s.to(device)
        y_train_s = y_train_s.to(device)
        x_train_t = x_train_t.to(device)

        emb_s, pred_s, discr_s = model(x_train_s)
        emb_t, pred_t, discr_t = model(x_train_t)

        gt_da = torch.cat([torch.zeros(discr_s.shape[0]), torch.ones(discr_t.shape[0])], dim=0).long().to(device)
        
        loss_da = torch.mean(  criterion( torch.cat([discr_s,discr_t],dim=0), gt_da)   )
        loss_s = torch.mean( criterion(  pred_s , y_train_s))
        
        with torch.no_grad():
            source_pred = torch.argmax(pred_s, axis=1)
            target_pred = torch.argmax(pred_t, axis=1)
            first_cond = (source_pred == y_train_s)
            second_cond = (source_pred == target_pred)
            result = (first_cond & second_cond).type(torch.int)
        
        loss_combined = None
        loss_pseudo = None
        nnz = torch.count_nonzero(result)
        if nnz.item() == 0:
            loss_combined = (1 - alpha) * (loss_s + loss_da) #+ alpha * loss_pseudo
            loss_pseudo = torch.zeros(1)
        else:
            loss_pseudo = torch.sum(criterion(  pred_t , target_pred) * result) / torch.sum(result)
            loss_combined = (1 - alpha) * (loss_s + loss_da) + alpha * loss_pseudo

        loss_combined.backward()
        opt.step()

        tot_loss.append(loss_combined.item())
        tot_loss_source.append(loss_s.item())
        tot_loss_target.append(loss_pseudo.item())
        tot_loss_da.append(loss_da.item())
        tot_result.append(result.cpu().detach().numpy())
    
    end = time.time()
    tot_result = np.concatenate(tot_result, axis=0)
    model.eval()
    tot_f1 = []
    for x_batch, y_batch in test_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        _, pred, _ = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        f1 = f1_score(y_batch.cpu().detach().numpy(),pred_npy,average="weighted")
        tot_f1.append(f1)
    print("epoch %d with ALPHA %f | TOT LOSS %.3f | SOURCE L %.3f | PSEUDO L %.3f | DA L %.3f | F1 TARGET %.4f with TIME %d"%(epoch, alpha, np.mean(tot_loss),np.mean(tot_loss_source),np.mean(tot_loss_target),np.mean(tot_loss_da), np.mean(tot_f1)*100, (end-start) ))
    print("COND SATISFIED %f"%np.mean(tot_result))
    sys.stdout.flush()

def getData(year):
    data = np.load("data_%d.npy"%year)
    labels = np.load("gt_data_%d.npy"%year)
    labels = labels[:,2]-1
    return data, labels

source_year = int(sys.argv[1])
target_year = int(sys.argv[2])
source_x, source_y = getData(source_year)
target_x, target_y = getData(target_year)
n_classes = len( np.unique(source_y))

source_data = torch.tensor(source_x, dtype=torch.float32)
source_label = torch.tensor(source_y, dtype=torch.int64)
target_data = torch.tensor(target_x, dtype=torch.float32)
target_label = torch.tensor(target_y, dtype=torch.int64)


train_dataset = TensorDataset(source_data, source_label, target_data)
test_dataset = TensorDataset(target_data, target_label)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=256)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=512)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SpADANN(n_classes).to(device)
learning_rate = 0.0005
epochs = 500

loss_fn = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train_step(model, loss_fn, optimizer, train_dataloader, test_dataloader, epoch, epochs, device)