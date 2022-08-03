import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os
from torch import cosine_similarity
import numpy as np
import time
import datetime
import path
import shutil
import sklearn.metrics
import config
from  prepare import fetch_data,accuracy,normalise,initialise
import dgl
from utils import *
args = config.parse()
# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'
model_name = args.model_name

node_type=['user','poi','poi_class','time_point']
edge_type=['friend','check_in','trajectory']

G, node_attr, friend_edge_train, friend_edge_test, test_label,friend_edge_train_all, friend_edge_train_all_label,k =fetch_data(args)
model, optimizer,G = initialise( G,node_attr ,args,node_type,edge_type)
friend_edge_train_all_label=(torch.tensor(friend_edge_train_all_label,dtype=torch.float32)).to(device)

friend_edge_train_list = []
for i in range(len(friend_edge_train)):
    f = list(friend_edge_train[i])
    friend_edge_train_list.append((f[0], f[1]))
    friend_edge_train_list.append((f[1], f[0]))

friend_edge_train_list = np.array(friend_edge_train_list)
friend_edge_train_list = torch.tensor(friend_edge_train_list, dtype=torch.long).t().contiguous()
g = dgl.graph((friend_edge_train_list[0], friend_edge_train_list[1]))


for i in node_type:
    node_attr[i]= torch.tensor(node_attr[i],).to(device)

best_test_auc, test_auc, Z = 0, 0, None
for epoch in range(args.epochs):

    tic_epoch = time.time()
    model.train()
    optimizer.zero_grad()
    Z = model( node_attr)
    predic_label=F.cosine_similarity(Z[friend_edge_train_all[0] ],Z[friend_edge_train_all[1] ])
    loss_cross = F.binary_cross_entropy_with_logits(predic_label, friend_edge_train_all_label)
    loss_margin= margin_loss(predic_label[: k], predic_label[k: ]  )
    con_loss=contrastive_loss(Z,g )
    loss= con_loss* args.lam_1+loss_cross* args.lam_2 + loss_margin* args.lam_3
    print("Epoch:",epoch,"LOSS:",loss)
    loss.backward()
    optimizer.step()

    if epoch>2000:
        if epoch%100==0:
            auc=test(Z.cpu(),test_label,g,friend_edge_test)
            print("Epoch:",epoch, "AUC:",auc)
            if auc>best_test_auc:
                best_test_auc=auc
                print("best_test_auc:",best_test_auc)
            print("best_test_auc:",best_test_auc)

