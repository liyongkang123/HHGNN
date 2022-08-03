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

args = config.parse()
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def test(Z,test_label,g,friend_edge_test):
    user_number = args.user_number

    user_emb=Z[:user_number]
    test_predict_label=F.cosine_similarity( user_emb[friend_edge_test[0]],user_emb[friend_edge_test[1]])
    test_predict_label=test_predict_label.detach().numpy()

    auc=sklearn.metrics.roc_auc_score(test_label,test_predict_label)
    ap = sklearn.metrics.average_precision_score(test_label, test_predict_label)
    print("ap:",ap)
    mea= np.mean(test_predict_label)
    acc=sklearn.metrics.accuracy_score(test_label,test_predict_label>mea)
    print('acc_test:',acc)
    user_emb_norm = torch.norm(user_emb, dim=-1, keepdim=True)
    dot_numerator = torch.mm(user_emb, user_emb.t())
    dot_denominator = torch.mm(user_emb_norm, user_emb_norm.t())
    sim = (dot_numerator / dot_denominator )
    src, dst = g.edges()
    src=list(src)
    dst=list(dst)
    friend_ture={}
    for i in range(len(src)):
        if src[i] in friend_ture.keys():
            friend_ture[src[i]]=friend_ture[src[i]]+[dst[i]]
        else:
            friend_ture[src[i]]=[dst[i]]

    for i in range(user_number):
        sim[i][i]=-1
        if i in friend_ture.keys():
            x=friend_ture[i]
            for j in x:
                sim[i][j]=-1
    test_pos_src, test_pos_dst = [int(i) for i in list(friend_edge_test[0])[:int(len(friend_edge_test[0])/2)]], [int(i) for i in  list(friend_edge_test[1])[:int(len(friend_edge_test[0])/2)]]
    friend_test_true={}
    for i in range(len(test_pos_src)):
        if test_pos_src[i] in friend_test_true.keys():
            friend_test_true[test_pos_src[i]]=friend_test_true[test_pos_src[i]]+[int(test_pos_dst[i])]
        else:
            friend_test_true[test_pos_src[i]]=[int(test_pos_dst[i])]

    for i in range(len(test_pos_dst)):
        if test_pos_dst[i] in friend_test_true.keys():
            friend_test_true[test_pos_dst[i]]=friend_test_true[test_pos_dst[i]]+[int(test_pos_src[i])]
        else:
            friend_test_true[test_pos_dst[i]]=[int(test_pos_src[i])]
    y_true=[]
    y_score=[]
    for i in friend_test_true.keys():
        y_true.append( friend_test_true[i])
        y_score.append(sim[i])

    k=[1,5,10,15,20] #top-k
    right_k=[0 for i in range(len(k))]
    for i in range(len(y_true)):
        sim_i = y_score[i].cpu().detach().numpy()
        for j in range(len(k)):
            s = sim_i.argsort()[-k[j]:][::-1]
            if set(list(s)) & set(y_true[i]):
                right_k[j]+=1
    for j in range(len(k)):
        print("Top ",k[j],'accuracy score is:', right_k[j]/len(y_true))
    return  auc

def margin_loss(pos_score, neg_score):
    # hinge loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def contrastive_loss(user_emb,g):
    user_emb=user_emb[:args.user_number]

    adj_friend=g.adj(scipy_fmt='coo',)
    adj_friend=adj_friend.todense()
    row,col=np.diag_indices_from(adj_friend)
    adj_friend[row,col]=1
    user_emb_norm=torch.norm(user_emb,dim=1,keepdim=True)  #
    dot_numerator = torch.mm(user_emb, user_emb.t())
    dot_denominator = torch.mm(user_emb_norm, user_emb_norm.t())
    sim = torch.exp(dot_numerator / dot_denominator / 0.2)
    matrix_mp2sc = sim/(torch.sum(sim, dim=1).view(-1, 1) + 1e-8)
    adj_friend=torch.tensor(adj_friend).to(device)
    lori_mp = -torch.log(matrix_mp2sc.mul(adj_friend).sum(dim=-1)).mean()
    return lori_mp
