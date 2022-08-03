import torch
import torch.nn as nn, torch.nn.functional as F

import config
import math

from torch_scatter import scatter
from torch_geometric.utils import softmax
args = config.parse()
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')
device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

class hhgnnConv(nn.Module):
    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2, skip_sum=False,device=device):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=True)
        self.att_v_user=nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_poi = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_class = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_v_time = nn.Parameter(torch.Tensor(1, heads, out_channels))

        self.att_e_friend = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e_visit = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e_occurrence = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e_self = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_drop = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.skip_sum = skip_sum
        self.args = args
        self.edge_num=args.edge_num
        self.reset_parameters()
        self.edge_type=args.edge_type
        self.node_type=args.node_type
        self.edge_input_length = args.edge_input_length
        self.node_input_length = args.node_input_length

        self.V_class=(args.V_class).to(device)
        self.E_class=(args.E_class).to(device)

        self.relu = nn.ReLU()

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels, self.out_channels, self.heads)
    def reset_parameters(self):

        glorot(self.att_v_user)
        glorot(self.att_v_poi)
        glorot(self.att_v_class)
        glorot(self.att_v_time)
        glorot(self.att_e_friend)
        glorot(self.att_e_visit)
        glorot(self.att_e_occurrence)
        glorot(self.att_e_self)

    def xiangcheng(self,h,index_class,h_x):
        h_x=torch.cat(h_x,dim=0)
        index_class_2=(index_class.t()).repeat(1,self.heads,self.out_channels) .view(-1, self.heads, self.out_channels)
        h_x_2=torch.gather(h_x,0,index_class_2)
        output=(h * h_x_2).sum(-1)
        return output

    def forward(self, X, vertex, edges):
        H, C, N = self.heads, self.out_channels, X.shape[0]

        X0 = self.W(X)
        X = X0.view(N, H, C)
        Xve = X[vertex]

        X=Xve
        beta_v=self.xiangcheng(X, self.E_class, [self.att_e_friend,self.att_e_visit,self.att_e_occurrence,self.att_e_self])
        beta =self.leaky_relu(  beta_v)
        beta=softmax(beta,edges,num_nodes=self.edge_num)
        beta=beta.unsqueeze(-1)
        Xe=Xve*beta
        Xe=(scatter(Xe,edges,dim=0,reduce='sum',dim_size=self.edge_num))
        Xe2=Xe
        Xe=Xe2[edges]
        Xe_2=Xe
        alpha_e=self.xiangcheng(Xe_2,self.V_class,[self.att_v_user,self.att_v_poi,self.att_v_class ,self.att_v_time])
        alpha = self.leaky_relu(alpha_e)
        alpha = softmax(alpha, vertex, num_nodes = N)
        alpha = alpha.unsqueeze(-1)
        Xev = Xe * alpha
        Xv = scatter(Xev, vertex, dim=0, reduce='sum', dim_size=N)  # [N, H, C]
        X = Xv
        X = X.view(N, H * C)
        if self.skip_sum:
            X = X + X0
            # NOTE: concat heads or mean heads?
        # NOTE: skip concat here?
        return X

class HHGNN_multi(nn.Module):
    def __init__(self, args, nfeat, nhid, out_dim,  nhead, V, E,  node_input_dim,edge_type,node_type):
        super().__init__()
        self.conv_out = hhgnnConv(args, nhid * nhead, out_dim, heads=args.out_nhead,device=device2).to(device2)
        self.conv_in = hhgnnConv(args, nfeat, nhid, heads=nhead, device=device).to(device)
        self.V = V
        self.E = E
        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]
        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu=nn.ReLU().to(device2)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.lin_out1=nn.Linear( out_dim*args.out_nhead,out_dim,bias=True).to(device2)
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim]).to(device)
    def forward(self,  node_attr):
        node_feat={}
        for i in range(len(self.node_type)):
            node_feat[self.node_type[i]]=self.relu(self.fc_list_node[i](node_attr[self.node_type[i]].to(device)  ))
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat((X), 0).to(device)
        V, E = (self.V).to(device) , (self.E).to(device)
        X=self.conv_in(X, V, E)
        X = self.conv_out(X.to(device2), V.to(device2), E.to(device2))
        X = self.relu(X)
        X=self.lin_out1(X)
        return  X.to(device)

class HHGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, out_dim, nhead, V, E, node_input_dim,edge_type,node_type):

        super().__init__()
        self.conv_out = hhgnnConv(args, nhid * nhead, out_dim, heads=args.out_nhead,device=device)
        self.conv_in = hhgnnConv(args, nfeat, nhid, heads=nhead, device=device)
        self.V = V
        self.E = E
        act = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}
        self.act = act[args.activation]

        self.node_input_dim=node_input_dim
        self.edge_type=edge_type
        self.node_type=node_type
        self.relu=nn.ReLU().to(device2)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()
        self.lin_out1=nn.Linear( out_dim*args.out_nhead,out_dim,bias=True)
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, nfeat, bias=True)   for feats_dim in node_input_dim])

    def forward(self,  node_attr):
        node_feat={}
        for i in range(len(self.node_type)):
            node_feat[self.node_type[i]]=self.relu(self.fc_list_node[i](node_attr[self.node_type[i]]  ))
        X=[]
        for i in range(len(self.node_type)):
            X.append( node_feat[self.node_type[i]])
        X=torch.cat((X), 0)
        V, E = (self.V) , (self.E)
        X=self.conv_in(X, V, E)
        X = self.conv_out(X, V, E)
        X = self.relu(X)
        X=self.lin_out1(X)
        return  X
