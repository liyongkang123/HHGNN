import torch, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
import torch_sparse
import pickle
import config
from  model import HHGNN_multi,HHGNN
import random
from torch_scatter import scatter

args = config.parse()
device = torch.device('cuda:'+args.cuda if torch.cuda.is_available() else 'cpu')


# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def accuracy(Z, Y):
    return 100 * Z.argmax(1).eq(Y).float().mean().item()

def fetch_data(args):
    city=args.city
    print('city name: ',city)
    read_friend=open(  city+"/friend_list_index.pkl",'rb' )
    friend_edge=pickle.load(read_friend)
    friend_edge_num=len(friend_edge)
    args.friend_edge_num=friend_edge_num
    print("the number of friendship hyperedge in raw dataset is:", friend_edge_num)
    print("the number of friendship hyperedge used for training is:", round(friend_edge_num *args.split))

    visit_poi=open(  city+"/visit_list_edge_tensor.pkl",'rb' )
    visit_edge=pickle.load(visit_poi)
    visit_edge_num=len(visit_edge)
    args.visit_edge_num=visit_edge_num
    print("the number of check-in hyperedge is:", visit_edge_num)

    tra=open( city+"/trajectory_list_index.pkl",'rb')
    trajectory_edge=pickle.load(tra)
    trajectory_edge_num=len(trajectory_edge)
    args.trajectory_edge_num=trajectory_edge_num
    print("the number of trajectory hyperedge is:", trajectory_edge_num)

    user_number=args.user_number
    poi_number=args.poi_number
    poi_class_number=args.poi_class_number
    time_point_number=args.time_point_number
    user_node_attr=torch.tensor(np.random.randint(0,10,(user_number,args.input_dim)) , dtype=torch.float32)
    poi_node_attr=torch.tensor(np.random.randint(0,10,(poi_number, args.input_dim)) , dtype=torch.float32)

    poi_class_attr=torch.zeros(poi_class_number,poi_class_number)
    index=range(0,poi_class_number,1)
    index=torch.LongTensor( index).view(-1,1)
    poi_class_attr=poi_class_attr.scatter_(dim=1,index=index,value=1)

    time_point_attr=torch.zeros(time_point_number,time_point_number)
    index2=range(0,time_point_number,1)
    index2 = torch.LongTensor(index2).view(-1, 1)
    time_point_attr=time_point_attr.scatter_(dim=1,index=index2,value=1)

    node_attr={}
    node_attr['user']=user_node_attr
    node_attr['poi']=poi_node_attr
    node_attr['poi_class']=poi_class_attr
    node_attr['time_point'] = time_point_attr


    train_rate=args.split
    friend_edge_train_len  =round(friend_edge_num *train_rate)
    all_index = list(np.arange(0, friend_edge_num, 1))
    train_edge_index = sorted(random.sample(all_index, friend_edge_train_len))
    test_edge_index_true = sorted(list(set(all_index).difference(set(train_edge_index))))

    friend_edge_train={}
    for i in range(len(train_edge_index)):
        friend_edge_train[i]=friend_edge[train_edge_index[i]]


    friend_edge_test=[]
    for i in range(len(test_edge_index_true)):
        friend_edge_test.append(list(friend_edge[ test_edge_index_true[i]]))

    for i in range(len(test_edge_index_true)):
        friend_edge_test.append([list(friend_edge[ test_edge_index_true[i]])[0], random.randint(0, user_number-1) ])

    test_label=[]
    for i in range(2* len(test_edge_index_true)):
        if i <len(test_edge_index_true):
            test_label.append(1)
        else:
            test_label.append(0)
    test_label=np.array(test_label)
    friend_edge_test=(torch.tensor(friend_edge_test,dtype=torch.long)).t().contiguous()
    K=args.negative_K
    friend_edge_train_all=[]
    for i in range(len(friend_edge_train)):
        friend_edge_train_all.append(list(friend_edge_train[i]))

    for i in range (len(friend_edge_train)):
        for j in range(K):
            friend_edge_train_all.append( [ list(friend_edge_train[i])[0],  random.randint(0, user_number-1) ])

    friend_edge_train_all =(torch.tensor(np.array(friend_edge_train_all),dtype=torch.long)).t().contiguous()

    friend_edge_train_all_label=[]
    for i in range(len(friend_edge_train)):
        friend_edge_train_all_label.append(1)
    for i in range (len(friend_edge_train)):
        for j in range(K):
            friend_edge_train_all_label.append(0)
    friend_edge_train_all_label=torch.tensor(np.array(friend_edge_train_all_label),dtype=torch.long)

    G={}
    G['friend']=friend_edge_train
    G['check_in'] = visit_edge
    G['trajectory'] = trajectory_edge
    return   G, node_attr,friend_edge_train, friend_edge_test, test_label,friend_edge_train_all, friend_edge_train_all_label,len(friend_edge_train)

def initialise( G,node_attr , args, node_type,edge_type, unseen=None):

    G2={}
    z=0
    for i in edge_type:
        for j in range(len(G[i])):
            G2[z]=G[i][j]
            z+=1

    G=G2.copy()
    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for e, vs in G.items():
            G[e] =  list(set(vs) - unseen)

    node_number= args.user_number+ args.poi_number+ args.poi_class_number+ args.time_point_number
    if args.add_self_loop:
        Vs = set(range(node_number))
        # only add self-loop to those are orginally un-self-looped
        # TODO:maybe we should remove some repeated self-loops?
        for edge, nodes in G.items():
            if len(nodes) == 1 and list(nodes)[0] in Vs:
                Vs.remove(list(nodes)[0])
        for v in Vs:
            G[f'self-loop-{v}'] = [v]
    args.self_loop_edge_number=len(G)-len(G2)
    edge_type.append('self-loop')
    N, M = node_number, len(G)
    indptr, indices, data = [0], [], []
    for e, vs in G.items():
        indices += vs
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()
    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col

    degE = scatter(degV[V], E, dim=0, reduce='sum')
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge

    V, E = V.to(device), E.to(device)

    args.edge_num=max(E)+1
    args.degV = degV.to(device)
    args.degE = degE.to(device)
    args.degE2 = degE2.pow(-1.).to(device)

    nhid = args.nhid
    nhead = args.nhead
    edge_input_length=[round(args.friend_edge_num*args.split),args.visit_edge_num,args.trajectory_edge_num,args.self_loop_edge_number]
    node_input_dim=[]
    for i in node_type:
        node_input_dim.append(node_attr[i].shape[1])
    node_input_length = [args.user_number,args.poi_number,args.poi_class_number,args.time_point_number]
    args.edge_type=edge_type
    args.node_type=node_type

    a=0
    edge_input_length_raw = []
    for i in range(len(edge_input_length)):
        edge_input_length_raw.append(a+edge_input_length[i])
        a=a + edge_input_length[i]

    b=0
    node_input_length_raw=[]
    for i in range(len(node_input_length)):
        node_input_length_raw.append(b+node_input_length[i])
        b= b+ node_input_length[i]

    V_class=[]
    for i in range(V.shape[0]):
        if V[i] <node_input_length_raw[0]:
            V_class.append(0) #user
        elif node_input_length_raw[0]<= V[i] <node_input_length_raw[1]:
            V_class.append(1)#POI
        elif node_input_length_raw[1]<= V[i] <node_input_length_raw[2]:
            V_class.append(2)#POItype
        elif node_input_length_raw[2]<= V[i] <node_input_length_raw[3]:
            V_class.append(3)#timepoint

    E_class=[]
    for i in range(E.shape[0]):
        if E[i]<edge_input_length_raw[0]:
            E_class.append(0) #friend
        elif edge_input_length_raw[0]<=E[i]<edge_input_length_raw[1]:
            E_class.append(1) #check-in
        elif edge_input_length_raw[1]<=E[i]<edge_input_length_raw[2]:
            E_class.append(2)#Trajectory
        elif edge_input_length_raw[2]<=E[i]<edge_input_length_raw[3]:
            E_class.append(3) #self-loop
    args.V_class=torch.tensor([V_class],dtype=torch.long).to(device)
    args.E_class=torch.tensor([E_class],dtype=torch.long).to(device)

    args.edge_input_length=edge_input_length_raw
    args.node_input_length=node_input_length_raw


    args.dataset_dict={'hypergraph':G,'n':N,'features':torch.randn(N,args.input_dim)}

    if args.multi_cuda==0:
        model = HHGNN(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim,edge_type,node_type)
        optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        model.to(device)
    elif args.multi_cuda==1:
        model = HHGNN_multi(args, args.input_dim, nhid, args.out_dim, nhead, V, E, node_input_dim,edge_type,node_type)
        optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    return model, optimiser, G



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)
