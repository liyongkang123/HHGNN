import argparse


def city_number(city):
    if city=='SP':
        user_number=3811
        poi_number=6255
        poi_class_number=289
        time_point_number=549
    elif city=="NYC":
        user_number=3754
        poi_number=3626
        poi_class_number=281
        time_point_number=547
    elif city=='JK':
        user_number=6184
        poi_number=8805
        poi_class_number=314
        time_point_number=566
    elif city == 'KL':
        user_number = 6324
        poi_number = 10804
        poi_class_number = 337
        time_point_number = 573
    return  user_number,poi_number,poi_class_number,time_point_number  #user - POI - POi_type -  timepoint

def parse():
    p = argparse.ArgumentParser("HHGNN:Heterogeneous Hypergraph Neural Network with SCL for Friend Recommendation in LBSN ", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--city',type=str, default='NYC',help=' the name of the city ' )
    p.add_argument('--model_name', type=str, default='HHGNN')
    p.add_argument('--input_dim',type=int,default=64,help='  the  Dimensions of initial features')
    p.add_argument('--nhid', type=int, default=256,help='number of hidden features, note that actually it\'s #nhid x #nhead')
    p.add_argument('--out_dim',type=int, default=384,help='number of output node  features, note that actually it\'s #nhid x #nhead' )
    p.add_argument('--nhead', type=int, default=3, help='number of conv heads in first layer ')
    p.add_argument('--out_nhead', type=int, default=3, help='number of conv heads in second layer')

    p.add_argument('--negative_K', type=int, default=4, help='negative_K')
    p.add_argument('--cuda', type=str, default='0', help='gpu id to use')
    p.add_argument('--multi_cuda', type=int, default=0, help='Specifies whether to use two GPU. The default value is 0. If two GPU are used, the value is set to  1')
    p.add_argument('--lam_1', type=int, default=1, help='lam_1')
    p.add_argument('--lam_2', type=int, default=2, help='lam_2')
    p.add_argument('--lam_3', type=int, default=3, help='lam_3')

    p.add_argument('--add-self-loop', default=True,action="store_true", help='add-self-loop to hypergraph')
    p.add_argument('--activation', type=str, default='relu', help='activation')
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    p.add_argument('--wd', type=float, default=1e-8, help='weight decay')
    p.add_argument('--epochs', type=int, default=7001, help='number of epochs to train')
    p.add_argument('--seed', type=int, default=1, help='seed for randomness')
    p.add_argument('--split', type=int, default=0.8,  help='choose which train/test split to use')

    args =p.parse_args()
    user_number, poi_number, poi_class_number, time_point_number=city_number(args.city)
    args.user_number=user_number
    args.poi_number=poi_number
    args.poi_class_number=poi_class_number
    args.time_point_number=time_point_number

    return args
