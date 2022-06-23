import argparse
import imp
# from utils import *
import math
import os
import os.path as osp
import pdb
import pickle
import random
import shutil
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import mmd
import Model
from dataloader import Modelnet40_data, Scannet_data_h5, Shapenet_data
from model_utils import hlr,Entropy,_ECELoss

#from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source',
                    '-s',
                    type=str,
                    help='source dataset',
                    default='scannet')
parser.add_argument('-target',
                    '-t',
                    type=str,
                    help='target dataset',
                    default='modelnet')
parser.add_argument('-batchsize',
                    '-b',
                    type=int,
                    help='batch size',
                    default=64)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='7')
parser.add_argument('-epochs',
                    '-e',
                    type=int,
                    help='training epoch',
                    default=5)

parser.add_argument('-models',
                    '-m',
                    type=str,
                    help='alignment model',
                    default='MDA')
parser.add_argument('-lr', type=float, help='learning rate',
                    default=1e-4)  #0.0001
parser.add_argument('-scaler',
                    type=float,
                    help='scaler of learning rate',
                    default=1.)
parser.add_argument('-weight',
                    type=float,
                    help='weight of src loss',
                    default=1.)
parser.add_argument('-datadir',
                    type=str,
                    help='directory of data',
                    default='./dataset/')
parser.add_argument('-tb_log_dir',
                    type=str,
                    help='directory of tb',
                    default='./logs')
parser.add_argument('--output_dir_src', type = str,default='models_with_bn_copy')
parser.add_argument('--output_dir', type=str,default= 'ckps/target_tent/1e-4')
# add n_components, subspace_option to the args
parser.add_argument('--n_components',type=int,default =300)
parser.add_argument('--subspace_option',type=str,default ='svd',choices=['pca','svd'])
parser.add_argument('--standardize',type=bool,default =True)
parser.add_argument('--reduced_data_percent',type = float, default=0.99)
parser.add_argument('--ent_par', type=float, default=0.2)
parser.add_argument('--sa_loss_par',type = float, default =1)
parser.add_argument('--gent', type=int, default=1)
args = parser.parse_args()

if args.gent:
    args.output_dir = 'ckps/target_tent_plus/1e-4'
if not os.path.exists(os.path.join(os.getcwd(), args.tb_log_dir)):
    os.makedirs(os.path.join(os.getcwd(), args.tb_log_dir))
#writer = SummaryWriter(log_dir=args.tb_log_dir)
args.output_dir=args.output_dir
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

BATCH_SIZE = args.batchsize * len(args.gpu.split(','))
LR = args.lr
weight_decay = 5e-4
momentum = 0.9
max_epoch = args.epochs
num_class = 10
dir_root = os.path.join(args.datadir, 'PointDA_data/')

def feat_load(args,name='source'):
    feat_dir = args.output_dir_src
    if name =='source':
        path = os.path.join(feat_dir, 'source_feats_labels.pkl')
    else:
        path = os.path.join(feat_dir, 'target_feats_labels.pkl')
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    return data
def get_subspace(feats,args,name='source'):
    feat_dir = args.output_dir_src
    subspace_file = os.path.join(feat_dir,name+'_subspace_'+str(args.n_components)+'.pkl')

    if os.path.exists(subspace_file):
        return torch.tensor(pickle.load(open(subspace_file,'rb'))).cuda()
    else:
        if args.standardize:
            feats = StandardScaler().fit_transform(feats)
        else:
            feats = feats
        if args.subspace_option=='pca':
            pca1 = PCA(n_components=args.n_components).fit(feats)
        elif args.subspace_option=='svd': 
            pca1 = TruncatedSVD(n_components=args.n_components).fit(feats)
        else:
            raise Exception('pca must be pca or svd')
        basis = pca1.components_.T
        with open(subspace_file,'wb') as f:
            pickle.dump(basis,f)
        
    return torch.tensor(basis).cuda()
# print(dir_root)
def call_acc_ece(dataloader,model_f,model_c,epoch,name='target_train'):
    with torch.no_grad():
        model_f.eval()
        model_c.eval()
        loss_total = 0
        correct_total = 0
        data_total = 0


        correct_total = 0
        data_total = 0

        all_outputs=[]
        all_labels=[]
        for batch_idx, (data, label,
                        _) in enumerate(dataloader):
            data = data.to(device=device)
            label = label.to(device=device).long()
            
            features_test = model_f(data)
           
            outputs = model_c(features_test)
            #output = (pred1 + pred2)/2
            #loss = criterion(outputs, label)
            _, pred = torch.max(outputs, 1)
            all_outputs.append(outputs)
            all_labels.append(label)
            #loss_total += loss.item() * data.size(0)
            correct_total += torch.sum(pred == label)
            data_total += data.size(0)
        all_outputs = torch.cat(all_outputs)
        all_labels= torch.cat(all_labels)
        #pred_loss = loss_total/data_total
        pred_acc_t = correct_total.double() / data_total
        ece = _ECELoss(15)(all_outputs, all_labels).cpu().data.item()
        logs= name+' {} Accuracy: {:.4f} ECE:{:.2f}'.format(epoch, pred_acc_t,ece)

        
        return_dict = {"acc": pred_acc_t, "ece":ece,"log_str":logs}


        return return_dict
def main():
    print('Start Training\nInitiliazing\n')
    print('src:', args.source)
    print('tar:', args.target)

    # Data loading
    
    data_func = {
        'modelnet': Modelnet40_data,
        'scannet': Scannet_data_h5,
        'shapenet': Shapenet_data
    }

    source_train_dataset = data_func[args.source](pc_input_num=1024,
                                                  status='train',
                                                  aug=True,
                                                  pc_root=dir_root +
                                                  args.source)
    target_train_dataset1 = data_func[args.target](pc_input_num=1024,
                                                   status='train',
                                                   aug=True,
                                                   pc_root=dir_root +
                                                   args.target)
    target_train_dataset_no_aug = data_func[args.target](pc_input_num=1024,
                                                   status='train',
                                                   aug=False,
                                                   pc_root=dir_root +
                                                   args.target)
    source_test_dataset = data_func[args.source](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.source)
    target_test_dataset1 = data_func[args.target](pc_input_num=1024, status='test', aug=False, pc_root= \
        dir_root + args.target)

    num_source_train = len(source_train_dataset)
    num_source_test = len(source_test_dataset)
    num_target_train1 = len(target_train_dataset1)
    num_target_test1 = len(target_test_dataset1)

    source_train_dataloader = DataLoader(source_train_dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=8,
                                         drop_last=True)
    source_test_dataloader = DataLoader(source_test_dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=8,
                                        drop_last=True)
    target_train_dataloader = DataLoader(target_train_dataset1,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=8,
                                         drop_last=True)
    target_train_dataloader_no_aug = DataLoader(target_train_dataset_no_aug,
                                         batch_size=BATCH_SIZE*3,
                                         shuffle=False,
                                         num_workers=8,
                                         drop_last=False)

    target_test_dataloader = DataLoader(target_test_dataset1,
                                        batch_size=BATCH_SIZE*3,
                                        shuffle=False,
                                        num_workers=8,
                                        drop_last=False)


    print(
        'num_source_train: {:d}, num_source_test: {:d}, num_target_test1: {:d} '
        .format(num_source_train, num_source_test, num_target_test1))
    print('batch_size:', BATCH_SIZE)

    # Model

    model_f = Model.FE()
    model_f = model_f.to(device=device)

    model_c = Model.Pointnet_c()
    model_c = model_c.to(device=device)

    modelpath = args.output_dir_src + '/F_src.pt'
    model_f.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/C_src.pt'
    model_c.load_state_dict(torch.load(modelpath))

    # load the source and target train_feats. 
    
   

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)

    remain_epoch = 50

    # Optimizer

    # params = [{
    #     'params': v
    # } for k, v in model_f.named_parameters() if 'pred_offset' not in k]
    param_group = []
    for k, v in model_f.named_parameters():
        
        if 'bn' in k:
            
            param_group += [{'params': v, 'lr': args.lr }]
        else:
            v.requires_grad = False
    
    optimizer_g = optim.Adam(param_group, lr=LR, weight_decay=weight_decay)
    lr_schedule_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g,
                                                         T_max=args.epochs +
                                                         remain_epoch)

    optimizer_c = optim.Adam([{
        'params': model_c.parameters()
    }],
                             lr=LR * 2,
                             weight_decay=weight_decay)
    lr_schedule_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c,
                                                         T_max=args.epochs +
                                                         remain_epoch)

   

    best_target_full_acc=0.0
    for epoch in range(max_epoch):

        model_f.train()
        model_c.train()

        since_e = time.time()

        lr_schedule_g.step(epoch=epoch)
        lr_schedule_c.step(epoch=epoch)

        loss_total = 0
        loss_adv_total = 0
        loss_node_total = 0
        correct_total = 0
        data_total = 0
        data_t_total = 0
        cons = math.sin((epoch + 1) / max_epoch * math.pi / 2)

        # Training
        for batch_idx, batch_s in enumerate(target_train_dataloader):

            data, label, tar_idx = batch_s

            data = data.to(device=device)
            
            inputs_target = data.cuda()

            features_test = model_f(inputs_target)

            outputs = model_c(features_test)
            softmax_out = nn.Softmax(dim=1)(outputs)
            
            
            entropy_loss = torch.mean(Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax +1e-5))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss = im_loss
            

            optimizer_g.zero_grad()
            #optimizer_c.zero_grad()
            classifier_loss.backward()
            optimizer_g.step()
            #optimizer_c.step()



        target_train_acc_ece_dict = call_acc_ece(target_train_dataloader_no_aug,model_f,model_c,epoch,name='target_train')
        print(target_train_acc_ece_dict['log_str'])
        target_test_acc_ece_dict = call_acc_ece(target_test_dataloader,model_f,model_c,epoch,name='target_test')
        print(target_test_acc_ece_dict['log_str'])

        full_target_acc = target_train_acc_ece_dict['acc']*num_target_train1+target_test_acc_ece_dict['acc']*num_target_test1
        full_target_acc = full_target_acc/(num_target_train1+num_target_test1)
        #writer.add_scalar('accs/target_test_acc', pred_acc_s, epoch)
        args.out_file.write(target_train_acc_ece_dict['log_str'] + '\n')
        args.out_file.write(target_test_acc_ece_dict['log_str'] + '\n')
        logs= 'full_target_acc'+' {} Accuracy: {:.4f} '.format(epoch, full_target_acc)
        
        args.out_file.write( logs+ '\n')

        args.out_file.flush()
        
        if full_target_acc > best_target_full_acc:
            best_target_full_acc= full_target_acc
            print('saving with best target full accuracy',best_target_full_acc )
            torch.save(model_f.state_dict(), osp.join(args.output_dir,'model_f_adapted.pt'))
            torch.save(model_c.state_dict(), osp.join(args.output_dir,'model_c_adapted.pt'))
        
        time_pass_e = time.time() - since_e
        print('The {} epoch takes {:.0f}m {:.0f}s'.format(
            epoch, time_pass_e // 60, time_pass_e % 60))
        print(args)
        


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '9'
    args.class_num = 10
    '''args.K=2
    args.KK=2'''

    SEED = 2021
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    since = time.time()

    #task = ['scannet', 'modelnet', 'shapenet']
    args.output_dir_src = osp.join(args.output_dir_src, args.source + '2' + args.target)
    args.output_dir = osp.join(args.output_dir, args.source + '2' + args.target)
    '''if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    task_s = args.source
    task.remove(task_s)
    task_all = [task_s + '2' + i for i in task]
    for task_sameS in task_all:
        path_task = os.getcwd() + '/' + 'model/' + task_sameS
        if not osp.exists(path_task):
            os.mkdir(path_task)'''
    args.output_dir = args.output_dir+'_lr_'+str(args.lr)+'_ent_loss_'+str(args.ent_par)
    os.makedirs(args.output_dir,exist_ok=True)
    args.out_file = open(osp.join(args.output_dir, 'tar.txt'), 'w')
    args.out_file.write('\n')
    args.out_file.flush()

    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_pass // 60, time_pass % 60))
