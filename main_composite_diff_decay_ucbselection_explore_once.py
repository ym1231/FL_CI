
# Composite feedback (Summation of single class ratio)
import os
from tqdm import tqdm

import random
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

# modules
from model import ToyCifarNet, ToyMLP
from dataSplit1 import get_data_loaders
from utils import AverageMeter
from utils_mab import *
# from sampling import uniform_sampling
# from LossRatio import get_auxiliary_data_loader, compute_ratio_per_client_update
from LossRatio_test import *
from scipy.stats import entropy

# GPU settings
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

NUM_CLASSES = 10
# Parameters that can be tuned during different simulations
num_clients = 16
num_selected = 4
device_set = np.arange(0, num_clients)
num_rounds = 3150	# num of communication rounds

epochs = 5			# num of epochs in local client training (An epoch means you go through the entire dataset in that client)
batch_size = 10 # batch size already set in train dataloaderin dataSplit.py
num_batch = 10 # change here choose 100 - 300 default:50

# hyperparameters of deep models
lr = 0.05 # learning rate
decay_factor = 0.996
decay_stop = 0.0001

decay_ratio = 0.8 # 0.8



losses_train = []
losses_test = []
acc_train = []
acc_test = []
client_idx_lst = []
T_pull_lst = []

d_num = epochs*batch_size*num_batch
data_num_g = d_num*num_selected

ratio1 = 0.01# 0.2,0.5,0.8
alpha = 0.3 # 0.1/3/5/7
weighted1 = True # True or False



dataset='cifar' # 'cifar' or 'mnist'
arch = 'small' # 'small' or 'big'



# cucb parameters
reward_est = np.zeros(num_clients)
reward_bias = np.zeros(num_clients)
T_pull = np.zeros(num_clients)
reward_global = np.zeros(num_rounds)
reward_client = np.zeros(num_clients)

# V_pt statistic
V_pt_avg = np.zeros((num_clients, NUM_CLASSES))

# m, sample times
num_m = 1
dim_H = int(2 * num_selected)
num_split = int(num_clients / (2*num_selected))

# generate hadamard matrix
h = hadamard(dim_H)
H_converted = convert_h(h, dim_H)

# num of sets
num_set = int(num_clients / dim_H)

# partition N into set of size 2k
device_set_split = np.zeros((num_set, dim_H))
for i in range(num_set):
    device_set_split[i, :] = device_set[(i*dim_H):((i+1)*dim_H)]

# b = np.zeros((num_clients, 1))
# A = np.zeros((num_clients, num_clients))
# reward_actual = np.zeros(num_rounds)
num_explore_round = num_split * num_m * dim_H * 2
V_avg = np.zeros((num_explore_round, NUM_CLASSES))
V_all = np.zeros((num_explore_round, NUM_CLASSES))
indicate_v = np.zeros((num_explore_round, num_clients))



def normalized(c):
    c_max = max(c)
    c_min = min(c)
    c_normalized = np.zeros(len(c))
    for i in range(len(c)):
        c_normalized[i] = (c[i] - c_min) / (c_max - c_min)
    return c_normalized


# FedAvg
def main():
    # stage index: a full exploration and exploitation stage called one stage
    s_idx = 0
    # to decide explore (1) or explot (0)
    explore_indicator = 0
    decay_ratio_sum = 0
	# data loader
	## client_train_loader: a list with #clients of **local** data loaders
	## test_loader: test.py only on **global** testset (local doesn't own seperate test.py data)
    
    client_train_loader, test_loader, data_size_per_client = get_data_loaders(num_clients, batch_size, True, ratio=ratio1, weighted=weighted1, dataset=dataset)
    data_size_weights = data_size_per_client / data_size_per_client.sum()
    
    aux_loader = get_auxiliary_data_loader(testset_extract=True, data_size=32, dataset=dataset)
    # model configurations
    # ASSUME that global model and client models are with exactly same structures
    # ASSUME that num_select is constant during FedAvg
    if dataset == 'cifar':
        if arch == 'small':
            global_model = ToyCifarNet(dataset, init_weights=True).to(device)
            client_models = [ToyCifarNet(dataset, init_weights=False).to(device) for _ in range(num_selected)]
        elif arch == 'big':
            global_model = MCifarNet(dataset, init_weights=True).to(device)
            client_models = [MCifarNet(dataset, init_weights=False).to(device) for _ in range(num_selected)]
        else:
            raise NotImplemented
    elif dataset == 'mnist':
        if arch == 'small':
            global_model = ToyCifarNet(dataset, init_weights=True).to(device)
            client_models = [ToyCifarNet(dataset, init_weights=False).to(device) for _ in range(num_selected)]
        elif arch == 'big':
            global_model = MCifarNet(dataset, init_weights=True).to(device)
            client_models = [MCifarNet(dataset, init_weights=False).to(device) for _ in range(num_selected)]
        else:
            raise NotImplemented
    elif dataset == 'svhn':
        global_model = ToyCifarNet().to(device)
        client_models = [ToyCifarNet(False).to(device) for _ in range(num_selected)]
    else:
        raise NotImplemented

    ## client models initialized by global model
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


	# client optimizers setting
	# ASSUME that client optimizer settings are all the same, and controlled by global server
	### XXXXXXXX ###
    ## [TODO] Here need to finetune to match the standard training settings common
    ##... in CifarNet training under datacenter SGD and FedAvg settings
    ##... consider later
    ### XXXXXXXX ###
    opt_lst = [optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4) for model in client_models]
    
    theta_log = np.zeros((num_clients, NUM_CLASSES))
    
    # for each communication round
    for r in range(num_rounds):

        # print(lr * (decay_factor ** r))
        
        # cal the global model class ratio at r - 1
        
        V_pt_global = compute_ratio_global_update(global_model, aux_loader)
        
        for opt in opt_lst:
            opt.param_groups[0]['lr'] = lr * (decay_factor ** r)
            if lr * (decay_factor ** r) <= decay_stop:
                opt.param_groups[0]['lr'] = decay_stop

        ## decide which stage: exploration or exploitation

        # obtain epoch index
        idx = r // num_explore_round

        # increase stage index at the start of next exploration
        if idx == np.power(2, s_idx + 1) + s_idx:
            s_idx += 1
        print("s_idx: ", s_idx)
        # if (np.power(2, s_idx) + s_idx - 1) <= idx and idx < (np.power(2, s_idx) + s_idx):
        #     explore_indi = 1
        # else:
        #     explore_indi = 0
        if idx < 1:
            explore_indi = 1
        else:
            explore_indi = 0
            


        # the start round of a new stage
        r_start = num_explore_round * (np.power(2, s_idx) + s_idx - 1)

        # obtain the reward with full bandit
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        if explore_indi:
            # exploration: explore the states of each client by Hadamod matrix
            # obtain client set
            client_idx = get_client_set(r, dim_H, H_converted, num_split, r_start, device_set_split)
            # print(client_idx, type(client_idx), int(client_idx[0]))
            print("Exploration")

        else:
            # exploitation: select the best subset with lowest class imbalance
            theta_hat = esti_theta_m(h, dim_H, num_clients, num_split, NUM_CLASSES, num_m, V_avg)
            theta_hat_normalized = np.zeros((num_clients, NUM_CLASSES))
            for i in range(num_clients):
                theta_hat_normalized[i, :] = normalized(theta_hat[i, :]) 
            print("Exploitation: theta_hat_normalized", theta_hat_normalized[0, :])
            reward = np.zeros(num_clients)
            for i in range(num_clients):
                reward[i] = 1 / cross_entropy(theta_hat_normalized[i, :])
            # obtain client set
            reward_bias = reward + alpha * np.sqrt(3 * np.log(r) / (2 * T_pull))
            client_idx = get_opt_set(reward_bias, num_selected, num_clients, theta_hat_normalized)
            if s_idx == 0:
                print('theta_hat_normalized: ', theta_hat_normalized)
                theta_log = theta_hat_normalized
            #exploit_index = r - (np.power(2, s_idx) + s_idx - 1) * num_explore_round
            #client_idx = get_opt_set_composite(reward, num_selected, num_clients, theta_hat_normalized, exploit_index)
            
        loss = 0

        for i in range(num_selected):
            loss += client_update(client_models[i], opt_lst[i], client_train_loader[client_idx[i]], epochs)

            
            
        decay_ratio_r = np.power(decay_ratio, s_idx)    
        # print("decay_ratio: ", decay_ratio, "decay_ratio_r: ", decay_ratio_r, "s_idx: ", s_idx)  
        decay_ratio_sum += decay_ratio_r
        
        
        # new: update pull time of each client
        print("round: ", r, "idx:", idx, "explore_indi: ", explore_indi, "client_idx:", client_idx)
        for s in client_idx:
            T_pull[s] += 1
            
            
        # get reward of global model
        reward_global[r] = 1/cross_entropy(compute_ratio_per_client_update([global_model], client_idx, aux_loader)[client_idx[0]])

        # loss needed to average across selected clients
        losses_train.append(loss / num_selected)

        # Step 4: Updated local models send back for server aggregate
        server_aggregate(global_model, client_models, data_size_weights, client_idx)
        # cal the global model class ratio at r
        V_pt_global_ = compute_ratio_global_update(global_model, aux_loader)
        # cal V_pt_diff
        V_pt_diff = V_pt_global_*data_num_g*r - V_pt_global*data_num_g*(r-1)
        # normalize V_pt_diff to [0,1]
        V_pt_diff_normalized = normalized(V_pt_diff)
        # decay normalize
        V_pt_diff_normalized_decay = V_pt_diff_normalized*decay_ratio_r
        print("V_pt_diff_normalized: ", V_pt_diff_normalized)
        # update V
        if explore_indi:
            V_all[r - r_start, :] = V_all[r - r_start, :] + V_pt_diff_normalized_decay
            V_avg[r - r_start, :] = V_all[r - r_start, :]/decay_ratio_sum
            # V_avg[r - r_start, :] = (V_avg[r - r_start, :]*s_idx + V_pt_diff_normalized) / (s_idx + 1)
            
        # Step 5: return loss and acc on testset
        test_loss, acc = test(global_model, test_loader)
        losses_test.append(test_loss)
        acc_test.append(acc)

        print('%d-th round' % r)
        print('average train loss %0.3g | test.py loss %0.3g | test.py acc: %0.3f' % (loss / num_selected, test_loss, acc))
        
        print("-----------------------------------------------")
        print("-----------------------------------------------")
        T_pull_lst.append(np.copy(T_pull))
        client_idx_lst.append(client_idx)
        name = "./mat/0623_log_composite_diff_bandit_decay_ucb_explore_once_lr"+str(lr)+"_decay"+str(decay_factor)+"_decaystop"+str(decay_stop)+"_decayratio"+str(decay_ratio)+"_alpha"+str(alpha)+"_C"+str(num_clients)+"_S"+str(num_selected)+"_Nbatch"+str(num_batch)+"_ratio"+str(ratio1)+"_weighted"+str(weighted1)+"_"+str(dataset)+".mat"

        sio.savemat(name, {'V_pt_avg': V_pt_avg, 'V_avg' : V_avg, 'V_pt_global' : V_pt_global, 'reward_client': reward_client, 'reward_global': reward_global, 'acc_test': acc_test, 'T_pull': T_pull_lst, 'client_idx': client_idx_lst, 'theta_log': theta_log})




# actually standard training in a local client/device
def client_update(client_model, optimizer, train_loader, epoch):
    loss_avg = AverageMeter()

    client_model.train()

    for e in range(epoch):
        # batch size from data loader is set to 32
        for batch_idx, (data, target) in enumerate(train_loader):

            if batch_idx == num_batch:
                break

            # transfer a mini-batch to GPU
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = client_model(data)
            # here use F.nll_loss() is wrong
            loss = F.cross_entropy(output, target)

            loss_avg.update(loss.item(), data.size(0))

            loss.backward()

            optimizer.step()



    return loss_avg.avg # average loss in this client over entire trainset over multiple epochs

def server_aggregate(global_model, client_models, data_size_weights, client_idx):
    global_dict = global_model.state_dict()

    non_sampled_client_idx = [i for i in list(range(num_clients)) if i not in list(client_idx)]

    for k in global_dict.keys():

        global_dict[k] = torch.stack([data_size_weights[client_idx[i]] * client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).sum(0) + sum([data_size_weights[idx] for idx in non_sampled_client_idx]) * global_model.state_dict()[k].float()


    global_model.load_state_dict(global_dict)



    # step of 'send aggregated global model to client' is here
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def test(global_model, test_loader):
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()

    global_model.eval()

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = global_model(data)

            loss = F.cross_entropy(output, target)
            loss_avg.update(loss.item(), data.size(0))


            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            acc_avg.update(pred.eq(target.view_as(pred)).sum().item(), data.size(0))

    return loss_avg.avg, acc_avg.avg



if __name__ == '__main__':
	main()
