# utils for composite MAB part
import numpy as np
# from sampling import *
from numpy.linalg import multi_dot
from scipy.stats import entropy
from scipy.linalg import hadamard


def cross_entropy(y):
    x = np.ones(y.shape)
    return entropy(x) + entropy(x, y + 1e-15)

def mse_m(a, b):
    l = a.shape
    err = 0
    for i in range(l[0]):
        for j in range(l[1]):
            err += np.power(a[i,j] - b[i,j], 2)

    err_avg = err / (l[0]*l[1])
    return err_avg

def uniform_sampling(num_clients, num_selected):
    return np.random.permutation(num_clients)[:num_selected]


def convert_h(H, dim_H):
    # convert H to matrix with indicator
    H_converted = np.zeros((2*dim_H, int(1/2*dim_H)))
    for i in range(dim_H):
        if i == 0:
            H_converted[2 * i, :] = uniform_sampling(dim_H, int(1 / 2 * dim_H))
            dim_set = np.arange(0, dim_H)
            for j in H_converted[2 * i, :]:
                del_ele = np.where(dim_set == j)[0][0]
                dim_set = np.delete(dim_set, del_ele)
            H_converted[2 * i + 1, :] = dim_set
        else:
            H_converted[2 * i, :] = np.where(H[i, :] == 1)[0]
            H_converted[2 * i + 1, :] = np.where(H[i, :] == -1)[0]
    return H_converted


def get_opt_set(reward, K, N, V_pt_avg):
    '''get client set for cucb'''
    # create dict
    V_pt_dict = {}
    for i in range(N):
        V_pt_dict[i] = V_pt_avg[i,:]
    # choose the max reward client as base
    r_max_index = np.argmax(reward)
    # remove the min index
    V_pt_dict.pop(r_max_index)
    # combination index set S
    S = np.array(r_max_index)
    # combination distribution set
    comb_set = np.array(V_pt_avg[r_max_index])

    while S.size < K:
        ce_reward_set = {}
        for key,value in V_pt_dict.items():
            # calculate the avg class distribution
            comb_dist = np.vstack([comb_set, V_pt_dict[key]])
            comb_dist_avg = np.sum(comb_dist, axis=0) / comb_dist.shape[0]
            # calculate cos loss of combined distribution
            ce_loss = cross_entropy(comb_dist_avg)
            # print(ce_loss)
            ce_reward_set[key] = 1 / ce_loss

        # get the cos ratio loss index
        reward_max_idx = max(V_pt_dict.keys(),key=(lambda x:ce_reward_set[x]))

        # remove the selected client

        S = np.append(S, reward_max_idx)
        comb_set = np.vstack([comb_set, V_pt_dict[reward_max_idx]])
        V_pt_dict.pop(reward_max_idx)

    return S

def get_opt_set_composite(reward, K, N, V_pt_avg, idx):
    '''get client set for cucb'''
    # create dict
    V_pt_dict = {}
    for i in range(N):
        V_pt_dict[i] = V_pt_avg[i,:]
    # choose the idx max reward client as base
    r_max_vec = np.argsort(idx)
    r_max_index = r_max_vec[-idx]
    # remove the min index
    V_pt_dict.pop(r_max_index)
    # combination index set S
    S = np.array(r_max_index)
    # combination distribution set
    comb_set = np.array(V_pt_avg[r_max_index])

    while S.size < K:
        ce_reward_set = {}
        for key,value in V_pt_dict.items():
            # calculate the avg class distribution
            comb_dist = np.vstack([comb_set, V_pt_dict[key]])
            comb_dist_avg = np.sum(comb_dist, axis=0) / comb_dist.shape[0]
            # calculate cos loss of combined distribution
            ce_loss = cross_entropy(comb_dist_avg)
            # print(ce_loss)
            ce_reward_set[key] = 1 / ce_loss

        # get the cos ratio loss index
        reward_max_idx = max(V_pt_dict.keys(),key=(lambda x:ce_reward_set[x]))

        # remove the selected client

        S = np.append(S, reward_max_idx)
        comb_set = np.vstack([comb_set, V_pt_dict[reward_max_idx]])
        V_pt_dict.pop(reward_max_idx)

    return S

def esti_theta_m(h, dim_H, num_clients, num_split, num_class, num_m, reward_sum):   # Z = np.zeros((1, dim_H))
    theta = np.zeros((num_clients, num_class))

    # avg reward over m
    reward_avg = np.zeros((num_split * dim_H * 2, num_class))

    for i in range(num_split * dim_H * num_m * 2):
        s = i % (num_split * dim_H * 2)
        # print("s: ", s)
        reward_avg[s] += reward_sum[s]

    for n in range(num_split):
        Z = np.zeros((dim_H, num_class))
        for r in range(n*dim_H*2, (n+1)*dim_H*2):
 #           stage =  r // (dim_H*2) # exploration stage, set 1 or 2, or others
 #           step = r % (dim_H*2) # exploration step, row of H

            idx = int(np.floor((r - n*dim_H*2)/2))
            # print("r: ", r, "n: ", n, "index: ", idx)
            if idx == 0:
                Z[idx, :] += reward_sum[r, :]
            else:
                if r % 2 == 0:
                    Z[idx, :] += reward_sum[r, :]
                else:
                    Z[idx, :] -= reward_sum[r, :]

        theta_temp = (1/dim_H)*np.dot(h.T, Z)
        theta[n*dim_H:(n+1)*dim_H, :] = theta_temp

    return theta

def get_client_set(round_total, dim_H, H_converted, num_split, r_start, device_set_split):
    # stage_idx = round_idx % (num_split * dim_H * 2)
    # exploration
    round_idx = round_total - r_start
    m_stage = (round_idx - r_start) // (num_split * dim_H * 2)  # idx of m
    m_step = (round_idx - r_start) % (num_split * dim_H * 2)  # idx of m

    stage =  (round_idx - r_start - num_split * m_stage * dim_H * 2) // (dim_H*2) # exploration stage, set 1 or 2, or others
    step = (round_idx - r_start - num_split * m_stage * dim_H * 2) % (dim_H*2) # exploration step, row of H

    choice = np.empty(int(1 / 2 * dim_H), dtype=int)
    for i in range(int(1 / 2 * dim_H)):
        choice[i] = device_set_split[stage, int(H_converted[step, i])]

    # print("round: ", r, "m_stage: ", m_stage, "m_step: ", m_step, "stage: ", stage, "step: ", step, "choice: ", choice)

    #for c in choice:
    #    indicate_v[round_idx, int(c)] = 1

    #indicator_m = np.asmatrix(indicate_v[round_idx, :])
    return choice