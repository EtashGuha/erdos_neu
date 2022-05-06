#!/usr/bin/env python
# coding: utf-8

# ## First take care of all imports

# In[1]:


import os
import pickle
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import time
from torch import tensor
from torch.optim import Adam
from torch.optim import SGD
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from math import ceil
from torch.nn import Linear
from torch.distributions import categorical
from torch.distributions import Bernoulli
import torch.nn
import copy
from matplotlib import pyplot as plt
# import pygraphviz as pgv
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
# import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
from torch.utils.data.sampler import RandomSampler
from torch.nn.functional import gumbel_softmax
from torch.distributions import relaxed_categorical
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.data import Batch 
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
from torch import autograd
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
from models import cut_MPNN
from modules_and_utils import derandomize_cut, GATAConv,get_diracs, total_var
import scipy
import scipy.io
import GPUtil
import numpy as np


# ## Prepare dataset

# In[2]:


# ## TRAIN MODEL

# In[3]:

def get_params(model):
    all_weights = []
    for param in model.parameters():
        all_weights.append(copy.deepcopy(param.data))
    return all_weights

def get_params_weights(model):
    all_weights = []
    for param in model.parameters():
        all_weights.append(torch.linalg.vector_norm(copy.deepcopy(param.data).to("cpu")))
    return all_weights

def calc_distance_params(a_params, b_params):
    all_distances = []
    for i in range(len(a_params)):
        a_param = a_params[i]
        b_param = b_params[i]
        all_distances.append(torch.linalg.vector_norm(a_param - b_param.to("cpu"))/torch.linalg.vector_norm(a_param))
    return all_distances

def predict(model, data_loader, recfield):
    net.eval()
    avg_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        data = data.to(device)
        data = get_diracs(data, 1, sparse = True, effective_volume_range=0.15, receptive_field = recfield)
        data = data.to(device)
        retdict = net(data)
        avg_loss += retdict['loss'][0].item()/len(data_loader)

    return avg_loss


# In[4]:
def train_model(taus, seed):
    datasets = ["facebook", "sf","twitter", 'facebook_ct1']
    curr_dataset= datasets[2] 
    #set random seed
    rseed = 201

    if curr_dataset=="facebook":
        datasetname = "facebook_graphs"
        dataset = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #set up facebook data 
        upper_limit = 15000
        lower_limit = 0
        path_to_facebook_dataset= "/nethome/eguha3/erdos_neu/datasets/facebook100/"
        for file in os.listdir(path_to_facebook_dataset):
                if file.endswith(".mat"):
                    print("hi")
                    adj_matrix = scipy.io.loadmat(path_to_facebook_dataset+str(file))
                    edge_index = from_scipy_sparse_matrix(adj_matrix['A'])[0]
                    x = torch.ones(adj_matrix['local_info'].shape[0])
                    if (adj_matrix['local_info'].shape[0] < lower_limit) or (adj_matrix['local_info'].shape[0] > upper_limit):
                        continue
                    data_temp = Batch(x = x, edge_index = edge_index.long(), batch = torch.zeros_like(x).long())
                    data_proper = get_diracs(data_temp.to('cuda'), 1, sparse = True)
                    r,c = data_proper.edge_index
                    data = Batch(x = data_temp.x, edge_index = data_temp.edge_index)
                    degrees = degree(r, adj_matrix['local_info'].shape[0])
                    print("Graph specs: ")
                    print("number of nodes: ", adj_matrix['A'].shape[0])
                    print("average degree: ", degrees.mean(0))
                    print("total volume: ", data_proper.total_vol)
                    print("-------------")
                    dataset += [data]
    elif curr_dataset=="sf":
        datasetname = "SF-295"
        dataset = TUDataset(root='/nethome/eguha3/erdos_neu/datasets/SF-295'+datasetname, name=datasetname)

    elif curr_dataset=="twitter":
        path_to_twitter_dataset = "/nethome/eguha3/erdos_neu/datasets/TWITTER_SNAP_2.p"
        stored_dataset = open(path_to_twitter_dataset, 'rb')        
        dataset = pickle.load(stored_dataset)   
        dataset = [Data.from_dict(data) for data in dataset] 

    elif curr_dataset=="facebook_ct1":
        datasetname = "facebook_ct2"
        dataset = TUDataset(root='/nethome/eguha3/erdos_neu/datasets/facebook_ct2'+datasetname, name=datasetname)


    dataset_scale = 0.1
    total_samples = int(np.floor(len(dataset)*dataset_scale))
    dataset = dataset[:total_samples]

    num_trainpoints = int(np.floor(0.6*len(dataset)))
    num_valpoints = int(np.floor(num_trainpoints/3))
    num_testpoints = len(dataset) - (num_trainpoints + num_valpoints)


    traindata= dataset[0:num_trainpoints]
    valdata = dataset[num_trainpoints:num_trainpoints + num_valpoints]

    testdata = dataset[num_trainpoints + num_valpoints:]

    batch_size = 1
    train_loader = DataLoader(traindata, batch_size, shuffle=True)
    test_loader = DataLoader(testdata, batch_size, shuffle=False)
    val_loader =  DataLoader(valdata, batch_size, shuffle=False)

    #set up random seed 
    # torch.manual_seed(rseed)
    np.random.seed(2)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    epochs=150
    numlayers=6
    elasticity = 0.25
    receptive_field= numlayers + 1
    val_losses = []

    #for sf/twitter
    #net =  cut_MPNN(dataset,numlayers, 64, 64,1, elasticity = elasticity)

    #for faceboook
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net =  cut_MPNN(dataset,6, 256, 24,1, elasticity = 0.25)
    initial_params = get_params(net)




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_decay_step_size = 5
    lr_decay_factor = 0.95

    net.to(device).reset_parameters()
    optimizer = Adam(net.parameters(), lr=0.0001, weight_decay=0.00)
    net.train()
    retdict = {}

    for epoch in range(epochs):
        final_losses = []
        print("Current epoch: ", epoch)
        totalretdict = {}
        count=0
        #learning rate schedule
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_decay_factor * param_group['lr']
    
        net.train()
        for data in train_loader:
            count += 1 
            optimizer.zero_grad(), 
            data = data.to(device)
            data_prime = get_diracs(data, 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
            data = data.to('cpu')
            data_prime = data_prime.to(device)  
            retdict = net(data_prime, tau=taus[epoch])
            for key,val in retdict.items():
                if "sequence" in val[1]:
                    if key in totalretdict:
                        totalretdict[key][0] += val[0].item()
                    else:
                        totalretdict[key] = [val[0].item(),val[1]]
            
            if epoch > 0:
                    final_losses.append(copy.deepcopy(retdict["loss"][0].item()))
                    retdict["loss"][0].backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),1)
                    optimizer.step()                   

            del data_prime


    # ## EVALUATE

    # In[ ]:


    #initialize
    net.eval()
    rcuts =  {}
    rvols =  {}
    rconds = {}
    mcuts =  []
    mvols =  []
    mconds = []
    best_cuts =  []
    best_vols =  []
    best_conds = []
    count = 0

    #select number of diracs
    num_diracs = 10

    cuts = torch.zeros((num_testpoints, num_diracs))
    conds =  torch.zeros((num_testpoints, num_diracs))
    vols =  torch.zeros((num_testpoints, num_diracs))
    randtargets = torch.zeros((num_testpoints, num_diracs))
    best_sets = {}
    totalvols = []



    t_0 = time.time()
    with torch.no_grad():
        for data2 in test_loader:
            batch = data2.batch
            
            count += 1
            
            print("Batch count: ", count)
            dirac_count = 0 
            for dirac in range(num_diracs):
                data2 = data2.to(device)            
                data_new = get_diracs(data2, 1, sparse=True, effective_volume_range=0.2)            
                
                feasible_vols = (data_new.recfield_vol/data_new.total_vol)*0.85
                target_vol = torch.rand_like(feasible_vols, device=device)*feasible_vols + 0.1
                data_new = data_new.to(device) 
                retdict2 = net(data_new, target_vol)
                netprobs = retdict2['output'][0]
                batch_new = data_new.batch
                num_graphs = batch_new.max().item() + 1
                e_i = data_new.edge_index 
                r,c = e_i
                deg = degree(r)
                bestcond = torch.ones(num_graphs)
                bestcut = 1000*torch.ones(num_graphs)
                bestvol = torch.zeros(num_graphs)
                outp =  derandomize_cut(data_new.to('cuda'), netprobs.cuda(), target_vol.cuda()*data_new.total_vol.cuda(), elasticity=0.25, draw=False)
                tv_hard = total_var(outp, data_new.edge_index.cuda(), data_new.batch.cuda())
                vol_hard = scatter_add(deg*outp, batch_new, 0, dim_size = batch_new.max().item()+1)
                mycond = tv_hard/vol_hard
                conds[(count-1)*batch_size:count*batch_size, dirac] = mycond
                cuts[(count-1)*batch_size:count*batch_size, dirac] = tv_hard
                vols[(count-1)*batch_size:count*batch_size, dirac] = vol_hard
                randtargets[(count-1)*batch_size:count*batch_size, dirac] = target_vol.cpu()*data_new.total_vol.cpu()          
                dirac_count += 1
    t_final = time.time() - t_0
    print(f"average time per graph: {t_final/len(testdata)}")   


    ## Print out mean conductance +/- std

    final_params = get_params(net)
    meanconds = conds.mean(0)
    print(f"meanconds: {meanconds.mean()} +/- {meanconds.std()}")
    return meanconds.mean(), meanconds.std(), net, initial_params, final_params, calc_distance_params(initial_params, final_params), final_losses

# In[ ]:
if __name__ == "__main__": 
    all_params = []

    list_of_taus = []
    num_epochs = 200
    name = "mincut_actually_linear_ablation"
    final_solution = []
    if os.path.exists("{}.pkl".format(name)):
        with open("{}.pkl".format(name), "rb") as f:
            final_solution = pickle.load(f)
    # initial_taus_betas = [(50000, 1e-4), (50000, 1e-5), (50000, 1e-6), (50000, 1e-1), (50000, 1e-2), (50000, 1e-3)]
    initial_taus_betas = [(10000, -1), (20000, -1), (30000, -1), (40000, -1), (50000, -1), (60000, -1)]
    all_best_mus = []
    all_best_distances = []
    for initial_tau, beta in initial_taus_betas:
        best_mu = 100
        best_net = None
        taus = [initial_tau]
        for epoch in range(500):
            taus.append(initial_tau * (150 - epoch)/(epoch + 1))
        taus.extend([0] * 5)
        mus = []
        stds = []
        distance_measures = []
        seeds = list(range(8))
        for seed in seeds:
            mu_sam, std_sam,  net, initial_params, final_params,  distances, final_losses = train_model(taus, seed)
            mus.append(mu_sam)
            stds.append(std_sam)
            distance_measures.append(np.mean(distances))
            if mu_sam < best_mu:
                best_mu = mu_sam
                best_net = net
            final_solution.append((initial_tau, beta, seed, mu_sam, std_sam,  net, initial_params, final_params,  distances, final_losses))
        mu = np.mean(mus)
        std = np.mean(stds)
        torch.save(net, "models/{}_{}it_{}bt_is.pt".format(name, initial_tau, beta))
        with open("{}.pkl".format(name), "wb") as f:
            pickle.dump(final_solution, f)
        all_best_mus.append(best_mu)
        all_best_distances.append(np.mean(distance_measures))
    print(all_best_mus)
    print(all_best_distances)


# In[ ]:




