import torch
import torch.nn.functional as F
from torch.nn import Linear
from itertools import product
import time
import argparse
from tqdm import tqdm
import os
import math
from torch import tensor
from torch.optim import Adam
from torch.optim import SGD
from math import ceil
from torch.nn import Linear
from torch.distributions import categorical
from torch.distributions import Bernoulli
import torch.nn
# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
from torch.utils.data.sampler import RandomSampler
from torch.nn.functional import gumbel_softmax
from torch.distributions import relaxed_categorical
import myfuncs
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, NNConv, GCNConv
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
import argparse
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Batch 
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
from torch import autograd
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
from torch_geometric.utils import dropout_adj, to_undirected, to_networkx
from torch_geometric.utils import is_undirected
from cut_utils import get_diracs
import scipy
import scipy.io
from matplotlib.lines import Line2D
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import GPUtil
from networkx.algorithms.approximation import max_clique
import pickle
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel
from torch_geometric.data import DataListLoader, DataLoader
from random import shuffle
from networkx.algorithms.approximation import max_clique
from networkx.algorithms import graph_clique_number
from networkx.algorithms import find_cliques
from torch_geometric.nn.norm import graph_size_norm
from torch_geometric.datasets import TUDataset
import visdom 
from visdom import Visdom 
import numpy as np
import matplotlib.pyplot as plt
from  cut_utils import solve_gurobi_maxclique
import gurobipy as gp
from gurobipy import GRB
from models import clique_MPNN
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
from modules_and_utils import decode_clique_final, decode_clique_final_speed
from torch_geometric.utils.convert import from_networkx
import pickle
import copy
from hanging_threads import start_monitoring
# start_monitoring(seconds_frozen=10, test_interval=100)
# ## Set up data

def get_params(model):
    all_weights = []
    for param in model.parameters():
        all_weights.append(copy.deepcopy(param.data))
    return all_weights

def get_params_weights(model):
    all_weights = []
    for param in model.parameters():
        all_weights.append(torch.linalg.vector_norm(copy.deepcopy(param.data)))
    return all_weights

def calc_distance_params(a_params, b_params):
    all_distances = []
    for i in range(len(a_params)):
        a_param = a_params[i]
        b_param = b_params[i]
        all_distances.append(torch.linalg.vector_norm(a_param - b_param)/torch.linalg.vector_norm(a_param))
    return all_distances

# In[10]:

def run_training(dataset, taus, seed, beta, device):
        
    dataset_scale = 1


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


    #set up random seeds 
    # torch.manual_seed(1)
    # np.random.seed(2)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # In[14]:


    #number of propagation layers
    numlayers = 5

    #size of receptive field
    receptive_field = numlayers + 1

    # val_losses = []
    # cliq_dists = []
    np.random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net =  clique_MPNN(dataset,numlayers, 32, 32,1)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    lr_decay_step_size = 5
    lr_decay_factor = 0.95

    net.to(device).reset_parameters()
    optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.00)


    # ## Train model

    # In[15]:


    #Temporarily here
    # epochs = 200
    # net.train()
    # retdict = {}
    # edge_drop_p = 0.0
    # edge_dropout_decay = 0.90
    # penalty_coeff = 9.00
    # penalty_increase = -0.00
    # validation_timeout = 500

    # b_sizes = [32]
    # l_rates = [0.001]
    # depths = [3]
    # coefficients = [3.5]
    # rand_seeds = [68]
    # widths = [64]

    #THEBEST
    # epochs = 100
    # net.train()
    # retdict = {}
    # edge_drop_p = 0.0
    # edge_dropout_decay = 0.90
    # penalty_coeff = 9.00
    # penalty_increase = -0.00
    # validation_timeout = 75



    b_sizes = [1]
    l_rates = [0.001]
    depths = [4]
    coefficients = [beta]
    rand_seeds = [66]
    widths = [64]

    epochs = 150
    net.train()
    retdict = {}
    edge_drop_p = 0.0
    edge_dropout_decay = 0.90



    for batch_size, learning_rate, numlayers, penalty_coeff, r_seed, hidden_1 in product(b_sizes, l_rates, depths, coefficients, rand_seeds, widths):
    
        torch.manual_seed(r_seed)


        train_loader = DataLoader(traindata, batch_size, shuffle=True)
        test_loader = DataLoader(testdata, batch_size, shuffle=False)
        val_loader =  DataLoader(valdata, batch_size, shuffle=False)

        receptive_field= numlayers + 1
        val_losses = []
        cliq_dists = []

        #hidden_1 = 128
        hidden_2 = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        net =  clique_MPNN(dataset,numlayers, hidden_1, hidden_2 ,1)
        net.to(device).reset_parameters()

        initial_params = get_params(net)
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00000)

        losses = []
        epochs_list = []
        for epoch in tqdm(range(epochs)):
            totalretdict = {}
            count=0
            final_losses = []
            if epoch % 5 == 0:
                edge_drop_p = edge_drop_p*edge_dropout_decay
                # print("Edge_dropout: ", edge_drop_p)

            if epoch % 10 == 0:
                penalty_coeff = penalty_coeff + 0.
                # print("Penalty_coefficient: ", penalty_coeff)

            #learning rate schedule
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_decay_factor * param_group['lr']

            #show currrent epoch and GPU utilizations9s
            # print('Epoch: ', epoch)
            # GPUtil.showUtilization()



            #print("here3")
            net.train()
            total_loss = 0
            for data in train_loader:
                count += 1 
                optimizer.zero_grad(), 
                data = data.to(device)
                data_prime = get_diracs(data, 1, device_name=device, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
                
                data = data.to('cpu')
                data_prime = data_prime.to(device)

                retdict = net(data_prime, None, penalty_coeff, tau=taus[epoch])
                
                for key,val in retdict.items():
                    if "sequence" in val[1]:
                        if key in totalretdict:
                            totalretdict[key][0] += val[0].item()
                        else:
                            totalretdict[key] = [val[0].item(),val[1]]

                if epoch > 2:
                        final_losses.append(retdict["loss"][0].item())
                        total_loss += retdict["loss"][0].item()
                        retdict["loss"][0].backward()
                        #reporter.report()
                        
                        torch.nn.utils.clip_grad_norm_(net.parameters(),1)
                        optimizer.step()
                        del(retdict)

            if epoch > -1:        
                for key,val in totalretdict.items():
                    if "sequence" in val[1]:
                        val[0] = val[0]/(len(train_loader.dataset)/batch_size)
                
                del data_prime
            losses.append(total_loss/len(train_loader))
    # ## Get ground truths from Gurobi

    # In[16]:

    test_data_clique = []

    for data in testdata:
        my_graph = to_networkx(Data(x=data.x, edge_index = data.edge_index)).to_undirected()
        print(my_graph)
        if "clique_number" not in data.keys():
            cliqno, _ = solve_gurobi_maxclique(my_graph, 500)
            data.clique_number = cliqno
        test_data_clique += [data]


    # ## Evaluate on test set

    # In[18]:


    tbatch_size = batch_size
    num_data_points = num_testpoints

    batch_size = 1
    test_data = testdata
    test_loader = DataLoader(test_data, batch_size, shuffle=False)
    net.to(device)
    count = 1

    final_params = get_params(net)
    #Evaluation on test set
    net.eval()

    gnn_nodes = []
    gnn_edges = []
    gnn_sets = {}

    #set number of samples according to your execution time, for 10 samples
    max_samples = 8

    gnn_times = []
    num_samples = max_samples
    t_start = time.time()

    for data in test_loader:
        num_graphs = data.batch.max().item()+1
        bestset = {}
        bestedges = np.zeros((num_graphs))
        maxset = np.zeros((num_graphs))

        total_samples = []
        for graph in range(num_graphs):
            curr_inds = (data.batch==graph)
            g_size = curr_inds.sum().item()
            if max_samples <= g_size: 
                samples = np.random.choice(curr_inds.sum().item(),max_samples, replace=False)
            else:
                samples = np.random.choice(curr_inds.sum().item(),max_samples, replace=True)

            total_samples +=[samples]

        data = data.to(device)
        t_0 = time.time()

        for k in range(num_samples):
            t_datanet_0 = time.time()
            data_prime = get_diracs(data.to(device), 1, device_name=device, sparse = True, effective_volume_range=0.15, receptive_field = 7)
    
            initial_values = data_prime.x.detach()
            data_prime.x = torch.zeros_like(data_prime.x)
            g_offset = 0
            for graph in range(num_graphs):
                curr_inds = (data_prime.batch==graph)
                g_size = curr_inds.sum().item()
                graph_x = data_prime.x[curr_inds]
                data_prime.x[total_samples[graph][k] + g_offset]=1.
                g_offset += g_size
                
            retdz = net(data_prime)
            
            t_datanet_1 = time.time() - t_datanet_0
            print("data prep and fp: ", t_datanet_1)
            t_derand_0 = time.time()

            sets, set_edges, set_cardinality = decode_clique_final_speed(data_prime,(retdz["output"][0]), weight_factor =0.,draw=False, beam = 1)

            t_derand_1 = time.time() - t_derand_0
            print("Derandomization time: ", t_derand_1)

            for j in range(num_graphs):
                indices = (data.batch == j)
                if (set_cardinality[j]>maxset[j]):
                        maxset[j] = set_cardinality[j].item()
                        bestset[str(j)] = sets[indices].cpu()
                        bestedges[j] = set_edges[j].item()

        t_1 = time.time()-t_0
        print("Current batch: ", count)
        print("Time so far: ", time.time()-t_0)
        gnn_sets[str(count)] = bestset
        
        gnn_nodes += [maxset]
        gnn_edges += [bestedges]
        gnn_times += [t_1]

        count += 1

    t_1 = time.time()
    total_time = t_1 - t_start
    print("Average time per graph: ", total_time/(len(test_data)))


    # In[20]:


    #flatten output
    flat_list = [item for sublist in gnn_edges for item in sublist]
    for k in range(len(flat_list)):
        flat_list[k] = flat_list[k].item()
    gnn_edges = (flat_list)

    flat_list = [item for sublist in gnn_nodes for item in sublist]
    for k in range(len(flat_list)):
        flat_list[k] = flat_list[k].item()
    gnn_nodes = (flat_list)


    # In[21]:


    tests = test_data_clique
    ratios = [gnn_nodes[i]/tests[i].clique_number for i in range(len(tests))]
    print(f"Mean ratio: {(np.array(ratios)).mean()} +/-  {(np.array(ratios)).std()}")
    return (np.array(ratios)).mean(), (np.array(ratios)).std(), losses, net, initial_params, final_params, calc_distance_params(initial_params, final_params), final_losses

def get_data(problem):

    datasets = ["TWITTER_SNAP", "COLLAB", "IMDB-BINARY"]
    # dataset_name = datasets[0]
    #datasetname = "COLLAB_shuffle_1"
    #datasetname = "TWITTER_SNAP"
    dataset_name = "RB-MODEL"

    #TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

    if dataset_name == "TWITTER_SNAP":
        stored_dataset = open('datasets/TWITTER_SNAP.p', 'rb')
        dataset = pickle.load(stored_dataset)
    elif dataset_name == "COLLAB":
        dataset = TUDataset(root='/tmp/'+dataset_name, name=dataset_name)
        #stored_dataset = open('datasets/dataset_shuffle_1.p', 'rb')
    elif dataset_name == "IMDB-BINARY":
        #stored_dataset = open('datasets/IMDB_BINARY.p', 'rb')
        dataset = TUDataset(root='/tmp/'+dataset_name, name=dataset_name)
    elif dataset_name == "RB-MODEL":
        f = open("/nethome/eguha3/FGOPT/gdrive/maxclique-xu-n_200_300/train-0-1000.pkl", "rb")
        dataset = []
        idx = 0
        while True:
            idx += 1
            print(idx)
            try:
                graph, true_sol = pickle.load(f)
                if problem == "mc":
                    dataset.append(from_networkx(nx.complement(graph)))
                elif problem == "is":
                    dataset.append(from_networkx(graph))
            except EOFError:
                break
    print(len(dataset))
    return dataset

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem")
    parser.add_argument("--device")
    parser.add_argument("--name")
    parser.add_argument("--cooling")
    parser.add_argument("--beta", type=int, default=4)
    args = parser.parse_args()
    list_of_taus = []
    dataset = get_data(args.problem)
    num_epochs = 150
    name = args.name
    final_solution = []
    if os.path.exists("{}.pkl".format(name)):
        with open("{}.pkl".format(name), "rb") as f:
            final_solution = pickle.load(f)
    if args.cooling == "linear":
        initial_tau_alphas = [(1e3, -1), (1e4, -1), (5e4, -1), (1e5, -1)]
    elif args.cooling == "recip":
        initial_tau_alphas = [(50000, 1e-5), (50000, 3e-5), (50000, 1e-4), (50000, 3e-4)]
    elif args.cooling == "none":
        initial_tau_alphas = [(0, -1)]
    all_best_mus = []
    for seed in range(0, 20):
        for initial_tau, alpha in initial_tau_alphas:
            best_mu = -1
            best_net = None
            taus = [initial_tau]
            for epoch in range(num_epochs - 10):
                if args.cooling == "linear" or args.cooling == "none":
                    taus.append(initial_tau * (num_epochs - epoch)/(num_epochs))
                elif args.cooling == "recip":
                    taus.append(taus[-1]/(1 + alpha * taus[-1]))
            
            final_val = taus[-1]
            taus = [tau - final_val for tau in taus]
            taus.extend([0] * 10)

            mus = []
            stds = []
            for _ in range(1):
                mu_sam, std_sam, losses, net, initial_params, final_params, distances, final_losses = run_training(dataset, taus, seed, args.beta, args.device)
                mus.append(mu_sam)
                stds.append(std_sam)
                if mu_sam > best_mu:
                    best_mu = mu_sam
                    best_net = net
                final_solution.append((initial_tau, alpha, seed, mu_sam, std_sam,  net, initial_params, final_params,  distances, final_losses))
            mu = np.mean(mus)
            std = np.mean(stds)
            torch.save(net, "models/{}_{}it_{}bt_is.pt".format(name, initial_tau, alpha))
            # final_solution.append((mu, std, initial_tau, beta, losses))
            with open("{}.pkl".format(name), "wb") as f:
                pickle.dump(final_solution, f)
            all_best_mus.append(best_mu)
    print(all_best_mus)

    # final_solution = []
    # initial_tau = 10
    # taus = [initial_tau]

    # taus.extend([0] * 101)
    # mu, std = run_training(dataset, taus)
    # final_solution.append((mu, std, initial_tau, alpha))
    
 
    
        
    



