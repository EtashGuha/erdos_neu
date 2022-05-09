from mimetypes import init
import pickle
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import torch

name = "lin_300e_rb23_is.pkl"
with open("/nethome/eguha3/erdos_neu/{}.pkl".format(name), "rb") as f:
	final_solution = pickle.load(f)

# final_solution = [ele for ele in final_solution if len(ele) == 6]

initial_tau, alpha, seed, mu_sam, std_sam,  net, initial_params, final_params,  distances, final_losses = list(zip(*final_solution))
distances = torch.tensor(distances).numpy()
mu_sam = torch.tensor(mu_sam).numpy()
seed = np.matrix(seed)
untaus = np.unique(initial_tau)
unalphas = np.unique(alpha)
tau_b_idxs = {}

tau_alpha_pairs = list(product(untaus, unalphas))
for tau_alpha_pair in tau_alpha_pairs:
	tau_b_idxs[tau_alpha_pair] = []
for tau_alpha_pair in tau_alpha_pairs:
	for i in range(distances.shape[0]):
		if initial_tau[i] == tau_alpha_pair[0] and alpha[i] == tau_alpha_pair[1]:
			tau_b_idxs[tau_alpha_pair].append(i)

all_means = []
all_distances = []
for tau_alpha_pair in tau_alpha_pairs:
	percent_deviation = distances[tau_b_idxs[tau_alpha_pair]] 
	percent_deviation[percent_deviation == np.inf] = np.nan
	percent_deviation = abs(percent_deviation)
	means = mu_sam[tau_b_idxs[tau_alpha_pair]]
	distancemean = np.nanmean(abs(percent_deviation), axis=1)
	
	all_means.extend(list(means))
	all_distances.extend(list(distancemean))
	print(seed[:, tau_b_idxs[tau_alpha_pair]])
	print(tau_alpha_pair)
	print(distancemean)

plt.scatter(all_means, all_distances)
plt.xlabel("means")
plt.ylabel("distances")
plt.title(name)
plt.savefig("/nethome/eguha3/erdos_neu/{}_distanceratio.png".format(name))
breakpoint()
all_vals = {}


for idx, alpha in enumerate(alphas):
	if alpha not in all_vals:
		all_vals[alpha] = []
	all_vals[alpha].append(distances[idx])

for key in all_vals:
	all_vals[key] = np.asarray(all_vals[key])

all_percent_deviations = []
for key in all_vals:
	percent_deviation = np.divide(all_vals[key], initializations)
	percent_deviation[percent_deviation == np.inf] = np.nan
	mean = np.nanmean(abs(percent_deviation))
	all_percent_deviations.append(mean)


distances_mat = np.asarray(distances)


def draw_means(mus, stds, initials_t0s, alphas):
	dic = {}
	for i in range(len(initials_t0s)):
		tau = initials_t0s[i]
		if tau in dic:
			dic[tau].append((mus[i], initials_t0s[i], alphas[i]))
		else:
			dic[tau] = [(mus[i], initials_t0s[i], alphas[i])]
	
	for key in dic:
		ms, t0s, als= list(zip(*dic[key]))
		plt.scatter(als, ms, label=key)
	plt.xscale("log")
	plt.xlabel("Alphas")
	plt.ylabel("Means")
	plt.legend()
	plt.title(name)

	plt.savefig("{}_means.png".format(name))
	plt.clf()
# def draw_convergences(mus, stds, initials_t0s, alphas, losses):
	
# 	for i in range(len(initials_t0s)):
# 		t0 = initials_t0s[i]
# 		print(i)
# 		plt.plot(list(range(len(losses[i]))), losses[i], label=alphas[i])

# 	plt.legend()
# 	plt.xlabel("Epochs")
# 	plt.ylabel("Loss")

# 	plt.title(name)

# 	plt.savefig("{}_losses.png".format(name))
# 	plt.clf()

# draw_means(mus, stds, initials_t0s, alphas)
# draw_convergences(mus, stds, initials_t0s, alphas, losses)
