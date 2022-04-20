from mimetypes import init
import pickle
import matplotlib.pyplot as plt

name = "entropy_alpha_800_1200_200e_is"
with open("{}.pkl".format(name), "rb") as f:
	final_solution = pickle.load(f)

mus, stds, initials_t0s, alphas, losses = list(zip(*final_solution))


def draw_means(mus, stds, initials_t0s, alphas, losses):
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
def draw_convergences(mus, stds, initials_t0s, alphas, losses):
	
	for i in range(len(initials_t0s)):
		t0 = initials_t0s[i]
		print(i)
		plt.plot(list(range(len(losses[i]))), losses[i], label=alphas[i])

	plt.legend()
	plt.xlabel("Epochs")
	plt.ylabel("Loss")

	plt.title(name)

	plt.savefig("{}_losses.png".format(name))
	plt.clf()

draw_means(mus, stds, initials_t0s, alphas, losses)
draw_convergences(mus, stds, initials_t0s, alphas, losses)