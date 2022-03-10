import pickle
import matplotlib.pyplot as plt

with open("recip_temperature.pkl", "rb") as f:
	final_solution = pickle.load(f)

mus, stds, initials_t0s, alphas = list(zip(*final_solution))

dic = {}
for i in range(len(initials_t0s)):
	tau = initials_t0s[i]
	if tau in dic:
		dic[tau].append((mus[i], alphas[i]))
	else:
		dic[tau] = [(mus[i], alphas[i])]

for key in dic:
	ms, als = list(zip(*dic[key]))
	plt.scatter(als, ms, label=key)
plt.xlabel("Alphas")
plt.ylabel("Means")
plt.legend()
plt.title("Recip Cooling Schedule")

plt.savefig("Recip_cool.png")