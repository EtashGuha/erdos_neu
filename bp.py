import factorgraph as fg
import numpy as np
import time

def solve_mvc(graph):
	factor_graph = fg.Graph()
	for node in graph.nodes():
		factor_graph.rv(str(node), 2)
		factor_graph.factor([str(node)], potential=np.array([.5, .5]))

	for u, v in graph.edges():
		factor_graph.factor([str(u), str(v)], potential=np.array([[0.10923, 0.296922],[0.296922, 0.296922]]))
	t1 = time.time()
	factor_graph.lbp(normalize=True, max_iters=2)
	t2 = time.time()
	breakpoint()
	sol_val = 0

	final_probs = [0] * len(graph.nodes())
	for node in factor_graph._sorted_nodes():
		if isinstance(node, fg.RV):
			# probs = [1, 1]
			# for factor in node._factors:
			#     if len(factor._rvs) > 1:
			#         probs[1] *= factor.get_outgoing()[0][1]
			#         probs[0] *= factor.get_outgoing()[0][0]
			# prob_of_one =  probs[1] / sum(probs)
			probs, _ = node.get_belief()
			final_probs[int(node.name)] = probs[1]/np.sum(probs)

	#         assignment = np.random.binomial(1, prob_of_one)
	#         sol_val += assignment

	sol_val = cond_eval(graph, final_probs)
	return sol_val, t2 - t1

def cond_eval(g, node_prob):
	breakpoint()
	penalty = 1.0
	# float* node_prob = static_cast<float*>(_node_prob);
	picked = [False] * len(g.nodes())
	visited = [False] * len(g.nodes())

	cover_size = [0] * len(g.nodes())

	for u, v in g.edges():
		cover_size[u] += 1
		cover_size[v] += 1
	
	covered_edge = 0
	cnt = 0
	
	for i in range(len(g.nodes())):
		zero_cost = 0
		for v in g[i]:
			if cover_size[v] > 0:
				v_prob = picked[v] * visited[v] + (1 - visited[v]) * node_prob[v]
				zero_cost += 1 - v_prob

		zero_cost *= penalty
		visited[i] = True

		if zero_cost >= 1:  
			picked[i] = True
			covered_edge += cover_size[i]
			for v in g[i]:
				if cover_size[v] > 0:
					cover_size[v] -= 1
			cover_size[i] = -1
			cnt += 1
			if covered_edge == len(g.edges()):
				break
		else:
			picked[i] = False

	if covered_edge == len(g.edges()):
		return cnt
	# // in case we didn't cover all the edges
	for i in range(len(g.nodes())):
		k = g.nodes[i]
		if picked[i] or cover_size[i] <= 0:
			continue
		covered_edge += cover_size[i]
		for v in g[i]:
			if cover_size[v] > 0:
				cover_size[v]       
		cover_size[i] = -1
		cnt += 1
		if covered_edge == len(g.edges()):
			break

	return cnt