import numpy as np

def uniform_sampling(num_clients, num_selected):
	return np.random.permutation(num_clients)[:num_selected]