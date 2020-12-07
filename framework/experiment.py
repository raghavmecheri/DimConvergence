import numpy as np
from sklearn.datasets import load_digits, fetch_openml, fetch_olivetti_faces, fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from .sampler import Sampler
from .result import Result

import umap
from sklearn.manifold import TSNE

def _fetch_mnist():
	return load_digits()

def _fetch_fasion_mnist():
	return fetch_openml(name="Fashion-MNIST")

def _fetch_olivetti():
	return fetch_olivetti_faces()

def _fetch_coil():
	raise Exception("20newsgroups not supported at the moment!")
	# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
	return fetch_20newsgroups()

OPENML_MAP = {
	"mnist": _fetch_mnist,
	"fmnist": _fetch_fasion_mnist,
	"olivetti": _fetch_olivetti,
	"coil20": _fetch_coil
}

def _spread(emb_x, y):
	raise Exception("WARNING: Spread loss function not implemented yet. Returning zero...")

def _avg_recall(emb_x, y):
	print("WARNING: Average recall measure function not implemented yet. Returning zero...")
	return 0

def _interpoint(emb_x, y):
	from scipy.spatial.distance import cdist
	return np.mean(cdist(emb_x, emb_x, metric='euclidean'))

EXP_ONE_LOSSES = {
	"spread": _spread,
	"avgrecall": _avg_recall
}

EXP_TWO_LOSSES = {
	"interpoint": _interpoint
}

class Experiment():

	def __init__(self, size, sampling, convergence, dataset, algorithm):
		def _check_param_validity(sizes, sampling, convergence, dataset, algorithm):
			def _throw_if_invalid(x, options, var):
				if x in options:
					return None
				raise Exception("Value {} for parameter {} is not a valid parameter. Please use one of: {}".format(x, var, str(options)))

			_throw_if_invalid(size, np.linspace(0,1,11).tolist(), "size")
			_throw_if_invalid(sampling, ["random", "hist", "stratified"], "sampling")
			_throw_if_invalid(convergence, ["spread", "avgrecall", "interpoint"], "convergence")
			_throw_if_invalid(dataset, ["mnist", "fmnist", "olivetti", "coil20"], "dataset")
			_throw_if_invalid(algorithm, ["umap", "tsne"], "algorithm")

		_check_param_validity(size, sampling, convergence, dataset, algorithm)

		self.size = size
		self.sampling = sampling
		self.convergence = convergence
		self.dataset = dataset
		self.algorithm = algorithm

	def generate_dataset(self):
		return Sampler(self.sampling).run(OPENML_MAP[self.dataset](), self.size)

	def fetch_algorithm(self):
		reducer = None
		if self.algorithm == "umap":
			reducer = umap.UMAP()
		else:
			reducer = TSNE(n_components=2)
		return reducer

	def embed(self, dataset, labels):
		reducer = self.fetch_algorithm()
		scaled_data = StandardScaler().fit_transform(dataset)
		embedding = reducer.fit_transform(scaled_data)
		return embedding

	def loss(self, emb_x, y):
		return LOSSES[self.convergence](emb_x, y)

	def run(self):
		raise Exception("Not implemented in base class")


class ExperimentOne(Experiment):
	def __init__(self, size, sampling, convergence, dataset, algorithm):
		if convergence not in EXP_ONE_LOSSES:
			raise Exception("{} function not defined for experiment one. Only losses in the format f(emb_x, y) are supported.".format(convergence))
		super().__init__(size, sampling, convergence, dataset, algorithm)

	def run(self):
		dataset, labels = self.generate_dataset()
		emb_x = self.embed(dataset, labels)
		return Result(self.size, self.sampling, self.convergence, self.dataset, self.algorithm, self.loss(emb_x, labels), emb_x, labels)
