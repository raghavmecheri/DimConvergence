import numpy as np
import math
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

def _spread(emb_x, y):
	raise Exception("WARNING: Spread loss function not implemented yet. Returning zero...")

def _avg_recall(emb_x, y):
	print("WARNING: Average recall measure function not implemented yet. Returning zero...")
	return 0

def _none_loss(emb_x, y):
	return 0

def _interpoint(x1, x2):
	print("WARNING: Interpoint distance function not implemented yet. Returning zero...")
	return 0

OPENML_MAP = {
	"mnist": _fetch_mnist,
	"fmnist": _fetch_fasion_mnist,
	"olivetti": _fetch_olivetti,
	"coil20": _fetch_coil
}

EXP_ONE_LOSSES = {
	"spread": _spread,
	"avgrecall": _avg_recall,
	"none": _none_loss
}

EXP_TWO_LOSSES = {
	"none": _none_loss
}

LOSSES = {**EXP_ONE_LOSSES, **EXP_TWO_LOSSES}

EXP_TWO_HOLDOUT = 0.3

class Experiment():

	def __init__(self, size, sampling, convergence, dataset, algorithm):

		def _fix_linspace(values):
			def _truncate(number, decimals=0):
			    if decimals == 0:
			        return math.trunc(number)
			    factor = 10.0 ** decimals
			    return math.trunc(number * factor) / factor

			return [_truncate(x, 1) for x in values]

		def _check_param_validity(sizes, sampling, convergence, dataset, algorithm):
			def _throw_if_invalid(x, options, var):
				if x in options:
					return None
				raise Exception("Value {} for parameter {} is not a valid parameter. Please use one of: {}".format(x, var, str(options)))

			_throw_if_invalid(size, _fix_linspace(np.linspace(0,1,11).tolist()), "size")
			_throw_if_invalid(sampling, ["random", "stratified"], "sampling")
			_throw_if_invalid(convergence, ["spread", "avgrecall", "interpoint", "none"], "convergence")
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

class ExperimentTwo(Experiment):
	def __init__(self, size, sampling, convergence, dataset, algorithm):
		if convergence not in EXP_TWO_LOSSES:
			raise Exception("{} function not defined for experiment two. Only losses in the format f(emb_x1, emb_x2) are supported.".format(convergence))
		super().__init__(size, sampling, convergence, dataset, algorithm)

	def generate_dataset(self):
		sampler = Sampler(self.sampling)
		x, y = sampler.run(OPENML_MAP[self.dataset](), self.size, holdout=EXP_TWO_HOLDOUT)
		x_h, y_h = sampler.fetch_holdout()
		return x, y, x_h, y_h

	def embed(self, dataset, labels, dataset_h, labels_h):
		reducer = self.fetch_algorithm()
		scaled_data = StandardScaler().fit_transform(dataset)
		reducer.fit(dataset)
		return reducer.transform(dataset_h)

	def run(self):
		dataset, labels, dataset_h, labels_h = self.generate_dataset()
		emb_x = self.embed(dataset, labels, dataset_h, labels_h)
		return Result(self.size, self.sampling, self.convergence, self.dataset, self.algorithm, self.loss(emb_x, labels_h), emb_x, labels)
