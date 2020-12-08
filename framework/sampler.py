import random
import numpy as np
from numpy.linalg import norm
from numpy import quantile
from collections import defaultdict
from sklearn.model_selection import train_test_split

def _couple(data, targets):
	return [tuple([x,y]) for x,y in zip(data, targets)]

def _decouple(coupled):
	x, y = [], []
	for entry in coupled:
		x.append(entry[0])
		y.append(entry[1])
	return np.array(x), np.array(y)

class Sampler():
	def __init__(self, stype="random"):
		self.stype = stype
		self.x_holdout = None
		self.y_holdout = None

	def _random(self, x, y, fraction):
		size = int(len(x) * fraction)
		sample_set = _couple(x, y)
		return _decouple(random.sample(sample_set, size))

	def _stratified(self, x, y, fraction):
		if fraction == 1:
			return x, y

		x_target, _, y_target, _ = train_test_split(x, y, test_size=(1-fraction), random_state=42)
		return x_target, y_target

	def run(self, dataset, fraction, holdout=0):
		x, y = dataset.data, dataset.target
		
		if holdout == 0:
			x_target = x
			y_target = y
		else:
			x_target, x_holdout, y_target, y_holdout = train_test_split(x, y, test_size=holdout, random_state=42)
			self.x_holdout = x_holdout
			self.y_holdout = y_holdout

		if self.stype == "random":
			return self._random(x_target, y_target, fraction)
		elif self.stype == "stratified":
			return self._stratified(x_target, y_target, fraction)

	def fetch_holdout(self):
		return self.x_holdout, self.y_holdout
