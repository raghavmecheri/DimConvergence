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

	def _random(self, dataset, fraction):
		size = int(len(dataset.target) * fraction)
		sample_set = _couple(dataset.data, dataset.target)
		return _decouple(random.sample(sample_set, size))

	def _hist(self, dataset, fraction):
		size = int(len(dataset.target) * fraction)
		def _transform(data):
			x = []
			for point in data:
				x.append(norm(point))
			return x
		def _hist_group(data, coupled, target_size):
			coupled = np.array(data)
			q1, q2, q3 = quantile(coupled, 0.25), quantile(coupled, 0.5), quantile(coupled, 0.75)
			stratified = defaultdict(list)
			for x, y in coupled:
				key = "q1"
				if x > q1 and x <= q2:
					key = "q2"
				elif x > q2 and x <= q3:
					key = "q3"
				else:
					key = "q4"
				stratified[key].append(tuple([x,y]))

			per_class = int(target_size/stratified.keys())
			for key in stratified.keys():
				rs = random.sample(stratified[key], per_class)
				x += rs

			return x

		datapoints = _transform(dataset.data)
		sample_set = _couple(datapoints, dataset.target)
		return _decouple(_hist_group(datapoints, sample_set, size))

	def _stratified(self, dataset, fraction):
		x, y = dataset.data, dataset.target
		x_target, _, y_target, _ = train_test_split(x, y, test_size=(1-fraction), random_state=42)
		return x_target, y_target

	def run(self, dataset, fraction):
		if self.stype == "random":
			return self._random(dataset, fraction)
		elif self.stype == "hist":
			raise Exception("Histogram sampling not implemented")
			return self._hist(dataset, fraction)
		elif self.stype == "stratified":
			return self._stratified(dataset, fraction)