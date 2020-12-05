class Result():
	
	def __init__(self, size, sampling, convergence, dataset, algorithm, loss, emb_x, y):
		self.size = size
		self.sampling = sampling
		self.convergence = convergence
		self.dataset = dataset
		self.algorithm = algorithm
		self.loss = loss
		self.emb_x = emb_x
		self.y = y

	def fetch(self):
		return {
		"size": self.size,
		"sampling": self.sampling,
		"convergence": self.convergence,
		"dataset": self.dataset,
		"algorithm": self.algorithm,
		"loss": self.loss,
		"emb_x": self.emb_x,
		"y": self.y
		}

	def __str__(self):
		return str(self.fetch())