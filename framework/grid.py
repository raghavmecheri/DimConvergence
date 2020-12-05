import itertools
from tqdm.auto import tqdm

class Grid():
	def __init__(self, spec, experiment):
		def _validate(spec_obj):
			for key in spec_obj.keys():
				if key not in ["size", "sampling", "convergence", "dataset", "algorithm"]:
					raise Exception("Invalid key {} found in spec object!".format(key))
			for key in ["size", "sampling", "convergence", "dataset", "algorithm"]:
				if key not in spec_obj.keys():
					raise Exception("Required key {} not found in spec object!".format(key))

		_validate(spec)
		self.spec = spec
		self.exp_class = experiment

	def run(self):
		keys, values = zip(*self.spec.items())
		results = []
		permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
		for permutation in tqdm(permutations):
			size = permutation['size']
			sampling = permutation['sampling']
			convergence = permutation['convergence']
			dataset = permutation['dataset']
			algorithm = permutation['algorithm']
			results.append(self.exp_class(size, sampling, convergence, dataset, algorithm).run())

		return results


		