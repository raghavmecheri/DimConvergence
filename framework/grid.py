import itertools
from tqdm.auto import tqdm

class Grid():
    def __init__(self, spec, convergence, experiment, progress=True):
        def _validate(spec_obj):
            for key in spec_obj.keys():
                if key not in ["size", "sampling", "dataset", "algorithm"]:
                    raise Exception(
                        "Invalid key {} found in spec object!".format(key))
            for key in ["size", "sampling", "dataset", "algorithm"]:
                if key not in spec_obj.keys():
                    raise Exception(
                        "Required key {} not found in spec object!".format(key))

        _validate(spec)
        self.spec = spec
        self.exp_class = experiment
        self.convergence = convergence
        self.progress = progress

    def run(self):
        keys, values = zip(*self.spec.items())
        results = []
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        permutations = tqdm(permutations) if self.progress == True else permutations
        for permutation in permutations:
            size = permutation['size']
            sampling = permutation['sampling']
            dataset = permutation['dataset']
            algorithm = permutation['algorithm']
            results += self.exp_class(size, sampling, self.convergence, dataset, algorithm).run()

        return results