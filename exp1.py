from framework.grid import Grid
from framework.experiment import ExperimentOne

import pickle

from multiprocessing import Pool

CONVERGENCE = ["scatter", "nn_precision", "fn_precision", "epsilon_precision_recall"]

PROCESSES = 10

def _save(obj, name):
	with open(name, "wb") as f:
		pickle.dump(obj, f)

grid = {
"size": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
"sampling": ["random", "stratified"],
"dataset": ["mnist", "olivetti"],
"algorithm": ["umap", "tsne"]
}

def _gs(filename):
	print("Running ExperimentOne pointing to: {}".format(str(filename)))
	gs = Grid(grid, CONVERGENCE, ExperimentOne, progress=False)
	results = gs.run()
	_save(results, filename)

def main():
	targets = ["data/experimentone_{}.pkl".format(str(x+1)) for x in range(0, PROCESSES)]
	pool = Pool(PROCESSES)
	pool.map(_gs, targets)

if __name__ == '__main__':
	main()