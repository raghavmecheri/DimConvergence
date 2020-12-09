from framework.grid import Grid
from framework.experiment import ExperimentOne

import pickle

from multiprocessing import Pool

PROCESSES = 10

def _save(obj, name):
	with open(name, "wb") as f:
		pickle.dump(obj, f)

grid = {
"size": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
"sampling": ["random", "stratified"],
"convergence": ["nn_precision", "fn_precision"],
"dataset": ["mnist", "fmnist", "olivetti"],
"algorithm": ["umap", "tsne"]
}

def _gs(filename):
	gs = Grid(grid, ExperimentOne)
	results = gs.run()
	results_json = [x.fetch() for x in results]
	_save(results_json, filename)

def main():
	targets = ["data/experimentone_{}.pkl".format(str(x+1)) for x in range(0, PROCESSES)]
	pool = Pool(PROCESSES)
	pool.map(_gs, targets)

if __name__ == '__main__':
	main()