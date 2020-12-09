import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from .sampler import Sampler
from .losses import get_KNN_precision, get_original_pr
from .fetcher import _fetch_mnist, _fetch_fasion_mnist, _fetch_olivetti, _fetch_coil

import umap
from openTSNE import TSNE

OPENML_MAP = {
    "mnist": _fetch_mnist,
    "fmnist": _fetch_fasion_mnist,
    "olivetti": _fetch_olivetti,
    "coil20": _fetch_coil
}


def _none_loss(*args):
    return 0


def _interpoint(x1, x2):
    print("WARNING: Interpoint distance function not implemented yet. Returning zero...")
    return 0


def _scatter(_, emb_x, y):
    from collections import defaultdict
    dd = defaultdict(list)
    mean = np.mean(emb_x, axis=0)
    for point, label in zip(emb_x, y):
        dd[label].append(point)
    num, denom = 0, 0

    for label, items in dict(dd).items():
        local_mean = np.mean(items, axis=0)
        diff = np.array([(local_mean-mean)])
        num += np.matmul(diff, diff.T)[0][0]
        denom += np.linalg.norm(np.var(items, axis=0)) ** 2

    return num/denom


def _nn_precision(X, emb_x, _):
    return get_KNN_precision(X, emb_x, mode="NN")


def _fn_precision(X, emb_x, _):
    return get_KNN_precision(X, emb_x, mode="FN")


def _epsilon_precision_recall(X, emb_x, _):
    return get_original_pr(X, emb_x)


EXP_ONE_LOSSES = {
    "scatter": _scatter,
    "nn_precision": _nn_precision,
    "fn_precision": _fn_precision,
    "epsilon_precision_recall": _epsilon_precision_recall,
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
                raise Exception("Value {} for parameter {} is not a valid parameter. Please use one of: {}".format(
                    x, var, str(options)))

            _throw_if_invalid(size, _fix_linspace(
                np.linspace(0, 1, 11).tolist()), "size")
            _throw_if_invalid(sampling, ["random", "stratified"], "sampling")
            _throw_if_invalid(
                dataset, ["mnist", "fmnist", "olivetti", "coil20"], "dataset")
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
            reducer = umap.UMAP(random_state=42)
        else:
            reducer = TSNE(random_state=42)
        return reducer

    def embed(self, dataset, labels):
        reducer = self.fetch_algorithm()
        scaled_data = StandardScaler().fit_transform(dataset)
        if self.algorithm == "umap":
            return reducer.fit_transform(scaled_data)
        embedding = reducer.fit(scaled_data)
        return embedding.transform(scaled_data)

    def loss(self, X, emb_x, y):
        result = []
        for loss in self.convergence:
            c = loss
            l = LOSSES[loss](X, emb_x, y)
            result.append(tuple([c, l]))
        return result

    def run(self):
        raise Exception("Not implemented in base class")


class ExperimentOne(Experiment):
    def __init__(self, size, sampling, convergence, dataset, algorithm):
        for c in convergence:
            if c not in EXP_ONE_LOSSES:
                raise Exception("{} function not defined for experiment one. Only losses in the format f(emb_x, y) are supported.".format(c))
        super().__init__(size, sampling, convergence, dataset, algorithm)

    def run(self):
        dataset, labels = self.generate_dataset()
        emb_x = self.embed(dataset, labels)
        losses = self.loss(dataset, emb_x, labels)
        items = []
        for loss in losses:
            c, l = loss
            items.append({
                "size": self.size,
                "sampling": self.sampling,
                "dataset": self.dataset,
                "algorithm": self.algorithm,
                "emb_x": emb_x,
                "labels": labels,
                "loss": l,
                "convergence": c
            })

        return items


class ExperimentTwo(Experiment):
    def __init__(self, size, sampling, convergence, dataset, algorithm):
        for c in convergence:
            if c not in EXP_TWO_LOSSES:
                raise Exception("{} function not defined for experiment two. Only losses in the format f(emb_x1, emb_x2) are supported.".format(c))
        super().__init__(size, sampling, convergence, dataset, algorithm)

    def generate_dataset(self):
        sampler = Sampler(self.sampling)
        x, y = sampler.run(OPENML_MAP[self.dataset]
                           (), self.size, holdout=EXP_TWO_HOLDOUT)
        x_h, y_h = sampler.fetch_holdout()
        return x, y, x_h, y_h

    def embed(self, dataset, labels, dataset_h, labels_h):
        reducer = self.fetch_algorithm()
        ss = StandardScaler()
        scaled_data = ss.fit_transform(dataset)
        scaled_holdout = ss.transform(dataset_h)

        if self.algorithm == "umap":
            reducer.fit(scaled_data)
            return reducer.transform(scaled_holdout)

        embedding = reducer.fit(scaled_data)
        return embedding.transform(scaled_holdout)

    def run(self):
        dataset, labels, dataset_h, labels_h = self.generate_dataset()
        emb_x = self.embed(dataset, labels, dataset_h, labels_h)
        losses = self.loss(dataset, emb_x, labels_h)
        items = []
        for loss in losses:
            c, l = loss
            items.append({
                "size": self.size,
                "sampling": self.sampling,
                "dataset": self.dataset,
                "algorithm": self.algorithm,
                "emb_x": emb_x,
                "labels": labels,
                "loss": l,
                "convergence": c
            })

        return items
