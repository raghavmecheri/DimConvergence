
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import numpy as np
from numpy.linalg import norm

K_VALUES = [10, 20, 30, 40, 50, 60]
EPSILON_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8]

def _get_k_neighborhood(dataset, k, radius=False):
    if radius:
        neighbours = NearestNeighbors(radius=k, algorithm='ball_tree').fit(dataset)
        return neighbours

    neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dataset)
    return neighbors


def get_KNN_precision(original, embedded, mode="NN"):
    result = []
    for k in K_VALUES:
        if mode == "NN":
            orig_neighbors  = _get_k_neighborhood(original, k)
            emb_neighbors  = _get_k_neighborhood(embedded, k)


            precision = 0
            for x,y in zip(original, embedded):

                x_neighbors = orig_neighbors.kneighbors(x.reshape(1,-1), return_distance=False)
                y_neighbors = emb_neighbors.kneighbors(y.reshape(1, -1), return_distance=False)

                precision += len(np.intersect1d(x_neighbors, y_neighbors))

            precision = precision / (k*len(original))

            result.append(precision)

        elif mode == "FN":
            orig_dist = pairwise_distances(original)
            emb_dist = pairwise_distances(embedded)

            precision=0
            for ind in range(len(original)):
                x_farthest = np.argpartition(orig_dist[ind], -k)[-k:]
                y_farthest = np.argpartition(emb_dist[ind], -k)[-k:]

                precision += len(np.intersect1d(x_farthest, y_farthest))

            precision = precision/ (k*(len(original)))

            result.append(precision)


        else:
            print(str(mode) + " is an invalid mode. Please select NN or FN")

    return result

def _get_epsilon_neighborhood_pr(dataset, embedded, latent=False):
    if latent:
        print("Warning Latent Embedding distances not built. Returning 0")
        return 0

    result_precision, result_recall = [], []

    for epsilon in EPSILON_VALUES:
        dataset_sim = _get_similarity_matrix(dataset)
        embedded_sim = _get_similarity_matrix(embedded)

        precision, recall = 0,0

        for ind in range(len(embedded)):

            d_mask = dataset_sim[ind] > epsilon
            emb_mask = embedded_sim[ind] > epsilon
            intersection = np.logical_and(d_mask, embedded_mask)

            intersection_count = np.count_nonzero(intersection)
            d_count = np.count_nonzero(d_mask)
            emb_count = np.count_nonzero(emb_mask)

            if emb_count:
                precision += intersection_count/emb_count
            if d_count:
                recall += intersection_count/d_count

        precision = precision/len(embedded)
        recall = recall/len(embedded)

        result_precision.append(precision)
        result_recall.append(recall)
    return result_precision, result_recall

def get_original_pr(original, embedded):
    return _get_epsilon_pr(original, embedded)

def get_latent_pr(latent, embedded):
    return _get_epsilon_pr(latent, embedded, latent=True)

def interpoint_distance(x1, x2):
    net = 0
    for a,b in zip(x1, x2):
        net += norm(a-b)
    return net

