
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import numpy as np
from numpy.linalg import norm
import scipy
from sklearn.metrics.pairwise import rbf_kernel as rbf


K_VALUES = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

def _get_k_neighborhood(dataset, k, radius=False):
    if radius:
        neighbours = NearestNeighbors(radius=k, algorithm='ball_tree').fit(dataset)
        return neighbours

    neighbors = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(dataset)
    return neighbors

def _get_similarity_matrix(dataset):
    gamma = (1/np.var(pairwise_distances(dataset)))*0.1
    return rbf(dataset, gamma=gamma)

def _compute_embedded_similarities(Y):
    N = Y.shape[0]
    sqdistance = calculate_distances(Y)
    one_over = 1./(sqdistance + 1)
    p_Yp_given_Y =  one_over/one_over.sum(axis=1).reshape((N, 1)) 
    return p_Yp_given_Y

def _compute_original_similarities(X, sigma, metric, approxF=0):

    N = X.shape[0]
    sigma = np.full((1, 1797), sigma)
    if metric == 'euclidean':
        sqdistance = calculate_distances(X)
    elif metric == 'precomputed':
        sqdistance = X**2
    else:
        raise Exception('Invalid metric')
    euc_dist     = np.exp(-sqdistance / (np.reshape(2*(sigma**2), [N, 1])))
    np.fill_diagonal(euc_dist, 0.0 )

    if approxF > 0:
        sorted_euc_dist = euc_dist[:,:]
        np.sort(sorted_euc_dist, axis=1)
        row_sum = np.reshape(np.sum(sorted_euc_dist[:,1:approxF+1], axis=1), [N, 1])
    else:
        row_sum = np.reshape(np.sum(euc_dist, axis=1), [N, 1])

    return euc_dist/row_sum 


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


EPSILON_VALUES = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

def _get_epsilon_neighborhood_pr(dataset, embedded, latent=False):
    if latent:
        print("Warning Latent Embedding distances not built. Returning 0")
        return 0

    result_precision, result_recall = [], []

    dataset_sim = _compute_original_similarities(dataset, sigma=11.63, metric="euclidean")
    embedded_sim = _compute_embedded_similarities(embedded)

    for epsilon in EPSILON_VALUES:

        precision = 0
        recall = 0

        for ind in range(len(embedded)):

            d_mask = dataset_sim[ind] > epsilon
            emb_mask = embedded_sim[ind] > epsilon
            intersection = np.logical_and(d_mask, emb_mask)
            
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
    return _get_epsilon_neighborhood_pr(original, embedded)

def get_latent_pr(latent, embedded):
    return _get_epsilon_pr(latent, embedded, latent=True)

def interpoint_distance(x1, x2):
    net = 0
    for a,b in zip(x1, x2):
        net += norm(a-b)
    return net

