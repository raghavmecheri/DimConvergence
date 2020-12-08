
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import numpy as np
from numpy.linalg import norm

K_VALUES = [10, 20, 30, 40, 50, 60]
EPSILON_VALUES = [1, 3, 5, 7, 9]

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

def _get_epsilon_pr(dataset, embedded, latent=False):
    result = []

    for epsilon in EPSILON_VALUES:
        precision, recall = 0,0

        if latent:
            print("Warning Latent Embedding distances not built. Returning 0")
            return 0

        dataset_neighbors  = _get_k_neighborhood(dataset, epsilon, radius=True)
        embedded_neighbors  = _get_k_neighborhood(embedded, epsilon, radius=True)

        for x,y in zipped(dataset, embedded):

            x_neighbors = dataset_neighbors.radius_neighbors(x.reshape(1,-1), return_distance=False)
            y_neighbors = embedded_neighbors.radius_neighbors(y.reshape(1, -1), return_distance=False)
            
            precision += len(np.intersect1d(x_neighbors, y_neighbors))/len(y_neighbors)
            recall += len(np.intersect1d(x_neighbors, y_neighbors))/len(x_neighbors)

        precision = precision/len(embedded)
        recall = recall/len(embedded)

        result.append(tuple(precision, recall))

def get_original_pr(original, embedded):
    return _get_epsilon_pr(original, embedded)

def get_latent_pr(latent, embedded):
    return _get_epsilon_pr(latent, embedded, latent=True)


def interpoint_distance(x1, x2):
    net = 0
    for a,b in zip(x1, x2):
        net += norm(a-b)
    return net




