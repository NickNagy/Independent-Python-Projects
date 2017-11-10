# kmeans.py
import numpy as np
import random

class KMeans(object):
    """ K Means Algorithm

    Parameters
    -------------
    k_max: int
        Maximum number of clusters over which to run the algorithm
    """

    def __init__(self, k_max=10):
        self.k_max = k_max

    def k_fits(self, X):
        """
        :param X: {array-like}, shape = [n_samples, n_features]
        :return: a tuple of the cluster centroids associated with the lost cost from k = 1 to k = k_max

        """
        results = []
        centroids, cost = self.k_means(X, 1)
        min_cost = cost
        min_cost_index = 0
        results.append(centroids)
        i = 1
        while i <= self.k_max:
            centroids, cost = self.k_means(X, i)
            if cost < min_cost:
                min_cost = cost
                min_cost_index = i
            results.append(centroids)
        return results[min_cost_index]

    def k_means(self, X, k):
        """
        :param X: {array-like}, shape = [n_samples, n_features]
        :param k: {int} number of clusters
        :return: a tuple of the optimal cluster centroids and associated cost function

        """
        m = X.shape[0]
        centroids = []  # centroid values
        cost = -1
        for i in range(k):
            centroids.append(X[random(m)])  # picks k number of random data points for centroids
        # while (not optimal):
        centroids_data = k * [0]  # data associated w/ each centroid
        # assign each sample to a centroid
        for i in range(m):
            min = np.subtract(X[i], centroids[0]) ** 2
            index = 0
            j = 1
            while j < k:
                if np.subtract(X[i], centroids[j]) ** 2 < min:
                    min = np.subtract(X[i], centroids[j]) ** 2
                    index = j
                j += 1
            centroids_data[index].append(i)
        # update each centroid to average of its associated data points
        for i in range(k):
            n = len(centroids_data[i])
            sum = []
            for j in range(n):
                sum = np.add(sum, centroids_data[j])
            centroids[i] = [element / n for element in sum]
        # --- while loop end ---
        return (centroids, cost)
