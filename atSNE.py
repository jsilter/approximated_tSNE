#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:03:43 2017

@author: diegollarrull
"""

#!/bin/bash
#from __future__ import division  # Python 2 users only

__doc__= """ Script for implementation of approximated tSNE algorithm.
See "Approximated and User Steerable tSNE for Progressive Visual Analytics" """

from pyflann import FLANN
import numpy as np
import scipy.sparse as sp

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.manifold.t_sne import _joint_probabilities_nn, _joint_probabilities
from sklearn.neighbors import BallTree
from sklearn.utils import check_array, check_random_state

import matplotlib.pyplot as plt
plt.style.use('ggplot')


string_types = "str"

def auto_params(dataset, target_precision):
    flann = FLANN()
    params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9);
    return params
    

class TSNE_mod(TSNE):
    
    def __init__(self, *args, **kwargs):
        self.rho = kwargs.pop("rho", 1.0)
        assert 0 < self.rho <= 1.0
        super(TSNE_mod, self).__init__(*args, **kwargs)
        
    def _fit(self, X, skip_num_points=0):
        """Fit the model using X as training data.

        Note that sparse arrays can only be handled by method='exact'.
        It is recommended that you convert your sparse array to dense
        (e.g. `X.toarray()`) if it fits in memory, or otherwise using a
        dimensionality reduction technique (e.g. TruncatedSVD).

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. Note that this
            when method='barnes_hut', X cannot be a sparse array and if need be
            will be converted to a 32 bit float array. Method='exact' allows
            sparse arrays and 64bit floating point inputs.

        skip_num_points : int (optional, default:0)
            This does not compute the gradient for points with indices below
            `skip_num_points`. This is useful when computing transforms of new
            data where you'd like to keep the old data fixed.
        """
        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.method == 'barnes_hut' and sp.issparse(X):
            raise TypeError('A sparse matrix was passed, but dense '
                            'data is required for method="barnes_hut". Use '
                            'X.toarray() to convert to a dense numpy array if '
                            'the array is small enough for it to fit in '
                            'memory. Otherwise consider dimensionality '
                            'reduction techniques (e.g. TruncatedSVD)')
        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                            dtype=np.float64)
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is "
                             "%f" % self.early_exaggeration)

        if self.n_iter < 200:
            raise ValueError("n_iter should be at least 200")

        if self.metric == "precomputed":
            if isinstance(self.init, string_types) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be used "
                                 "with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")
            distances = X
        else:
            if self.verbose:
                print("[t-SNE] Computing pairwise distances...")

            if self.metric == "euclidean":
                distances = pairwise_distances(X, metric=self.metric,
                                               squared=True)
            else:
                distances = pairwise_distances(X, metric=self.metric)

        if not np.all(distances >= 0):
            raise ValueError("All distances should be positive, either "
                             "the metric or precomputed distances given "
                             "as X are not correct")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1.0, 1)
        n_samples = X.shape[0]
        # the number of nearest neighbors to find
        k = min(n_samples - 1, int(3. * self.perplexity + 1))

        neighbors_nn = None
        if self.method == 'barnes_hut':
            if self.verbose:
                print("[t-SNE] Computing %i nearest neighbors..." % k)
            if self.metric == 'precomputed':
                # Use the precomputed distances to find
                # the k nearest neighbors and their distances
                neighbors_nn = np.argsort(distances, axis=1)[:, :k]
            elif self.rho >= 1:
                # Find the nearest neighbors for every point
                bt = BallTree(X)
                # LvdM uses 3 * perplexity as the number of neighbors
                # And we add one to not count the data point itself
                # In the event that we have very small # of points
                # set the neighbors to n - 1
                distances_nn, neighbors_nn = bt.query(X, k=k + 1)
                neighbors_nn = neighbors_nn[:, 1:]
            elif self.rho < 1:
                # Use pyFLANN to find the nearest neighbors
                myflann = FLANN()
                testset = X
                params = myflann.build_index(testset, algorithm="autotuned", target_precision=self.rho, log_level='info');
                neighbors_nn, distances = myflann.nn_index(testset, k+1, checks=params["checks"])
                neighbors_nn = neighbors_nn[:, 1:]
                
            P = _joint_probabilities_nn(distances, neighbors_nn,
                                        self.perplexity, self.verbose)
        else:
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
        assert np.all(np.isfinite(P)), "All probabilities should be finite"
        assert np.all(P >= 0), "All probabilities should be zero or positive"
        assert np.all(P <= 1), ("All probabilities should be less "
                                "or then equal to one")

        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X)
        elif self.init == 'random':
            X_embedded = None
        else:
            raise ValueError("Unsupported initialization scheme: %s"
                             % self.init)

        return self._tsne(P, degrees_of_freedom, n_samples, random_state,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points)

if __name__ == "__main__":
    
    # Generate test data
    num_clusters = 10
    num_high_dims = num_clusters
    num_points_per_cluster = 100
    num_pts = num_clusters * num_points_per_cluster
    
    test_data_high = np.zeros([num_pts, num_high_dims])
    test_class = np.zeros([num_pts])
    test_centers = np.zeros([num_clusters, num_high_dims])
    test_centers += 2*np.identity(num_clusters)
    test_cov = np.identity(num_clusters)
    
    np.random.seed(120489)
    
    for ci in xrange(num_clusters):
        cur_strt = ci*num_points_per_cluster
        cur_end = cur_strt + num_points_per_cluster
        
        cur_center = test_centers[ci,:]
        cur_pts = cur_center + np.random.normal(scale=0.5, size=[num_points_per_cluster, num_high_dims])
        
        test_data_high[cur_strt:cur_end, :] = cur_pts
        test_class[cur_strt:cur_end] = ci
        
    perplexity = 50
    
    if False:
        myflann = FLANN()
        precision = 0.5
        testset = test_data_high
        params = myflann.build_index(testset, algorithm="autotuned", target_precision=precision, log_level='info');
        result, dists = myflann.nn_index(testset, 3*50, checks=params["checks"]);
        
    
    tsne = TSNE_mod(perplexity=perplexity, n_components=2, init='pca', n_iter=500, random_state=1941)
    low_dim_embeds = tsne.fit_transform(test_data_high)
    
    color_map_name = 'gist_rainbow'
    cmap = plt.get_cmap(color_map_name)
    
    plt.figure()
    plt.hold(True)
    for cc in range(num_clusters):
        # Plot each class using a different color
        cfloat = (cc+1.0) / num_clusters
        keep_points = np.where(test_class == cc)[0]
        cur_plot = low_dim_embeds[keep_points, :]
        
        cur_color = cmap(cfloat)
        
        # Scatter plot
        plt.plot(cur_plot[:,0], cur_plot[:,1], 'o', color=cur_color, alpha=0.5)
    
    plt.title('tSNE Visualization. Perplexity %d' % perplexity)
    plt.legend(loc='lower right', numpoints=1, fontsize=6, framealpha=0.5)
    plt.show()
    
    
    
    