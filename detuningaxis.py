from signal import signal
import numpy as np

import sklearn as sk
import sklearn.exceptions
import sklearn.cluster
import sklearn.preprocessing

import unittest

class DetuningAxisFitter:
    def __init__(self, clustering_alg = sk.cluster.SpectralClustering) -> None:
        self.clustering_alg = clustering_alg
        self.pca = None
        self.pca_coordinate_center = None
        self.pca_coordinate_width = None
    def fit(self,x,y,z) -> None:
        signal = self.__clustering__(x,y,z)
        self.__fit_rectangle__(signal)
    def __clustering__(self,x,y,z):
        assert x.shape == y.shape == z.shape
        rw_data = np.stack((
        x.reshape(-1),
        y.reshape(-1),
        z.reshape(-1)) , axis = 1)
        #Normalise the dataset
        data = sk.preprocessing.StandardScaler().fit_transform(rw_data)
        #segment the data by clusterring
        cl = self.clustering_alg(n_clusters = 2)
        clusters = cl.fit_predict(data)
        #The signal coordinates should have a lower variance than the background
        masks = [clusters == i for i in range(2)]
        im = np.argmin( [ np.var( data[m,0:2] ) for m in masks] ) 
        signal = rw_data[masks[im]]
        self.signal = signal
        return signal

    def __fit_rectangle__(self,signal):
        #Use PCA analysis to find a new basis where
        #the covariance matrix of the signals coordinates is
        #diagonal.
        coordinate = signal[:,0:2]
        pca = sk.decomposition.PCA()
        tf_coordinate = pca.fit_transform(coordinate)
        #Use the median to find the center,
        #should be more robust to outlier than the mean.
        pca_coordinate_center = np.median(tf_coordinate ,axis = 0 )
        #Deduce the width of the uniform distribution
        #from the variances.
        pca_coordinate_width = np.sqrt( 12 * np.var(tf_coordinate ,axis = 0) ) / 2
        self.pca = pca
        self.pca_coordinate_center = pca_coordinate_center
        self.pca_coordinate_width = pca_coordinate_width

    def inverse_transform(self,x,y):
        x = np.reshape(x,-1)
        y = np.reshape(y,-1)
        if self.pca_coordinate_center is None or self.pca_coordinate_width is None or self.pca is None:
            raise sk.exceptions.NotFittedError("The estimator is not fitted")
        pca_pts = np.stack([ np.array(x), np.array(y) ], axis = 1)
        pca_pts =  pca_pts*self.pca_coordinate_width + self.pca_coordinate_center
        pts = self.pca.inverse_transform( pca_pts )
        return pts[:,0], pts[:,1]
    def __call__(self, x,y):
        return self.inverse_transform(x,y)
        
class TestDetuningAxisFitter(unittest.TestCase):
    def setUp(self):
        pass
    def test_creation(self):
        pass
    def test_fit_rectangle(self):
        pass

if __name__ == "__main__" :
    print("Hello world")
    unittest.main()

