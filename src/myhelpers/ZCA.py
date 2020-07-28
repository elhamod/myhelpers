import numpy as np
import torch

class ZCA:
    def __init__(self, data):
        self.epsilon = 1e-5
        self.mean = None
        self.std = None
            
        self.data = self.scaleData(data)

    def computeZCAMatrix(self):
        #This function computes the ZCA matrix for a set of observables X where
        #rows are the observations and columns are thde variables (M x C x W x H matrix)
        #C is number of color channels and W x H is width and height of each image

        #reshape data from M x C x W x H to M x N where N=C x W x H 
        X = self.data
        X = X.reshape(-1, X.shape[1]*X.shape[2]*X.shape[3])

        # compute the covariance 
        cov = np.cov(X, rowvar=False)   # cov is (N, N)

        # singular value decomposition
        U,S,V = np.linalg.svd(cov)     # U is (N, N), S is (N,1) V is (N,N)
        # build the ZCA matrix which is (N,N)
        self.zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + self.epsilon)), U.T))


        return torch.from_numpy(self.zca_matrix).float()
    
    def scaleData(self, data):
        data=np.stack(data, axis=0)
        
        #normalize the data to [0 1] range
        data=data/255

        #compute mean and std and normalize the data to -1 1 range with 1 std
        if self.mean is None:
            self.mean=(data.mean(axis=0))
        if self.std is None:
            self.std=(data.std(axis=0))
        
#         scaled = torch.tensor(np.multiply(1/(self.std+self.epsilon),np.add(data,-self.mean)))
        scaled = torch.tensor(np.add(data,-self.mean))
        return scaled
    
    def scaleSampleToUnity(self, sample):
        scaled = (sample - sample.min())/(sample.max() - sample.min())
        return scaled
        
    