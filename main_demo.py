import os

#import h5py
import matplotlib.pyplot as plt

import scipy
import scipy.io as sio
import sklearn
import pandas as pd
import umap
import numpy as np
from mda import *

# data = sio.loadmat('D:/MDA_visualization/Codes_For_Hongyi_MDA/Codes_For_Hongyi_MDA/mda/mda/data_MDA.mat')
# Y = data['Y']
# trainingFeatures = data['trainingFeatures']

#print(Y.shape)
#print(trainingFeatures.shape)

trainingFeatures = np.load('SR/feature4_test_pca.npy')
Y = np.load('SR/y_test.npy')
Y = Y.reshape(Y.shape[0],-1)
Y_pred = np.load('SR/y_test_pred_trained.npy')
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)

print(Y.shape)
print(trainingFeatures.shape)


X=trainingFeatures
    
FS = 16

neighborNum = 5
#clusterIdx = discoverManifold(Y, neighborNum)
clusterIdx_pred = discoverManifold(Y, neighborNum)

Yreg = mda(X,clusterIdx_pred)   

embedding_df = pd.DataFrame(Yreg)
embedding_df.to_csv('embeddingMDA_P.csv', index=False)
clusterIdx_pred_df = pd.DataFrame(clusterIdx_pred)
clusterIdx_pred_df.to_csv('clusterIdx_pred_P.csv', index=False)


# Visualize the data using a scatter plot
plt.scatter(Yreg[:, 0], Yreg[:, 1], c=clusterIdx_pred)
plt.show()

 
    