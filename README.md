# Manifold discovery and analysis

This code implements the  manifold discovery and analysis (MDA) algorithm presented in "Revealing Hidden Patterns in Deep Neural Network Feature Space Continuum via Manifold Learning, in press, Nature Communications, 2023".
Please run demo_MDA.ipynb for analyzing deep learning features for five different tasks (medical image segmentation and superresolution, gene expression prediction, survival prediction, and COVID-19 x-ray image classification). You can also run our reproducible codes in Code Ocean (https://doi.org/10.24433/CO.0076930.v1). 


# Required packages

scikit-learn, scipy, tensorflow, umap-learn, pandas, matplotlib, jupyter, jupyterlab and mat73. The tested package versions are: jupyter (1.0.0),jupyterlab (3.6.1), mat73 (0.62), matplotlib (3.4.1), pandas (1.2.4), scikit-learn (0.24.2), scipy (1.6.1), tensorflow (2.5.1), umap-learn (0.5.3).

# Data

The analyzed data can be downloaded from https://drive.google.com/drive/folders/1MUvngB04qd1XU6oFV_aJwSaScj0KP2c3?usp=sharing.

# Example code
## The following code shows the MDA analyses of deep neural network (DNN) features at intermediate layers for five different tasks
```
# For the tasks below, five datasets analysed in the manuscript will be automatically loaded. 
# However, you can upload your own dataset, and analyze it using MDA
# Our data were saved as .npy file to reduce the data size (normally .csv file needs more disk space). 
# However, .csv or other type of files can also be loaded and analyzed using MDA
```
```
# Load all necessary python packages needed for the reported analyses
# in our manuscript
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

%matplotlib inline

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import sklearn
import umap
import pandas as pd
from umap.parametric_umap import ParametricUMAP
import numpy as np
from mda import *

# Font size for all the MDA visualizations shown below   
FS = 16
```
## Example 1 - MDA analysis of the DNN features in superresolution task

### Segmentation Network

In the segmentation task, we employed Dense-UNet for automatic brain tumor segmentation from MR images. The Dense-UNet combines the U-net with the dense concatenation to deepen the depth of the network architecture and achieve feature reuse. The network is formed from seven dense blocks (four in encoder and three in decoder), each of them stacks eight convolutional layers. Every two convolutional layers are linked together in a feed-forward mode to maximize feature reuse.

### Dataset and feature selection
Here, we used BraTS 2018 dataset, which provides multimodality 3D MRI images with tumor segmentation labels annotated by physicians. The dataset includes 484 cases in total, which can be divided into 210 high-grade gliomas (HGG) and 75 low-grade gliomas (LGG) cases.
To visualize the intermediate layers of the Dense-UNet, we selected features of (a) the second convolutional layer in the third dense block, (b) the 8th convolutional layer in the fourth dense block, (c) the second convolutional layer in the 6th dense block, and (d) the last convolutional layer before the final output. In this demo, feature (d) is given as a example.

```
# Number of neighbors in MDA analyses
neighborNum = 5

# Load feature data extracted by the SRGAN at umsampling block from test images
testDataFeatures = np.load('../data/SR/feature4_test_pca.npy')
# Load data labels (target high resolution images) corresponding to low resolution test images
Y = np.load('../data/SR/y_test.npy')
# Reshape the target images into vectors so that they can be analyzed by MDA 
Y = Y.reshape(Y.shape[0],-1)
# Load output images prediced by the SRGAN
Y_pred = np.load('../data/SR/y_test_pred_trained.npy')
# Reshape the predicted output images into vectors so that they can be analyzed by MDA 
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)

# Create color map for MDA visualization from the target manifold topology
clusterIdx = discoverManifold(Y, neighborNum)
# Compute the outline of the output manifold
clusterIdx_pred = discoverManifold(Y_pred, neighborNum)
# Use the outline of the output manifold to generate the MDA visualization of the SRGAN features
Yreg = mda(testDataFeatures,clusterIdx_pred)   

# Plot the MDA results
plt.figure(1)
plt.scatter(Yreg[:,0],Yreg[:,1],c=clusterIdx.T, cmap='jet', s=5)
plt.xlabel("MDA1")
plt.ylabel("MDA2")
plt.title('MDA visualization of the SRGAN features for superresolution task')
```

### Visualization and analysis of Dense-UNet features for image segmentation task:

<img src="mda1.png">

Figure 1. Visualization and analysis of Dense-UNet features for segmentation task after training the network. Here, B3-L2 denotes the 2nd layer of the 3rd dense block, B4-L8 denotes the 8th layer of the 4th dense block, B6-L2 denotes the 2nd layer of the 6th dense block, and B7-L8 denotes the last layer before the final output. t-SNE, UMAP and MDA results are shown in (a), (b), (c), respectively for training and testing datasets at different network layers. The colorbar denotes the normalized manifold distance.
