# Manifold discovery and analysis

This code implements the  manifold discovery and analysis (MDA) algorithm presented in "Revealing Hidden Patterns in Deep Neural Network Feature Space Continuum via Manifold Learning, in press, Nature Communications, 2023".
Please run demo_MDA.ipynb for analyzing deep learning features for five different tasks (medical image segmentation and superresolution, gene expression prediction, survival prediction, and COVID-19 x-ray image classification). You can also run our reproducible codes in Code Ocean (https://doi.org/10.24433/CO.0076930.v1). 

# Citation

Md Tauhidul Islam, Zixia Zhou, Hongyi Ren1, Masoud Badiei Khuzani,
Daniel Kapp, James Zou, Lu Tian, Joseph C. Liao and Lei Xing. 2023, "Revealing Hidden Patterns in Deep Neural Network Feature Space Continuum via Manifold Learning", in press, \emph{Nature Communications}.

# Installation
The easiest way to start with MDA is to install it using PyPI.

```python
pip install MDA-learn
```

# Required packages

scikit-learn, scipy, tensorflow, umap-learn, pandas, matplotlib, jupyter, jupyterlab and mat73. The tested package versions are: jupyter (1.0.0),jupyterlab (3.6.1), mat73 (0.62), matplotlib (3.4.1), pandas (1.2.4), scikit-learn (0.24.2), scipy (1.6.1), tensorflow (2.5.1), umap-learn (0.5.3).

# Data

The analyzed data can be downloaded from https://drive.google.com/drive/folders/1MUvngB04qd1XU6oFV_aJwSaScj0KP2c3?usp=sharing.

# Example code

## The following code gives an example of how to extract interlayer features

```python
# All trained models used in our experiments, including those for the super-resolution task, segmentation task, gene expression prediction task,
# survival prediction task, and classification task, have been uploaded to the provided data drive link (MDA_Datasets/data/Trained Models/...).
# Here, we provide an example of the feature extraction process from Tensorflow models for the segmentation task. For PyTorch models, the
# process is demonstrated in example_pytorch_feature_extraction.py.
```

```python
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf

# Path to the pre-trained model
model_path='../data/trained_models/model_seg.h5'
# Load the pre-trained model
model=load_model(model_path)

# Load test data
X_test=np.load('../data/trained_models/X_test_seg.npy')

# Extract output from a specific layer ('conv2d_9') of the model
interlayer_output=model.get_layer('conv2d_9').output
# Create a new model that outputs the interlayer output
inter_model = tf.keras.Model(inputs=model.input, outputs=interlayer_output)

# Initialize an empty list to store outputs
inter_out=[]
# Loop through the test data to extract features
for i in range(len(X_test)):
    test_img=X_test[i]  # Get an individual test image
    test_img=test_img[np.newaxis,:, :]  # Add an extra dimension
    test_img=test_img/255  # Normalize the image
    test_out=inter_model.predict(test_img)  # Predict using the intermediate model
    test_out=np.squeeze(test_out)  # Remove single-dimensional entries
    inter_out.append(test_out)  # Append the output to the list
# Convert list to numpy array
inter_out=np.array(inter_out)  

# Reshape the output for PCA
n1, h1, w1, c1 = inter_out.shape
inter_out = inter_out.reshape(-1, h1*w1*c1)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=400, svd_solver='arpack')
inter_out = pca.fit_transform(inter_out)
print(inter_out.shape)

# Save the PCA-transformed features
np.save('../data/Seg/feature_test.npy',inter_out)

```


## The following code shows the MDA analyses of deep neural network (DNN) features at intermediate layers for five different tasks
```python
# For the tasks below, five datasets analysed in the manuscript will be automatically loaded. 
# However, you can upload your own dataset, and analyze it using MDA
# Our data were saved as .npy file to reduce the data size (normally .csv file needs more disk space). 
# However, .csv or other type of files can also be loaded and analyzed using MDA
```
```python
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
### Superresolution Network

In the superresolution task, we employed the super resolution generative adversarial network (SRGAN) to enhance the resolution of dermoscopic images (ISIC-2019) from 32×32 to 64×64. The selected SRGAN is a well-established deep network for super resolution, which is composed of a generator and a discriminator. In our implementation, the generator contains 4 residual blocks with shortcut connection batch normalization and PReLU and 1 upsampling block; the discriminator contains 7 convolution layers with leaky RuLU. of them stacks eight convolutional layers. Every two convolutional layers are linked together in a feed-forward mode to maximize feature reuse.

### Dataset and feature selection
We adopted ISIC-2019 dataset, which consists of a total of 25,331 dermoscopic images, including 4522 melanoma, 12,875 melanocytic nevus, 3323 basal cell carcinoma, 867 actinic keratosis, 2624 benign keratosis, 239 dermatofibroma, 253 vascular lesion, and 628 squamous cell carcinoma cases.
To visualize the intermediate layers of the SRGAN, we selected features of (a) output of the first residual block, (b) output of the third residual block, (c) output of the fourth residual block, and (d) output of the upsampling block in the generator. In this demo, feature (d) is given as a example.
```python
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
### Visualization and analysis of SRGAN features for super resolution task:

![sr](https://github.com/xinglab-ai/mda/assets/26252653/062efbde-8db9-48fa-952d-ab557a53fd67)

Figure 1. MDA Visualization of SRGAN features for super resolution task after network training. Here, RB1 denotes the first residual block, RB3 denotes the third residual block, RB4 denotes the fourth residual block, and UB denotes the up-sampling block. t-SNE, UMAP and MDA results are shown in (a), (b), (c), respectively for training and testing datasets at different network layers. The colorbar denotes the normalized manifold distance. (d) Pearson correlations between the geodesic distances among feature data points in HD and low dimensional representation from different methods are shown for training and testing data.



## Example 2 - MDA analysis of the DNN features in segmentation task
### Segmentation Network

In the segmentation task, we employed Dense-UNet for automatic brain tumor segmentation from MR images. The Dense-UNet combines the U-net with the dense concatenation to deepen the depth of the network architecture and achieve feature reuse. The network is formed from seven dense blocks (four in encoder and three in decoder), each of them stacks eight convolutional layers. Every two convolutional layers are linked together in a feed-forward mode to maximize feature reuse.

### Dataset and feature selection
Here, we used BraTS 2018 dataset, which provides multimodality 3D MRI images with tumor segmentation labels annotated by physicians. The dataset includes 484 cases in total, which can be divided into 210 high-grade gliomas (HGG) and 75 low-grade gliomas (LGG) cases.
To visualize the intermediate layers of the Dense-UNet, we selected features of (a) the second convolutional layer in the third dense block, (b) the 8th convolutional layer in the fourth dense block, (c) the second convolutional layer in the 6th dense block, and (d) the last convolutional layer before the final output. In this demo, feature (d) is given as a example.

```python
# Load feature data extracted by the Dense-UNet from test images at the last layer before output 
testDataFeatures = np.load('../data/Seg/feature4_test.npy')
# Load data labels (segmented images) corresponding to input test images
Y = np.load('../data/Seg/y_test.npy')
# Reshape the binary images into vectors
Y = Y.reshape(Y.shape[0],-1)
# Load output segmentation prediced by the Dense-UNet
Y_pred = np.load('../data/Seg/y_test_pred_trained.npy')
# Reshape the output binary images into vectors
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)

# Create color map for MDA visualization from the topology of the target manifold   
clusterIdx = discoverManifold(Y, neighborNum)
# Compute the outline of the output manifold
clusterIdx_pred = discoverManifold(Y_pred, neighborNum)
# Use the outline of the output manifold to generate the MDA visualization of the Dense-UNet features
Yreg = mda(testDataFeatures,clusterIdx_pred)   

# Plot the MDA results
plt.figure(1)
plt.scatter(Yreg[:,0],Yreg[:,1],c=clusterIdx.T, cmap='jet', s=5)
plt.xlabel("MDA1")
plt.ylabel("MDA2")
plt.title('MDA visualization of the Dense-UNet features for segmentation task')
```

### Visualization and analysis of Dense-UNet features for image segmentation task before training:


![seg1](https://github.com/xinglab-ai/mda/assets/26252653/6ba719a9-853d-4230-9cb3-95606219344c)

Figure 2. Visualization and analysis of Dense-UNet features for segmentation task before training the network. Here, B3-L2 denotes the 2nd layer of the 3rd dense block, B4-L8 denotes the 8th layer of the 4th dense block, B6-L2 denotes the 2nd layer of the 6th dense block, and B7-L8 denotes the last layer before the final output. t-SNE, UMAP and MDA results are shown in (a), (b), (c), respectively for training and testing datasets at different network layers. The colorbar denotes the normalized manifold distance.

### Visualization and analysis of Dense-UNet features for image segmentation task after training:


![seg2](https://github.com/xinglab-ai/mda/assets/26252653/6e25402c-e674-49fb-8402-ab418f8482f5)

Figure 3. Visualization and analysis of Dense-UNet features for segmentation task after training the network. Here, B3-L2 denotes the 2nd layer of the 3rd dense block, B4-L8 denotes the 8th layer of the 4th dense block, B6-L2 denotes the 2nd layer of the 6th dense block, and B7-L8 denotes the last layer before the final output. t-SNE, UMAP and MDA results are shown in (a), (b), (c), respectively for training and testing datasets at different network layers. The colorbar denotes the normalized manifold distance. (d) Pearson correlations between the geodesic distances among feature data points in HD and low dimensional representation from different methods are shown for training and testing data. 




## Example 3 - MDA analysis of the DNN features in survival prediction task
### Survival Prediction Network

In the survival prediction task, we established an MLP model to predict the survival days of cancer patients from genomics data. The survival prediction network has six fully connected blocks in total, each containing two fully connected layers with the same dimension and one batch normalization layer. The numbers of dimensions are reduced from 2048 to 1024, then 512, 256, 128 and 64. After that, a dropout layer with rate = 0.25 and a fully connected layer with channel = 4 are adopted. Finally, the 1-dimensional output gives the prediction of the patients’ survival days.

### Dataset and feature selection
A public dataset called Cancer Genome Atlas (TCGA) is employed, which provides gene expression (normalized RNA-seq) and patient survival data for 10,956 tumors from 33 cancer types. Before training, data preprocessing was conducted. We first selected the cases where the information “days to death” is applicable, then standardize the survival days to 0-1 by dividing by the maximum value, finally save the corresponding gene expression value of each case and process the data by z-score normalization. After preprocessing, the applicable data includes 2,892 cases, each containing the normalized expression value of 20,531 genes and standardized survival day.
To visualize the intermediate layers of the survival prediction network, we selected features of (a) the second layers of the third fully connected blocks, (b) the second layers of the fourth fully connected blocks, (c) the second layers of the fifth fully connected blocks, and (d) the second layers of the sixth fully connected blocks. In this demo, feature (d) is given as a example.

```python
# Load feature data extracted by the MLP from test data at the 2nd layer of the 6th fully connected block
testDataFeatures = np.load('../data/SP/feature4_test.npy')
# Load data labels (survival days) corresponding to input test genomics data
Y = np.load('../data/SP/y_test.npy')
Y = Y.reshape(Y.shape[0],-1)
# Load output survival days prediced by the MLP
Y_pred = np.load('../data/SP/y_test_pred_trained.npy')
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)

# Create color map for MDA visualization from the topology of the target manifold  
clusterIdx = discoverManifold(Y, neighborNum)
# Compute the outline of the output manifold
clusterIdx_pred = discoverManifold(Y_pred, neighborNum)
# Use the outline of the output manifold to generate the MDA visualization of the MLP features
Yreg = mda(testDataFeatures,clusterIdx_pred)   

# Plot the MDA results
plt.figure(1)
plt.scatter(Yreg[:,0],Yreg[:,1],c=Y, cmap='jet', s=5)
plt.xlabel("MDA1")
plt.ylabel("MDA2")
plt.title('MDA visualization of the MLP features for survival prediction task')
```

### Visualization and analysis of MLP features for survival prediction task:


![sp1](https://github.com/xinglab-ai/mda/assets/26252653/f5c17deb-2b55-4710-9e77-ce8291312984)

Figure 4.Visualization of DNN features before network training by (a) t-SNE, (b) UMAP, and (c) MDA for survival prediction.


![sp2](https://github.com/xinglab-ai/mda/assets/26252653/60432114-e77a-46d6-ad88-86d4cd245434)

Figure 5. Visualization and analysis of MLP features for survival prediction task after training the network. Here, B3-L2 denotes the 2nd layer of the 3rd fully connected block, B4-L2 denotes the 2nd layer of the 4th fully connected block, B5-L2 denotes the 2nd layer of the 5th fully connected block, and B6-L2 denotes the 2nd layer of the 6th fully connected block. t-SNE, UMAP and MDA results are shown in (a), (b), (c), respectively for training and testing datasets at different network layers. The colorbar denotes the normalized manifold distance. (d) Pearson correlations between the geodesic distances among feature data points in HD and low dimensional representation from different methods are shown for training and testing data.




## Example 4 - MDA analysis of the DNN features in gene expression prediction task
### Survival Prediction Network

In the gene expression task, we established a gene expression prediction network, which can effectively estimate the gene expression profiles for different chemical perturbations. The gene expression prediction network first encode the textual string of molecule into one-hot vectors by using the SMILES grammar to parse this string into a parse tree, then uses a grammar variational autoencoder (VAE) to embed the one-hot vectors to continuous latent representation, finally utilizes multilayer perceptron (MLP) to predict the expression profiles of 978 landmark genes.

### Dataset and feature selection
The LINCS L1000 project has collected gene expression profiles for thousands of perturbagens at a variety of time points, doses, and cell lines. Here, we selected Level 3 of the L1000 project, which includes quantile-normalized gene expression profiles of 978 landmark genes, to build up our training and testing set.
To visualize the intermediate layers of the gene expression prediction network, we selected features of (a) the first MLP layer, (b) the second MLP layer, (c) the third MLP layer, and (d) the fourth MLP layer. In this demo, feature (d) is given as a example.

```python
# Load feature data extracted by the MLP from test data at the 4th layer
testDataFeatures = np.load('../data/GP/feature4_test.npy')
# Load data labels (gene expressions) corresponding to input test gene expression data
Y = np.load('../data/GP/y_test.npy')
Y = Y.reshape(Y.shape[0],-1)
# Load prediced gene expressions by the MLP
Y_pred = np.load('../data/GP/y_test_pred_trained.npy')
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)

# Create color map for MDA visualization from the topology of the target manifold  
clusterIdx = discoverManifold(Y, neighborNum)
# Compute the outline of the output manifold
clusterIdx_pred = discoverManifold(Y_pred, neighborNum)
# Use the outline of the output manifold to generate the MDA visualization of the MLP features
Yreg = mda(testDataFeatures,clusterIdx_pred)   

# Plot the MDA results
plt.figure(1)
plt.scatter(Yreg[:,0],Yreg[:,1],c=clusterIdx.T, cmap='jet', s=5)
plt.xlabel("MDA1")
plt.ylabel("MDA2")
plt.title('MDA visualization of the MLP features for gene prediction task')
```

### Visualization and analysis of grammar VAE + MLP features for gene prediction task:


![gp1](https://github.com/xinglab-ai/mda/assets/26252653/ae368ed4-9e36-4a73-b18b-7a3ac9310e34)

Figure 6. Visualization of DNN features before network training by (a) t-SNE, (b) UMAP, and (c) MDA for gene expression prediction.


![gp2](https://github.com/xinglab-ai/mda/assets/26252653/0286b714-914f-4e91-a1c9-89e02bc8376e)

Figure 7. Visualization and analysis of DNN features for gene expression prediction task after training the network. L1-L4 denote the features of the first to fourth MLP layers. t-SNE, UMAP and MDA results are shown in (a), (b), (c), respectively for training and testing datasets at different network layers. The colorbar denotes the normalized manifold distance. (d) Pearson correlations between the geodesic distances among feature data points in HD and low dimensional representation from different methods are shown for training and testing data.




## Example 5 - MDA analysis of the DNN features in classification task
### Classification Network

In the classification task, we utilized the ResNet50 model to classify the lung X-ray images. The ResNet50 consists of 4 substructures, which respectively have 3, 4, 6, 3 residual blocks, containing 3 convolutional layers each. Shortcut connections are also equipped in all residual blocks to solve the degradation problem.

### Dataset and feature selection
The COVID-19 radiography dataset contains 21,165 X-ray images in total, including 3616 COVID-19 positive cases along with 10,192 normal, 6012 lung opacity (non-COVID lung infection), and 1345 viral pneumonia cases.
To visualize the intermediate layers of the ResNet50, we selected features of (a) output of the 4th residual block’s last convolutional layer in substructure2, (b) output of the 2nd residual block’s last convolutional layer in substructure3, (c) output of the 6th residual block’s last convolutional layer in substructure3, and (d) output of the 3rd residual block’s last convolutional layer in substructure4. In this demo, feature (d) is given as a example.

```python
# Load feature data extracted by the ResNet50 from test x-ray images at the 3rd residual block's last convolutional
# layer in substructure 4.
testDataFeatures = np.load('../data/CL/feature4_test.npy')
# Load data labels (lung diseases including COVID) corresponding to input test lung x-ray images
Y = np.load('../data/CL/y_test.npy')
Y = Y.reshape(Y.shape[0],-1)
# Load predicted labels by the ResNet50
Y_pred = np.load('../data/CL/y_test_pred_trained.npy')
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)

# Compute the outline of the output manifold
clusterIdx_pred = discoverManifold(Y_pred, neighborNum)
# Use the outline of the output manifold to generate the MDA visualization of the ResNet50 features
Yreg = mda(testDataFeatures,clusterIdx_pred)   

# Plot the MDA results
plt.figure(1)
plt.scatter(Yreg[:,0],Yreg[:,1],c=Y, cmap='jet', s=5)
plt.xlabel("MDA1")
plt.ylabel("MDA2")
plt.title('MDA visualization of the ResNet50 features for classification task')
```

### Visualization and analysis of ResNet50 features for classification task:

![cl1](https://github.com/xinglab-ai/mda/assets/26252653/e16fec6c-1c47-4a85-bcb4-b06c933f9c12)

Figure 8.Visualization of DNN features before network training by (a) t-SNE, (b) UMAP, and (c) MDA for COVID-19 data classification.


![cl2](https://github.com/xinglab-ai/mda/assets/26252653/ee52b56e-ed4e-4c0b-8d63-69e0658e70af)

Figure 9.Investigation of the feature space of ResNet50 network applied on a public COVID-19 dataset for classification into four categories. (a, b, c) t-SNE, UMAP, and MDA visualizations of the feature spaces at four different layers before/after training. Here, S2-B4-L3 denotes the 4th residual block’s last convolutional layer in substructure 2, S3-B2-L3 denotes the 2nd residual block’s last convolutional layer in substructure 3, S3-B6-L3 denotes the 6th residual block’s last convolutional layer in substructure 3, and S4-B3-L3 denotes the 3rd residual block’s last convolutional layer in substructure 4. Before training, the data points are randomly distributed in MDA visualizations. However, after the training, the feature space becomes well clustered in MDA visualizations, especially in deeper layers. t-SNE and UMAP fail to show any information about the training status of the network. (d) k-nearest neighbor classification accuracy of the low dimensional representations from different techniques.

