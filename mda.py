import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import scipy
import scipy.io as sio
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import lil_matrix

import sklearn
from sklearn.metrics import pairwise_distances_chunked
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from umap.parametric_umap import ParametricUMAP
import umap

import numpy as np

class paramsMDA:
    """
    paramsMDA operator  sets the parameters for MDA analysis
    
    """
    # set the hyperparameters of gamma prior used for projection matrix
    alpha_phi = 1
    beta_phi = 1

    # set the hyperparameters of gamma prior used for bias parameters
    alpha_lambda = 1
    beta_lambda = 1

    # set the hyperparameters of gamma prior used for weight parameters
    alpha_psi = 1
    beta_psi = 1

    ### IMPORTANT ###
    # For gamma priors, you can experiment with three different (alpha, beta) values
    # (1, 1) => default priors
    # (1e-10, 1e+10) => good for obtaining sparsity
    # (1e-10, 1e-10) => good for small sample size problems

    # set the number of iterations
    iteration = 200

    # set the subspace dimensionality
    R = 16

    # determine whether you want to use automatic relevance determination priors for projection matrix (ard or entrywise)
    prior_phi = 'entrywise'
    
    # determine whether you want to calculate and store the lower bound values
    progress = 0

    # set the sample size used to calculate the expectation of truncated normals
    sample = 200

    # set the seed for random number generator used to initalize random variables
    seed = 1606

    # set the standard deviation of projected instances
    sigma_z = 0.1
    
def selectNdimNSORTidx(data, N):
    # selects N number of highly variable features from a data matrix 
    gvar = np.var(data, axis=0)
    var500idx = (-gvar).argsort()[:N]
    varNsamples = data[:,var500idx]
    return varNsamples, var500idx

def find_nnCorr(X, k=12):
    """
     Compute neighborhood matrix 
    
    Parameters:
        
        X: High dimensional data in tabular
        format. The rows denote the observations and columns denote the features.
        
        k: int, optional, default: 12.
        number of neighbors in the data
    """
    X_std = X - np.mean(X,axis=1, keepdims=True) # subtract the mean from the data
    X_norm = np.linalg.norm(X_std, axis=1, keepdims=True) # Normalize the data to make each 
    # feature of unit variance
    DD = np.sqrt(1 - X_std @ X_std.T / (X_norm @ X_norm.T) + np.finfo(np.float32).eps)# Compute
    # the Euclidean distance
    
    D = lil_matrix(DD.shape)
    x_idx = np.arange(DD.shape[0]).repeat(k+1)
    y_idx = np.argpartition(DD, kth=k+1, axis=-1)[:,:k+1].flatten()
    
    # Select only neighborhood distance info
    y_idx = np.delete(y_idx, y_idx==x_idx)
    x_idx = np.arange(DD.shape[0]).repeat(k)
    
    D[x_idx, y_idx] = DD[x_idx,y_idx]
    D[y_idx, x_idx] = DD[y_idx,x_idx]
    
    return D

def discoverManifold(GT, neighborNum=12):
    """
     Discover the manifold of deep learning feature space
    
    Parameters:
        
        GT: High dimensional data in tabular
        format. The rows denote the observations and columns denote the features.
        
        neighborNum: int, optional, default: 12.
        number of neighbors in the data
        
    """
    sz = GT.shape
    
   # if sz[1] > 1:
        #print('Constructing neighborhood graph...')
    # Compute the distance of the data points over the manifold
    D = pdist(GT, metric='euclidean')
    D = squareform(D)
    # Select the distance from the first data point
    geoDistance = D[0,:]
    # Find one endpoint of the manifold
    cMax, ik = np.max(geoDistance), np.argmax(geoDistance)
    corrTrainMax = D[ik,:]
    
    # Discretize the distance vector to obtain the outline of the manifold
    hist, bins = np.histogram(corrTrainMax,bins='auto')
    clusterIdx = np.digitize(corrTrainMax, bins).reshape(sz[0],1)
    
    return clusterIdx
   # else:
     #   return GT        

def bsdr(X, y, parameters):
    """
     Bayesian supervised dimensionality reduction
    
    Parameters:
        
        X: High dimensional data in tabular format. The rows denote the observations and columns denote the features.        
   
        y: int vector
        labels of the data
        

        parameters: parameters set by paramsMDA() class
    """
    np.random.seed(parameters.seed)
    
    D, N = X.shape
    K = np.max(y)
    R = parameters.R
    
    sigma_z = parameters.sigma_z
    
    log2pi = np.log(2 * np.pi)
    
    # If we want to estimate the best reduced dimension using 'ARD' method
    if parameters.prior_phi == 'ard':
        phi_alpha = (parameters.alpha_phi + 0.5 * D) * np.ones((R, 1))
        phi_beta = parameters.beta_phi * np.ones((R, 1))
    else:
        Phi_alpha = (parameters.alpha_phi + 0.5) * np.ones((D, R))
        Phi_beta = parameters.beta_phi * np.ones((D, R))

    # Initialize the variables
    # For Gaussian-distributed Q, initialize the mean and variance
    Q_mu = np.random.randn(D, R)
    Q_sigma = np.repeat(np.eye(D).reshape((D,D,1)),R,axis=-1)
    # For Gaussian-distributed Z, initialize the mean and variance
    Z_mu = np.random.randn(R, N)
    Z_sigma = np.eye(R)
    # For Gamma-distributed prior lambda, initialize the alpha and beta
    lambda_alpha = (parameters.alpha_lambda + 0.5) * np.ones((K, 1))
    lambda_beta = parameters.beta_lambda * np.ones((K, 1))
    # For Gamma-distributed prior Psi, initialize the alpha and beta
    Psi_alpha = (parameters.alpha_psi + 0.5) * np.ones((R, K))
    Psi_beta = parameters.beta_psi * np.ones((R, K))
    # For Gaussian-distributed b and W, initialize the mean and variance
    bW_mu = np.random.randn(R + 1, K)
    bW_sigma = np.repeat(np.eye(R + 1).reshape((R + 1,R + 1,1)),K,axis=-1)
    # For truncated Gaussian-distributed T, initialize the mean and variance
    T_mu = np.zeros((K, N))
    T_sigma = np.eye(K)
    for i in range(N):
        while 1:
            T_mu[:, i] = np.random.randn(K)
            if T_mu[y[i]-1, i] == np.max(T_mu[:, i]):
                break
    normalization = np.zeros((N, 1))
    
    XXT = X @ X.T
    phi_indices = np.repeat(np.eye(D).astype(bool).reshape((D,D,1)),R,axis=-1)
    psi_indices = np.repeat(np.block([[np.zeros((1, R + 1))],
                                      [np.zeros((R, 1)), np.eye(R)]]).astype(bool).reshape((R + 1,R + 1,1)),
                            K, axis=-1)
    
    # Estimation progress
    if parameters.progress == 1:
        bounds = np.zeros((parameters.iteration, 1))
    
    for iter_ in range(parameters.iteration):
        #if iter_ % 1 == 0:
        #    print('.', end="")
        #if iter_ % 10 == 0:
        #    print(' %5d\n'%iter_)
            
        if parameters.prior_phi == 'ard':
            for s in range(R):
            # update priors (eq. 15)
                phi_beta[s] = 1 / (1 / parameters.beta_phi + 0.5 * (Q_mu[:, s].T @ Q_mu[:, s] + np.sum(np.diag(Q_sigma[:, :, s]))))
            for s in range(R):
                # update variance of projection matrix Q (eq. 16)
                Q_sigma[:, :, s],_,_,_ = scipy.linalg.lstsq((phi_alpha[s] * phi_beta[s] * np.eye(D) + XXT / (sigma_z**2)), np.eye(D), lapack_driver='gelsy')
                # update mean of projection matrix Q (eq. 16)
                Q_mu[:, s] = Q_sigma[:, :, s] @ (X @ Z_mu[s, :].T / (sigma_z**2))
        else:
            # update priors (eq. 15)
            Phi_beta = 1 / (1 / parameters.beta_phi + 0.5 * (Q_mu**2 + np.reshape(Q_sigma[phi_indices], (D,R))))
            for s in range(R):
                # update variance of projection matrix Q (eq. 16)
                Q_sigma[:, :, s],_,_,_ = scipy.linalg.lstsq((np.diag(Phi_alpha[:, s] * Phi_beta[:, s]) + XXT / (sigma_z**2)), np.eye(D), lapack_driver='gelsy')
                # update mean of projection matrix Q (eq. 16)
                Q_mu[:, s] = Q_sigma[:, :, s] @ (X @ Z_mu[s, :].T / (sigma_z**2))

        # update variance of projected variable Z (eq. 17)        
        Z_sigma,_,_,_ = scipy.linalg.lstsq((np.eye(R) / (sigma_z**2) + bW_mu[1:R+1, :] @ bW_mu[1:R+1, :].T + np.sum(bW_sigma[1:R+1, 1:R+1, :], axis=-1)),
                                  np.eye(R), lapack_driver='gelsy')
        # update mean of projected variable Z (eq. 17)                          
        Z_mu = Z_sigma @ (Q_mu.T @ X / (sigma_z**2) + bW_mu[1:, :] @ T_mu - \
                          np.repeat((bW_mu[1:R+1, :] @ bW_mu[0, :].T + np.sum(bW_sigma[0, 1:R+1, :], axis=-1).T).reshape((R,1)), N, axis=-1))
        # update lambda (eq. 18)
        lambda_beta = 1 / (1 / parameters.beta_lambda + 0.5 * (bW_mu[0, :].T**2 + bW_sigma[0, 0, :])).reshape((K, 1))
        # update Psi (eq. 19)
        Psi_beta = 1 / (1 / parameters.beta_psi + 0.5 * (bW_mu[1:R+1, :]**2 + np.reshape(bW_sigma[psi_indices], (R, K))))

        # update b and W (eq. 20)
        for c in range(K):
            # variance update
            bW_sigma[:, :, c],_,_,_ = scipy.linalg.lstsq(np.block([[lambda_alpha[c, 0] * lambda_beta[c, 0] + N, np.sum(Z_mu, axis=-1, keepdims=True).T],
                                 [np.sum(Z_mu, axis=-1, keepdims=True), 
                                  np.diag(Psi_alpha[:, c] * Psi_beta[:, c]) + Z_mu @ Z_mu.T + N * Z_sigma]]),
                                                np.eye(R + 1), lapack_driver='gelsy')
            # mean update
            bW_mu[:, c] = bW_sigma[:, :, c] @ np.block([[np.ones((1, N))], [Z_mu]]) @ T_mu[c, :].T

        # Updtae score variable T (eq. 21)    
        T_mu = bW_mu[1:R+1, :].T @ Z_mu + np.repeat(bW_mu[0, :].reshape((K,1)), N, axis=-1)
        for c in range(K):
            pos = np.where((y-1).flatten() == c)[0]
            normalization[pos, 0], T_mu[:, pos] = truncated_normal_mean(T_mu[:, pos], c, parameters.sample, 0);
        
        # Calculation of lower bound for each of the estimation
        lb = 0
        if parameters.prior_phi == 'ard':
            lb = lb + np.sum((parameters.alpha_phi - 1) * (scipy.special.psi(phi_alpha) + np.log(phi_beta)) - \
                             phi_alpha * phi_beta / parameters.beta_phi - scipy.special.gammaln(parameters.alpha_phi) -\
                             parameters.alpha_phi * np.log(parameters.beta_phi))
            for s in range(R):
                lb = lb - 0.5 * Q_mu[:, s].T @ (phi_alpha[s] * phi_beta[s] * np.eye(D)) @ Q.mu[:, s] -\
                     0.5 * (D * log2pi - D * (scipy.special.psi(phi_alpha[s]) + np.log(phi_beta[s])))
        else:
            lb = lb + np.sum((parameters.alpha_phi - 1) * (scipy.special.psi(Phi_alpha) + np.log(Phi_beta)) -\
                                    Phi_alpha * Phi_beta / parameters.beta_phi - scipy.special.gammaln(parameters.alpha_phi) -\
                                    parameters.alpha_phi * np.log(parameters.beta_phi))
            for s in range(R):
                lb = lb - 0.5 * Q_mu[:, s].T @ np.diag(Phi_alpha[:, s] * Phi_beta[:, s]) @ Q_mu[:, s] -\
                     0.5 * (D * log2pi - np.sum(scipy.special.psi(Phi_alpha[:, s]) + np.log(Phi_beta[:, s])))
        # p(Z | Q, X)
        lb = lb - 0.5 * (sigma_z**-2) * (np.sum(Z_mu * Z_mu) + N * np.sum(np.diag(Z_sigma))) +\
             (sigma_z**-2) * np.sum((Q_mu.T @ X) * Z_mu) -\
             0.5 * (sigma_z**-2) * np.sum(X * ((Q_mu @ Q_mu.T + np.sum(Q_sigma, axis=-1)) @ X)) -\
             0.5 * N * D * (log2pi + 2 * np.log(sigma_z))
        # p(lambda)
        lb = lb + np.sum((parameters.alpha_lambda - 1) * (scipy.special.psi(lambda_alpha) + np.log(lambda_beta)) -\
                         lambda_alpha * lambda_beta / parameters.beta_lambda - scipy.special.gammaln(parameters.alpha_lambda) -\
                         parameters.alpha_lambda * np.log(parameters.beta_lambda))       
        # p(b | lambda)
        lb = lb - 0.5 * bW_mu[0, :] @ np.diag(lambda_alpha[:, 0] * lambda_beta[:, 0]) @ bW_mu[0, :].T -\
             0.5 * (K * log2pi - np.sum(scipy.special.psi(lambda_alpha[:, 0]) + np.log(lambda_beta[:, 0])))
        # p(Psi)
        lb = lb + np.sum((parameters.alpha_psi - 1) * (scipy.special.psi(Psi_alpha) + np.log(Psi_beta)) -\
                         Psi_alpha * Psi_beta / parameters.beta_psi - scipy.special.gammaln(parameters.alpha_psi) -\
                         parameters.alpha_psi * np.log(parameters.beta_psi))
        # p(W | Psi)
        for c in range(K):
            lb = lb - 0.5 * bW_mu[1:R+1, c].T @ np.diag(Psi_alpha[:, c] * Psi_beta[:, c]) @ bW_mu[1:R+1, c] -\
                 0.5 * (R * log2pi - np.sum(scipy.special.psi(Psi_alpha[:, c]) + np.log(Psi_beta[:, c])))
        
        WWT_mu = bW_mu[1:R+1, :] @ bW_mu[1:R+1, :].T + np.sum(bW_sigma[1:R+1, 1:R+1, :], axis=-1)
        lb = lb - 0.5 * (np.sum(T_mu * T_mu) + N * K) + np.sum(bW_mu[0, :] @ T_mu) + np.sum(Z_mu * (bW_mu[1:R+1, :] @ T_mu)) -\
             0.5 * (N * np.trace(WWT_mu @ Z_sigma) + np.sum(Z_mu * (WWT_mu @ Z_mu))) -\
             0.5 * N * (bW_mu[0, :] @ bW_mu[0, :].T + np.sum(bW_sigma[0, 0, :])) -\
             np.sum(Z_mu.T @ (bW_mu[1:R+1, :] @ bW_mu[0, :].T + np.sum(bW_sigma[1:R+1, 0, :], axis=-1))) - 0.5 * N * K * log2pi

        if parameters.prior_phi == 'ard':
            lb = lb + np.sum(phi_alpha + np.log(phi_beta) + scipy.special.gammaln(phi_alpha) +\
                             (1 - phi_alpha) * scipy.special.psi(phi_alpha))
        else:
            lb = lb + np.sum(Phi_alpha + np.log(Phi_beta) + scipy.special.gammaln(Phi_alpha) +\
                             (1 - Phi_alpha) * scipy.special.psi(Phi_alpha))
        
        # q(Q)
        for s in range(R):
            lb = lb + 0.5 * (D * (log2pi + 1) + logdet(Q_sigma[:, :, s]))
        # q(Z)
        lb = lb + 0.5 * N * (R * (log2pi + 1) + logdet(Z_sigma))
        # q(lambda)
        lb = lb + np.sum(lambda_alpha + np.log(lambda_beta) + scipy.special.gammaln(lambda_alpha) +\
                         (1 - lambda_alpha) * scipy.special.psi(lambda_alpha))
        # q(Psi)
        lb = lb + np.sum(Psi_alpha + np.log(Psi_beta) + scipy.special.gammaln(Psi_alpha) +\
                         (1 - Psi_alpha) * scipy.special.psi(Psi_alpha))
        # q(b, W)
        for c in range(K):
            lb = lb + 0.5 * ((R + 1) * (log2pi + 1) + logdet(bW_sigma[:, :, c]))
        
        # q(T)
        lb = lb + 0.5 * N * K * (log2pi + 1) + np.sum(np.log(normalization))
        
        if parameters.progress == 1:
            bounds[iter_] = lb
    state = {}
    if parameters.prior_phi == 'ard':
        phi = {'alpha':phi_alpha, 'beta':phi_beta}
        state['phi'] = phi
    else:
        Phi = {'alpha':Phi_alpha, 'beta':Phi_beta}
        state['Phi'] = Phi
    Q = {'mu':Q_mu, 'sigma':Q_sigma}
    Z = {'mu':Z_mu, 'sigma':Z_sigma}
    lmbd = {'alpha':lambda_alpha, 'beta':lambda_beta}
    Psi = {'alpha':Psi_alpha, 'beta':Psi_beta}
    bW = {'mu':bW_mu, 'sigma':bW_sigma}
    state['Q'] = Q
    state['lambda'] = lmbd
    state['Psi'] = Psi
    state['bW'] = bW
    if parameters.progress == 1:
        state['bounds'] = bounds
    state['parameters'] = parameters
    
    return state

def logdet(Sigma):
    # logarithm of determinant
    U = np.linalg.cholesky(Sigma)
    return 2 * np.sum(np.log(np.diag(U)))
        
def truncated_normal_mean(centers, active, S, tube):
    """
    Compute the mean of truncated normal distribution
    
    Parameters:
        
        centers: Mean values of the untrauncated distribution
        
        active: int vector. active label group for which the computation is being performed
        S: sample size used to calculate the expectation of truncated normals
        tube:  0

    returns the mean of truncated normal distribution
    """
    K,N = centers.shape[0:2]
    
    # Compute the difference from mean
    diff = np.repeat(centers[active, :].reshape((1,N,)), K, axis=0) - centers - tube
    u = np.random.randn(1, N, S)
    q = scipy.stats.norm().cdf(np.repeat(u, K, axis=0) + np.repeat(diff.reshape(K,N,1), S, axis=-1))
    pr = np.repeat(np.prod(q, axis=0, keepdims=True), K, axis=0)
    pr = pr / q
    ind = np.block([np.arange(0,active), np.arange(active+1,K)])
    pr[ind, :, :] = pr[ind, :, :] / np.repeat(q[active, :, :].reshape((1,N,S)), K - 1, axis=0)
    pr[ind, :, :] = pr[ind, :, :] * scipy.stats.norm().pdf(np.repeat(u, K - 1, axis=0) + np.repeat(diff[ind, :].reshape((K-1,N,1)), S, axis=-1))
    # normalize data
    normalization = np.mean(pr[active, :, :], axis=-1).reshape((1,-1))
    # compute expectation
    expectation = np.zeros((K, N))
    expectation[ind, :] = centers[ind, :] - np.repeat(1 / normalization, K - 1, axis=0) * np.reshape(np.mean(pr[ind, :, :], axis=-1), (K - 1, N))
    expectation[active, :] = centers[active, :] + np.sum(centers[ind, :] - expectation[ind, :], axis=0)
    
    return normalization, expectation
        
def mda(data,clusterIdx):
    """
    Manifold discovery analysis
    
    Parameters:
        
        data: High dimensional deep neural network feature data in tabular
        format. The rows are the data points and columns are the feaures.
        
        clusterIdx: int vector.
        pseudo labels of the data computed using discover_manifold function

    returns low dimensional representation    
    """

    # Use SVD to find components with non zero eigen values. This step is optional and used for 
    # reducing computational load 
    lambds = np.linalg.svd(data, full_matrices=False, compute_uv=False)
    data = data[:, lambds!=0]
    
    # prepare data and pseudo labels
    Xtrain = np.copy(data.T)
    ytrain = clusterIdx.reshape((Xtrain.shape[1],1))
    
    # Make NaN values to zero
    Xtrain = np.nan_to_num(Xtrain)
    
    # Set the parameters of MDA
    parameters = paramsMDA()
    
    # Run Bayesian dimensionality reduction
    state = bsdr(Xtrain, ytrain+1, parameters)  
    # Estimated expectation of projection matrix  
    vec = state['Q']['mu']
    # Compute projection of the data
    Ypro = data @ vec
    
    # Apply deep learning based visualization technique to obtain MDA components
    reducer = ParametricUMAP(parametric_embedding=False)
    Yreg = reducer.fit_transform(Ypro)

    return Yreg