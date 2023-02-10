import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.linalg as la
import networkx as nx
import random, time, math
from collections import Counter

import fun as f
from Graph import Graph
from Watts_Strogatz import watts_strogatz_graph
from Erdos_Renyi import erdos_renyi_graph

from sklearn.linear_model import LinearRegression

def Wasserstein_Dist(cdfX, cdfY):
  
    Res = 0
    power = 1
    n = len(cdfX)

    for ii in range(0, n-2):
        height = abs(cdfX[ii]-cdfY[ii])
        width = cdfX[ii+1] - cdfX[ii]
        Res = Res + (height ** power) * width 
 
    return Res


def r_eigenv(G_i, G_j):
    #Eigen-decomposition of G_j
    A_Gi = (nx.adjacency_matrix(G_i)).todense()
    D_i = np.diag(np.asarray(sum(A_Gi))[0])
    eigenvalues_Gi, eigenvectors_Gi = la.eig(D_i - A_Gi)
    r_eigenv_Gi = sorted(zip(eigenvalues_Gi.real, eigenvectors_Gi.T), key=lambda x: x[0])

    #Eigen-decomposition of G_j
    A_Gj = (nx.adjacency_matrix(G_j)).todense()
    D_j = np.diag(np.asarray(sum(A_Gj))[0])
    eigenvalues_Gj, eigenvectors_Gj = la.eig(D_j - A_Gj)
    r_eigenv_Gj = sorted(zip(eigenvalues_Gj.real, eigenvectors_Gj.T), key=lambda x: x[0])
    
    r = 4
    signs =[-1,1]
    temp = []
    for  sign_s in signs:
        for sign_l in signs:
            vri = sorted(f.normalize_eigenv(sign_s * r_eigenv_Gi[r][1]))
            vrj = sorted(f.normalize_eigenv(sign_l * r_eigenv_Gj[r][1]))
            cdf_dist = f.cdf_dist(vri, vrj)
            temp.append(cdf_dist)
    
    #Compute empirical CDF
    step = 0.005
    x=np.arange(0, 1, step)
    cdf_grid_Gip = f.cdf(len(r_eigenv_Gi[r][1]),x,
                   f.normalize_eigenv(sorted(r_eigenv_Gi[r][1], key=lambda x: x)))
    cdf_grid_Gin = f.cdf(len(r_eigenv_Gi[r][1]),x,
                   f.normalize_eigenv(sorted(-r_eigenv_Gi[r][1], key=lambda x: x)))

    cdf_grid_Gjp = f.cdf(len(r_eigenv_Gj[r][1]),x,
                   f.normalize_eigenv(sorted(r_eigenv_Gj[r][1], key=lambda x: x)))
    cdf_grid_Gjn = f.cdf(len(r_eigenv_Gj[r][1]),x,
                   f.normalize_eigenv(sorted(-r_eigenv_Gj[r][1], key=lambda x: x)))
    
    WD1 = Wasserstein_Dist(cdf_grid_Gip, cdf_grid_Gjp)
    WD2 = Wasserstein_Dist(cdf_grid_Gip, cdf_grid_Gjn)
    WD3 = Wasserstein_Dist(cdf_grid_Gin, cdf_grid_Gjp)
    WD4 = Wasserstein_Dist(cdf_grid_Gin, cdf_grid_Gjn)

    WD = [WD1, WD2, WD3, WD4]
    
    return max(temp), max(WD)

def xwhy_graph_edges(X_input_graph, model, num_perturb = 50, kernel_width = 0.25, num_top_features = 10, eps=1):    
    #num_perturb = 5000
    
    num_uniqe_words = len(X_input_graph.edges)

    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_uniqe_words))

    def perturb_graph_edge(input_graph, perturbation):
        perturbed_graph = input_graph.copy()
        ebun = []
        for ii, ed in enumerate(perturbed_graph.edges):
            if perturbation[ii] == 0:
                ebun.append(ed)
        perturbed_graph.remove_edges_from(ebun)
        return perturbed_graph
        
    predictions = []
    WD_dist = []
    for pert in perturbations:
        p_graph = perturb_graph_edge(X_input_graph, pert)
        pred = model.predict(graph_nets.utils_np.networkxs_to_graphs_tuple([p_graph]))
        predictions.append(pred)
        Sscore, WD_score = r_eigenv(X_input_graph, p_graph)
        WD_dist.append(WD_score)

    predictions = np.array(predictions)   
    WD_dist = np.array(WD_dist) 
    
    weights = np.sqrt(np.exp(-((eps*WD_dist)**2)/kernel_width**2)) #Kernel function
                
    class_to_explain = 0
    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions, sample_weight=weights)
    coeff = simpler_model.coef_[0]
    
    top_features = np.argsort(abs(coeff))[-num_top_features:] 
    
    coeff2 = simpler_model.coef_
    
    # https://stackoverflow.com/questions/39626401/how-to-get-odds-ratios-and-other-related-features-with-scikit-learn
    odds = np.exp(coeff2)
    
    from sklearn.preprocessing import normalize

    Bounded_coeff = 2*(normalize(coeff[:,np.newaxis], axis=0).ravel()+1)
    
    return coeff, odds, top_features, Bounded_coeff, simpler_model 
