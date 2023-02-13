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
from sklearn.preprocessing import normalize

import graph_nets
from graph_nets.graphs import GraphsTuple
import graph_attribution as gatt

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

    odds = np.exp(coeff2)

    Bounded_coeff = 2*(normalize(coeff[:,np.newaxis], axis=0).ravel()+1)
    
    return coeff, odds, top_features, Bounded_coeff, simpler_model 
  
def perturb_graph_node(input_graph, perturbation, Remove_Zero_Degree_Nodes=False):
    """
    This function modifies an input graph by removing nodes based on a specified perturbation.

    Parameters:
    - input_graph (networkx.Graph): The input graph to be perturbed.
    - perturbation (List[List[int]]): A 2D list of 0/1 values indicating whether each node in the input graph should be removed (1) or not (0).
    - Remove_Zero_Degree_Nodes (bool): Indicates whether nodes with zero degree should be removed (True) or not (False).

    Returns:
    - perturbed_node_graph_reordered (networkx.Graph): The perturbed graph with reordered node labels.

    Raises:
    - ValueError: If the length of `perturbation` does not match the number of nodes in `input_graph`.
    """

    if len(perturbation[0]) != len(input_graph.nodes):
        raise ValueError("The length of `perturbation` must match the number of nodes in `input_graph`.")
    
    perturbed_node_graph = input_graph.copy()
    ebun = []
    for ii, nd in enumerate(perturbed_node_graph.nodes):
        if perturbation[0][ii] == 1:
            ebun.append(nd)
    perturbed_node_graph.remove_nodes_from(ebun)

    if Remove_Zero_Degree_Nodes:
        Zdegree_nodes = []
        for n, d in perturbed_node_graph.degree():
            if d == 0:
                Zdegree_nodes.append(n)

        perturbed_node_graph.remove_nodes_from(Zdegree_nodes)

    perturbed_node_graph_reordered = nx.convert_node_labels_to_integers(perturbed_node_graph, first_label=0)

    return perturbed_node_graph_reordered
  
def perturb_graph_edge(input_graph, perturbation):
    """
    This function modifies an input graph by removing edges based on a specified perturbation.

    Parameters:
    - input_graph (networkx.Graph): The input graph to be perturbed.
    - perturbation (List[int]): A list of 0/1 values indicating whether each edge in the input graph should be removed (1) or not (0).

    Returns:
    - perturbed_graph (networkx.Graph): The perturbed graph.

    Raises:
    - ValueError: If the length of `perturbation` does not match the number of edges in `input_graph`.
    """

    if len(perturbation) != len(input_graph.edges):
        raise ValueError("The length of `perturbation` must match the number of edges in `input_graph`.")
    
    perturbed_graph = input_graph.copy()
    ebun = []
    for ii, ed in enumerate(perturbed_graph.edges):
        if perturbation[ii] == 1:
            ebun.append(ed)
    perturbed_graph.remove_edges_from(ebun)
    
    return perturbed_graph

  
 def explain_graph_nodes(input_graph, model, num_perturbations=5000, kernel_width=0.25, num_top_features=10, epsilon=1, remove_zero_degree_nodes=False):
    """
    This function explains the contribution of each node to the prediction of a given graph-based model. 
    It uses a white-box model interpretation method XWhy to calculate the contribution of each node by 
    perturbing the input graph and measuring the change in the model's prediction.

    Parameters:
    input_graph (nx.Graph) : A networkx graph representing the input graph.
    model (sklearn.BaseEstimator) : The graph-based model to be explained.
    num_perturbations (int) : The number of perturbations to be applied to the input graph (default is 5000).
    kernel_width (float) : The width of the kernel function (default is 0.25).
    num_top_features (int) : The number of top features to be returned (default is 10).
    epsilon (float) : The hyperparameter for the kernel function (default is 1).
    remove_zero_degree_nodes (bool) : A flag indicating whether to remove nodes with zero degree after perturbation (default is False).

    Returns:
    coeff (np.ndarray) : The coefficients of the linear regression model that explains the relationship between the node perturbations and the change in prediction.
    odds (np.ndarray) : The odds ratio of the linear regression model.
    top_features (np.ndarray) : The indices of the top `num_top_features` features based on the absolute values of the coefficients.
    bounded_coeff (np.ndarray) : The normalized coefficients of the linear regression model.
    simpler_model (LinearRegression) : The linear regression model used to explain the relationship between the node perturbations and the change in prediction.
    
    Raises:
    ValueError : If the number of perturbations is not a positive integer.
    ValueError : If the kernel width is not positive.
    ValueError : If the number of top features is not a positive integer.
    ValueError : If the epsilon is not positive.
    """
    # Validate the input arguments
    if not isinstance(num_perturbations, int) or num_perturbations <= 0:
        raise ValueError("The number of perturbations must be a positive integer.")
    if not isinstance(kernel_width, (int, float)) or kernel_width <= 0:
        raise ValueError("The kernel width must be a positive number.")
    if not isinstance(num_top_features, int) or num_top_features <= 0:
        raise ValueError("The number of top features must be a positive integer.")
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("The epsilon must be a positive number.")  
    
    num_uniqe_nodes = len(X_input_graph.nodes)

    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_uniqe_nodes))
        
    predictions = []
    WD_dist = []
    for pert in perturbations:
        p_graph = perturb_graph_node(X_input_graph, pert)
        if len(p_graph.nodes) != 0:
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

    odds = np.exp(coeff2)
    
    from sklearn.preprocessing import normalize

    Bounded_coeff = normalize(coeff[:,np.newaxis], axis=0).ravel()
    
    return coeff, odds, top_features, Bounded_coeff, simpler_model 
