import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import SafeML 

def WasserstainLIME2(X_input, model, num_perturb = 500, L_num_perturb = 100, kernel_width2 = 0.75, epsilon = 0.1):
    '''
    WasserstainLIME is a statistical version of LIME (local interpretable model-agnostic explanations) 
    in which instead of Euclidean distance, the ECDF-based distance is used.
    
    X_input: should be a numpy array that represents one point in a n-dimensional space.
    
    num_perturb: Is the number of perturbations that the algorithm uses.
    
    L_num_perturb: Is the number of perturbations in the local areas that the algorithm uses.
    
    kernel_width: Is the Kernel Width. When the decision space is very dynamic, the kernel width should be low like 0.2, 
    otherwise kernel with around 0.75 would be ideal.
    
    model: It is the trained model that can be for a classification or regression. 
    
    epsilon: It is used to normalize the WD values.
    
    '''
    
    X_input = (X_input - np.mean(X_input,axis=0)) / np.std(X_input,axis=0) #Standarization of data

    X_lime = np.random.normal(0,1,size=(num_perturb,X_input.shape[0]))
    
    Xi2 = np.zeros((L_num_perturb,X_input.shape[0]))
    
    for jj in range(X_input.shape[0]):
        Xi2[:,jj] = X_input[jj] + np.random.normal(0,0.05,L_num_perturb)

    y_lime2  = np.zeros((num_perturb,1))
    WD       = np.zeros((num_perturb,1))
    weights2 = np.zeros((num_perturb,1))
    
    for ind, ii in enumerate(X_lime):
        
        df2 = pd.DataFrame()
        
        for jj in range(X_input.shape[0]):
            temp1 = ii[jj] + np.random.normal(0,0.3,L_num_perturb)
            df2[len(df2.columns)] = temp1

        temp3 = model.predict(df2.to_numpy())

        y_lime2[ind] = np.mean(temp3)  # For classification: np.argmax(np.bincount(temp3))
        
        WD1 = np.zeros((X_input.shape[0],1))
        
        df2 = df2.to_numpy()
        
        for kk in range(X_input.shape[0]):
            #print( df2.shape)
            WD1[kk] = SafeML.Wasserstein_Dist(Xi2[:,kk], df2[:,kk])
        
        #print(WD1)
        #print(ind)
        WD[ind] = sum(WD1)
        #print(WD)
    
        weights2[ind] = np.sqrt(np.exp(-((epsilon*WD[ind])**2)/(kernel_width2**2))) 
        #print(weights2[ind])
        
        del df2
    
    weights2 = weights2.flatten()
    #print(weights2)
    
    simpler_model2 = LinearRegression() 
    simpler_model2.fit(X_lime, y_lime2, sample_weight=weights2)
    y_linmodel2 = simpler_model2.predict(X_lime)
    y_linmodel2 = y_linmodel2 < 0.5 #Conver to binary class
    y_linmodel2 = y_linmodel2.flatten()
    
    return X_lime, y_lime2, weights2, y_linmodel2, simpler_model2.coef_.flatten()

def WasserstainLIME(X_input, model, num_perturb = 500, kernel_width2 = 0.2):
    
    '''
    WasserstainLIME(X_input, num_perturb = 500, kernel_width2 = 0.2):

    X_input: The input feature data for the WassersteinLIME function. It should be a 1D numpy array with two elements.
    
    kernel_width2: The kernel width parameter for the Wasserstein distance calculation. It determines the size of the 
                   region around the original feature data that is considered for the linear regression model. Larger 
                   values of kernel_width2 result in a wider region and more perturbations being included in the explanation,
                   while smaller values result in a more localized explanation. The default value is 0.2.

    This function uses Wasserstein distance to generate local explanations for a binary classifier. 
    It creates num_perturb number of perturbed versions of the input feature data, and for each perturbation 
    it predicts the class probabilities, computes the Wasserstein distances between the original 
    and perturbed feature data, and uses the distances to weight a linear regression model that explains the binary predictions. 
    The function returns the perturbed feature data, binary predictions, weight of each perturbation, 
    coefficients of the linear regression model, and the predictions of the linear regression model.
    '''
    
    try:
        if not isinstance(X_input, np.ndarray) or X_input.ndim != 2:
            raise TypeError("X_input must be a 2-dimensional array.")
    except TypeError as te:
        print(te)
    
    try:
        if not isinstance(num_perturb, int):
            raise ValueError("num_perturb must be an integer.")
    except ValueError as ve:
        print(ve)
        
    try:
        if not np.isscalar(kernel_width2):
            raise ValueError("kernel_width2 must be a scalar.")
    except ValueError as ve:
        print(ve)
    
    X_lime = np.random.normal(0,1,size=(num_perturb,X_input.shape[1]))
    
    Xi2 = np.zeros((100,2))
    Xi2[:,0] = X_input[0] + np.random.normal(0,0.05,100)
    Xi2[:,1] = X_input[1] + np.random.normal(0,0.05,100)

    y_lime2  = np.zeros((num_perturb,1))
    WD       = np.zeros((num_perturb,1))
    weights2 = np.zeros((num_perturb,1))

    for ind, ii in enumerate(X_lime):
        temp1 = ii[0] + np.random.normal(0,0.4,100)
        temp2 = ii[1] + np.random.normal(0,0.4,100)
        df2 = pd.DataFrame()
        df2['x1'] = temp1
        df2['x2'] = temp2
        temp3 = model.predict(df2)
        y_lime2[ind] = np.argmax(np.bincount(temp3))
        WD1 = SafeML.Wasserstein_Dist(Xi2[:,0], df2[:]['x1'])
        WD2 = SafeML.Wasserstein_Dist(Xi2[:,1], df2[:]['x2'])
        WD[ind] = sum([WD1, WD2])
    
        weights2[ind] = np.sqrt(np.exp(-(WD[ind]**2)/(kernel_width2**2))) 
    
    weights2 = weights2.flatten()
    
    simpler_model2 = LinearRegression() 
    simpler_model2.fit(X_lime, y_lime2, sample_weight=weights2)
    y_linmodel2 = simpler_model2.predict(X_lime)
    y_linmodel2 = y_linmodel2 < 0.5 #Conver to binary class
    y_linmodel2 = y_linmodel2.flatten()
    
    return X_lime, y_lime2, weights2, y_linmodel2, simpler_model2.coef_.flatten()
