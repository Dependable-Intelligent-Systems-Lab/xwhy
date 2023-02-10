import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression

def Wasserstein_Dist(XX, YY):
    '''
    Wasserstein_Dist_PVal is for Wasserstein distance measure with Boostrap-based p-value calculation.
    The p-Value can be used to validate statistical distance measures.
    
    XX: The first input vector. It should be a numpy array with length of n.
    YY: The second input vector. It should be a numpy array with lenght of m.
    '''

    import numpy as np
    nx = len(XX)
    ny = len(YY)
    n = nx + ny

    XY = np.concatenate([XX,YY])
    X2 = np.concatenate([np.repeat(1/nx, nx), np.repeat(0, ny)])
    Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1/ny, ny)])

    S_Ind = np.argsort(XY)
    XY_Sorted = XY[S_Ind]
    X2_Sorted = X2[S_Ind]
    Y2_Sorted = Y2[S_Ind]

    Res = 0
    E_CDF = 0
    F_CDF = 0
    power = 1

    for ii in range(0, n-2):
        E_CDF = E_CDF + X2_Sorted[ii]
        F_CDF = F_CDF + Y2_Sorted[ii]
        height = abs(F_CDF-E_CDF)
        width = XY_Sorted[ii+1] - XY_Sorted[ii]
        Res = Res + (height ** power) * width;  
 
    return Res

def  Wasserstein_Dist_PVal(XX, YY):
    '''
    Wasserstein_Dist_PVal is for Wasserstein distance measure with Boostrap-based p-value calculation.
    The p-Value can be used to validate statistical distance measures.
    
    XX: The first input vector. It should be a numpy array with length of n.
    YY: The second input vector. It should be a numpy array with lenght of m.
    '''
    
    import random
    nboots = 1000
    WD = Wasserstein_Dist(XX,YY)
    na = len(XX)
    nb = len(YY)
    n = na + nb
    comb = np.concatenate([XX,YY])
    reps = 0
    bigger = 0
    for ii in range(1, nboots):
        e = random.sample(range(n), na)
        f = random.sample(range(n), nb)
        boost_WD = Wasserstein_Dist(comb[e],comb[f]);
        if (boost_WD > WD):
            bigger = 1 + bigger
            
    pVal = bigger/nboots;

    return pVal, WD

def WasserstainLIME(X_input, model = model, num_perturb = 500, kernel_width2 = 0.2):
    
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
        WD1 = Wasserstein_Dist(Xi2[:,0], df2[:]['x1'])
        WD2 = Wasserstein_Dist(Xi2[:,1], df2[:]['x2'])
        WD[ind] = sum([WD1, WD2])
    
        weights2[ind] = np.sqrt(np.exp(-(WD[ind]**2)/(kernel_width2**2))) 
    
    weights2 = weights2.flatten()
    
    simpler_model2 = LinearRegression() 
    simpler_model2.fit(X_lime, y_lime2, sample_weight=weights2)
    y_linmodel2 = simpler_model2.predict(X_lime)
    y_linmodel2 = y_linmodel2 < 0.5 #Conver to binary class
    y_linmodel2 = y_linmodel2.flatten()
    
    return X_lime, y_lime2, weights2, y_linmodel2, simpler_model2.coef_.flatten()
