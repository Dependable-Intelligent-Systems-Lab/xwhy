import numpy as np

import lime

import numpy as np
import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.metrics
#from __future__ import print_function

import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_20newsgroups

import string
import re
import nltk
from nltk.tokenize import TweetTokenizer

def xwhy_text(X_input_text, model, perturbations=perturbations, embd= embedding_google, num_perturb = 50, kernel_width = 0.25, num_top_features = 10, eps=0.5):
    
#     # NLP pre-processing
#     # remove urls, handles, and the hashtag from hashtags 
#     # (taken from https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression)
#     def remove_urls(text):
#         new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
#         return new_text

#     # make all text lowercase
#     def text_lowercase(text): 
#         return text.lower()

#     # remove numbers
#     def remove_numbers(text): 
#         result = re.sub(r'\d+', '', text) 
#         return result

#     # remove punctuation
#     def remove_punctuation(text): 
#         translator = str.maketrans('', '', string.punctuation)
#         return text.translate(translator)

#     # function for all pre-processing steps
#     def preprocessing(text):
#         text = text_lowercase(text)
#         text = remove_urls(text)
#         text = remove_numbers(text)
#         text = remove_punctuation(text)
#         return text

#     # pre-processing the text body column
#     pp_text = []
#     for text_data in X_input_text:
#         # check if string
#         if isinstance(text_data, str):
#             pp_text_data = preprocessing(text_data)
#             pp_text.append(pp_text_data)
#         # if not string
#         else:
#             pp_text.append(np.NaN)
#     cleaned_words = clean_text(X_input_text)

    wod = X_input_text.split()
# #     words = pp_text
    wod = [word.strip('-.,!;()[]@><:') for word in wod]
#     wod = [word.replace("'s", '') for word in wod]
    wod = [word.replace(".", '') for word in wod]
    wod = [word.replace("-", '') for word in wod]
#     words = [word.replace(":", '') for word in words]
#     words = [word.replace(">", '') for word in words]
    

    #finding unique
    unique = []
    for word in wod:
        if word not in unique:
            unique.append(word)

    #sort
    unique.sort()
    
    #num_perturb = 150
    num_uniqe_words = len(wod)
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_uniqe_words))
    text_list = wod.copy()
    
    def perturb_text(text_list, perturbation):
        for x, y in enumerate(text_list):
            if perturbation[x] == 0:
                text_list.remove(y)
                
    predictions = []
    WD_dist = []
    for pert in perturbations:
        perturbed_text = perturb_text(text_list, pert)
        pred = model.predict_proba([str(perturbed_text)])
        predictions.append(pred)
        WD_score = embd.wmdistance(str(wod), str(perturbed_text))
        WD_dist.append(WD_score)

    predictions = np.array(predictions)   
    WD_dist = np.array(WD_dist) 
    
    weights = np.sqrt(np.exp(-((eps*WD_dist)**2)/kernel_width**2)) #Kernel function
    
    print(perturbations.shape)
    
#     print(predictions.shape)
    
    print(weights.shape)
    
    print(predictions[:,:,0].shape)
                
    class_to_explain = 0
    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions[:,:, class_to_explain], sample_weight=weights)
    coeff = simpler_model.coef_[0]
    
    print(coeff.shape)

    coeff3 = coeff[0:50]
    
    num_top_features = 50
    top_features = np.argsort(abs(coeff))[-num_top_features:] 
    
    coeff2 = simpler_model.coef_
    
    print(coeff3.shape)
    # https://stackoverflow.com/questions/39626401/how-to-get-odds-ratios-and-other-related-features-with-scikit-learn
    odds = np.exp(coeff2)
     
    return coeff3, coeff, odds, top_features, wod
