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

def xwhy_text(X_input_text, model, perturbations, embd, num_perturb = 50, kernel_width = 0.25, num_top_features = 10, eps=0.5):
    
    wod = X_input_text.split()

    wod = [word.strip('-.,!;()[]@><:') for word in wod]

    #finding unique
    unique = []
    for word in wod:
        if word not in unique:
            unique.append(word)

    # Sorting the Unique Values
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

def plot_text_heatmap(words, scores, title="", width=10, height=0.4, verbose=0, max_word_per_line=20, word_spacing=20, score_fontsize=10):

    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(width, height))

    ax = plt.gca()

    ax.set_title(title, loc='left')

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)

    canvas = ax.figure.canvas
    t = ax.transData

    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5

    loc_y = -0.2

    for i, (token, score) in enumerate(zip(words, scores)):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)

        # Draw the word
        text = ax.text(0.0, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'round,pad=0.5'
                       }, transform=t, fontsize=14)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()

        # Draw the score beneath the word
        score_text = ax.text(0.01, loc_y -1, f"{score:.2f}",
                             transform=t, fontsize=score_fontsize, ha='center')

        score_text.draw(canvas.get_renderer())
        ex_score = score_text.get_window_extent()

        # Check if a new line is needed
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y -  2.5
            t = ax.transData
        else:
            # Increase the horizontal space between words
            t = transforms.offset_copy(text._transform, x=ex.width + word_spacing, units='dots')

    if verbose == 0:
        ax.axis('off')

    # shap_values = Similarities.copy()
    
    plot_text_heatmap(
        "What is the Meaning of Life?".split(' '),
        np.array(shap_values)
    )
