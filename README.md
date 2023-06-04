<p align="left"> </p>

 <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
 <a href="https://standardjs.com"><img src="https://img.shields.io/badge/code_style-standard-brightgreen.svg" alt="Standard - \Python Style Guide"></a> 
 
# X-Why
XWhy: eXplain Why with <b>SMILE</b> -- <b>S</b>tatistical <b>M</b>odel-agnostic <b>I</b>nterpretability with <b>L</b>ocal <b>E</b>xplanations
 
<p align="center">
 <img src="https://github.com/koo-ec/xwhy/blob/main/docs/graphics/XWhy_Logo_v1.png" alt="XWhy, SMILE, Explainability, Interpretability, XAI, machine learning explainability, responsible ai"> </p>

## Abstract
<p align="justify">Machine learning is currently undergoing an explosion in capability, popularity, and sophistication. However, one of the major barriers to widespread acceptance of machine learning (ML) is trustworthiness: most ML models operate as black boxes, their inner workings opaque and mysterious, and it can be difficult to trust their conclusions without understanding how those conclusions are reached. Explainability is therefore a key aspect of improving trustworthiness: the ability to better understand, interpret, and anticipate the behaviour of ML models. To this end, we propose a SMILE, a new method that builds on previous approaches by making use of statistical distance measures to improve explainability while remaining applicable to a wide range of input data domains.</p>


## Installation
```
pip install xwhy
```
<!--
## Simple Example
```
import xwhy
import xgboost

# train an XGBoost model
X, y = xwhy.datasets.boston()
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using xwhy
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = xwhy.Explainer(model)
xwhy_values = explainer(X)

# visualize the first prediction's explanation
xwhy.plots.waterfall(xwhy_values[0])

```
-->
 
 
 
## Citations
It would be appreciated a citation to our paper as follows if you use X-Why for your research:
```
@article{Aslansefat2021Xwhy,
   author  = {{Aslansefat}, Koorosh and {Hashemian}, Mojgan and {Martin}, Walker, {Akram} Mohammed Naveed, {Sorokos} Ioannis and {Papadopoulos}, Yiannis},
   title   = "{SMILE: Statistical Model-agnostic Interpretability with Local Explanations}",
   journal = {arXiv e-prints},
   year    = {2021},
   url     = {https://arxiv.org/abs/...},
   eprint  = {},
}
```
 
## Acknowledgment
<p align="justify">This project is supported by the <a href = "https://www.sesame-project.org"><b>Secure and Safe Multi-Robot Systems (SESAME)</b></a> H2020 Project under Grant Agreement 101017258.</p>

## Awards
<a href = "https://www.turing.ac.uk/post-doctoral-enrichment-awards-pdea">Post-Doctoral Enrichment Award from the Alan Turing Institute</a>

## Contribution 
If you are interested in contributing to this project, please check the [contribution guidelines](https://github.com/koo-ec/xwhy/blob/main/docs/contribute/contributing.md).
