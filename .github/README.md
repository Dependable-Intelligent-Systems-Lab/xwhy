<p align="left"> </p>

 <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
 <a href="https://standardjs.com"><img src="https://img.shields.io/badge/code_style-standard-brightgreen.svg" alt="Standard - \Python Style Guide"></a> 
 
# Explaining black boxes with a <b>SMILE</b> -- <b>S</b>tatistical <b>M</b>odel-agnostic <b>I</b>nterpretability with <b>L</b>ocal <b>E</b>xplanations
 
<p align="center">
 <img src="https://github.com/koo-ec/xwhy/blob/main/docs/graphics/XWhy_Logo_v1.png" alt="XWhy, SMILE, Explainability, Interpretability, XAI, machine learning explainability, responsible ai"> </p>

## Abstract
<p align="justify">Machine learning is currently undergoing an explosion in capability, popularity, and sophistication. However, one of the major barriers to widespread acceptance of machine learning (ML) is trustworthiness: most ML models operate as black boxes, their inner workings opaque and mysterious, and it can be difficult to trust their conclusions without understanding how those conclusions are reached. Explainability is therefore a key aspect of improving trustworthiness: the ability to better understand, interpret, and anticipate the behaviour of ML models. To this end, we propose a SMILE, a new method that builds on previous approaches by making use of statistical distance measures to improve explainability while remaining applicable to a wide range of input data domains</p>

## News
The SMILE approach has been extended for <b>Point Cloud Explainability</b>. Please check out the examples [<b>here</b>](https://github.com/Dependable-Intelligent-Systems-Lab/xwhy/tree/main/examples/Point%20Cloud%20Examples).

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

## Example of using SMILE for Image Explainability
<p align="center">
 <img src="https://github.com/Dependable-Intelligent-Systems-Lab/xwhy/blob/main/docs/graphics/SMILE_Sample.jpg" alt="XWhy, SMILE, Explainability, Interpretability, XAI, machine learning explainability, responsible ai"> </p>

## Publications 
<p align="justify"> [1] Aslansefat, K., Hashemian, M., Walker, M., Akram, M. N., Sorokos, I., & Papadopoulos, Y. (2023). Explaining black boxes with a SMILE: Statistical Model-agnostic Interpretability with Local Explanations. IEEE Software. <a href = "https://doi.org/10.1109/MS.2023.3321282">DOI: 10.1109/MS.2023.3321282</a>, <a href = "https://arxiv.org/abs/2311.07286">Arxiv</a>, <a href = "https://hull-repository.worktribe.com/output/4415493/explaining-black-boxes-with-a-smile-statistical-model-agnostic-interpretability-with-local-explanations">WorkTribe</a>.</p> 
 
## Citations
It would be appreciated a citation to our paper as follows if you use X-Why for your research:
```
@article{aslansefat2023explaining,
  title={Explaining black boxes with a SMILE: Statistical Model-agnostic Interpretability with Local Explanations},
  author={Aslansefat, Koorosh and Hashemian, Mojgan and Walker, Martin and Akram, Mohammed Naveed and Sorokos, Ioannis and Papadopoulos, Yiannis},
  journal={IEEE Software},
  year={2023},
  publisher={IEEE}
}
```
 
## Acknowledgment
<p align="justify">This project is supported by the <a href = "https://www.sesame-project.org"><b>Secure and Safe Multi-Robot Systems (SESAME)</b></a> H2020 Project under Grant Agreement 101017258.</p>

## Awards
<a href = "https://www.turing.ac.uk/post-doctoral-enrichment-awards-pdea">Post-Doctoral Enrichment Award from the Alan Turing Institute</a>

## Contribution 
If you are interested in contributing to this project, please check the [contribution guidelines](https://github.com/koo-ec/xwhy/blob/main/docs/contribute/contributing.md).

## Contributors

[![xwhy contributors](https://contrib.rocks/image?repo=Dependable-Intelligent-Systems-Lab/xwhy&max=2000)]([https://github.com/langchain-ai/langchain](https://github.com/Dependable-Intelligent-Systems-Lab/xwhy)https://github.com/Dependable-Intelligent-Systems-Lab/xwhy/graphs/contributors)
