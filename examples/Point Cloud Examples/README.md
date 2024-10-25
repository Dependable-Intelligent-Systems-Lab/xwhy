<p align="left"> </p>

 <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
 <a href="https://standardjs.com"><img src="https://img.shields.io/badge/code_style-standard-brightgreen.svg" alt="Standard - \Python Style Guide"></a> 
 
# Explainability of Point Cloud Neural Networks Using SMILE: Statistical Model-Agnostic Interpretability with Local Explanations
 
<p align="center">
 <img src="https://github.com/koo-ec/xwhy/blob/main/docs/graphics/XWhy_Logo_v1.png" alt="XWhy, SMILE, Explainability, Interpretability, XAI, machine learning explainability, responsible ai"> </p>

## Abstract
<p align="justify">This study explores the implementation of SMILE for Point Cloud offering enhanced robustness and interpretability, particularly when Anderson-Darling distance is used. The approach demonstrates superior performance in terms of fidelity loss, R^2 scores, and robustness across various kernel widths, perturbation numbers, and clustering configurations. Moreover, this study introduces a stability analysis for point cloud data using the Jaccard index, establishing a new benchmark and baseline for model stability in this field. The study further identifies dataset biases in the classification of the ‘person’ category, emphasizing the necessity for more comprehensive datasets in safety-critical applications like autonomous driving and robotics. The results underscore the potential of advanced explainability models and highlight areas for future research, including the application of alternative surrogate models and explainability techniques in point cloud data.</p>

## Proposed Flowchart

<p align="center">
 <img src="https://github.com/Dependable-Intelligent-Systems-Lab/xwhy/blob/main/examples/Point%20Cloud%20Examples/Figures/PC_SMILE.png" alt="XWhy, SMILE, Explainability, Interpretability, XAI, machine learning explainability, responsible ai"> </p>

## Sample Results
<p align="center">
 <img src="https://github.com/Dependable-Intelligent-Systems-Lab/xwhy/blob/main/examples/Point%20Cloud%20Examples/Figures/Screenshot%202024-09-27%20230842.png" alt="XWhy, SMILE, Explainability, Interpretability, XAI, machine learning explainability, responsible ai"> </p>


# Point Cloud Examples
Try the code on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ORrCAyQSRmtv08SHtnf_LzQnfm76Wzqz?usp=sharing)

- [Notebook 1](https://github.com/Dependable-Intelligent-Systems-Lab/xwhy/tree/main/examples/Point%20Cloud%20Examples/Notebooks)
- [Notebook 2 on Kaggle](https://www.kaggle.com/code/mohammadahmadi66/point-cloud-explainability-with-smile)
- [Notebook 3](https://github.com/Dependable-Intelligent-Systems-Lab/xwhy/tree/main/examples/Point%20Cloud%20Examples/Notebooks)

## Video Explanation
For a detailed explanation, check out my video on YouTube: [Watch here](https://www.youtube.com/watch?v=AzYz-JUJTxs&t=80s).

# Related Works
| Method                  | Ex. Type | Learning | Task | Approach | Flow     | Dataset                                     |
|-------------------------|----------|----------|------|----------|----------|---------------------------------------------|
| PointHop [[1]]          | ML       | Yes      | PC   | MS       | Forward  | Modelnet40 [[2]]                            |
| Non-Contribution Factors [[3]] | IL       | No       | PC   | MS       | Backward | Modelnet40 [[2]]                            |
| LIME [[4]]              | IL       | Yes      | PC   | MA       | Forward  | Modelnet40 [[2]]                            |
| Gradient-Based [[5]]    | IL       | No       | PC/VD| MS       | Backward | Modelnet40 [[2]]                            |
| BubblEX [[6]]           | IL       | No       | PC   | MA       | Backward | Modelnet40 [[2]], ScanObjectNN [[7]]        |
| AM [[8]]                | ML       | Yes      | PC   | MS       | Backward | ModelNet40 [[2]], ShapeNet [[9]]            |
| DAM [[10]]              | ML       | Yes      | PC   | MS       | Backward | ModelNet40 [[2]], ShapeNet [[9]]            |
| FBI [[11]]              | IL       | No       | PC   | MA       | Forward  | ModelNet40 [[2]], ModelNet-C [[12]], ScanObjectNN [[7]] |
| <b>SMILE (Our Method)</b>   | IL       | Yes      | PC   | MA       | Forward  | ModelNet40 [[2]]                            |



# Citation
If you find **SMILE for Point Cloud** helpful in your research, please consider citing our work:

```bibtex
@article{smile2024pointcloud,
  title={SMILE: Explainability of Point Cloud Neural Networks Using SMILE: Statistical Model-Agnostic Interpretability with Local Explanations},
  author={Ahmadi, Seyed Mohammad and Aslansefat, Kooroosh and Valcarce-Dineiro, Ruben and Barnfather, Joshua},
  journal={arXiv  preprint arXiv:2410.15374},
  year={2024}
  DOI: [Arxiv](https://arxiv.org/abs/2410.15374)
}
```




# References
1. Zhang, M., et al. (2020). *PointHop: An explainable machine learning method for point cloud classification*. IEEE Transactions on Multimedia, 22(7), 1744-1755.
2. Wu, Z., et al. (2015). *3D ShapeNets: A deep representation for volumetric shapes*. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
3. Liang, A., Zhang, H., & Hua, H. (2022). *Point Cloud Saliency Maps Based on Non-Contribution Factors*. Neurocomputing, 194-198.
4. Tan, H., & Kotthaus, H. (2022). *Surrogate model-based explainability methods for point cloud nns*. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision.
5. Gupta, A., Watson, S., & Yin, H. (2020). *3D point cloud feature explanations using gradient-based methods*. In 2020 International Joint Conference on Neural Networks (IJCNN). IEEE.
6. Matrone, F., et al. (2022). *BubblEX: An Explainable Deep Learning Framework for Point-Cloud Classification*. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 15, 1-18.
7. Uy, M. A., et al. (2019). *Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data*. In Proceedings of the IEEE/CVF International Conference on Computer Vision.
8. Tan, H. (2023). *Visualizing global explanations of point cloud dnns*. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision.
9. Chang, A. X., et al. (2015). *ShapeNet: An information-rich 3d model repository*. arXiv preprint arXiv:1512.03012.
10. Tan, H. (2024). *DAM: Diffusion Activation Maximization for 3D Global Explanations*. arXiv preprint arXiv:2401.14938.
11. Levi, M. Y., & Gilboa, G. (2024). *Fast and Simple Explainability for Point Cloud Networks*. arXiv preprint arXiv:2403.07706.
12. Ren, J. L., Pan, & Liu, Z. (2022). *Benchmarking and analyzing point cloud classification under corruptions*. In International Conference on Machine Learning. PMLR.
