## White-Box Testing and Benchmark Functions
<p align="justified">White-box testing is a software testing technique that examines the internal structure and logic of a system. When it comes to testing model-agnostic explainability methods like LIME, SHAP, and SMILE, white-box testing plays a crucial role in ensuring their correctness and effectiveness.</p>
<p align="justified">In the context of explainability methods, white-box testing involves verifying the accuracy of the underlying algorithms and evaluating their performance in generating meaningful explanations (local sensitivity). It entails examining the internal mechanisms of these methods, such as the local feature importance calculations and interpretability models they employ. Thoroughly testing these components is essential for assessing the consistency and trustworthiness of the explanations produced.
<p align="justified">To perform white-box testing for xwhy, the following benchmark functions can be utilized to evaluate the robustness and validity of the explainability methods:</p>

## Todo:
The following functions are in our todo list to be included for white-box testing of xwhy:

1. **Ishigami Function**: 

This is a commonly used test function for uncertainty and sensitivity analysis methods, because it exhibits strong nonlinearity and nonmonotonicity.

$$
f(\mathbf{x}) = \sin(x_1) + a \sin^2(x_2) + b x_3^4 \sin(x_1)
$$

where $\mathbf{x} = (x_1, x_2, x_3)$.

2. **Sobol Function**:

This is another benchmark function used for testing global sensitivity analysis methods, in particular for methods aimed at computing Sobol indices.

$$
f(\mathbf{x}) = \prod_{i=1}^{d} \left(\frac{|4x_i-2|+a_i}{1+a_i}\right)
$$

where $d$ is the dimension of the input vector $\mathbf{x}$, and $a_i$ are parameters that determine the importance of each input factor.

3. **Morris Function (also known as Morris OAT Function)**:

It is a polynomial of degree 2 designed for testing the Morris method for elementary effects.

$$
y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i=1}^{k} \beta_{ii} x_i^2 + \sum_{i < j} \beta_{ij} x_i x_j
$$

where $\beta$ are coefficients and $x_i$ are the input variables.

4. **Branin-Hoo Function**:

This function is often used in global optimization problems, but can also be used in sensitivity analysis.

$$
f(x, y) = a (y - b x^2 + c x - r)^2 + s (1-t) \cos(x) + s
$$

where $a, b, c, r, s,$ and $t$ are constant parameters.

5. **The Rosenbrock function**:

Also known as the Rosenbrock's valley or Rosenbrock's banana function, is a popular test example for optimization algorithms.

$$
f(x,y) = (a - x)^2 + b(y - x^2)^2
$$

where typically $a=1$ and $b=100$.
