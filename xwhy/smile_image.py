import numpy as np
from keras.applications.imagenet_utils import decode_predictions
import skimage.io
import skimage.segmentation
from sklearn.linear_model import LinearRegression

# Utility Functions
class WassersteinUtility:
    @staticmethod
    def calculate_distance(XX, YY):
        """
        Calculate Wasserstein distance between two distributions.
        """
        nx, ny = len(XX), len(YY)
        n = nx + ny

        XY = np.concatenate([XX, YY])
        X2 = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]

        Res, E_CDF, F_CDF = 0, 0, 0
        for ii in range(0, n - 2):
            E_CDF += X2_Sorted[ii]
            F_CDF += Y2_Sorted[ii]
            height = abs(F_CDF - E_CDF)
            width = XY_Sorted[ii + 1] - XY_Sorted[ii]
            Res += height * width
        return Res

    @staticmethod
    def calculate_p_value(XX, YY, nboots=1000):
        """
        Bootstrap-based p-value calculation for Wasserstein distance.
        """
        WD = WassersteinUtility.calculate_distance(XX, YY)
        comb = np.concatenate([XX, YY])
        na, nb = len(XX), len(YY)
        n = na + nb

        bigger = sum(
            WassersteinUtility.calculate_distance(
                comb[np.random.choice(range(n), na, replace=True)],
                comb[np.random.choice(range(n), nb, replace=True)]
            ) > WD
            for _ in range(nboots)
        )

        return bigger / nboots, WD

# Image Perturbation Class
class ImagePerturbation:
    @staticmethod
    def perturb_image(img, perturbation, segments):
        """
        Apply perturbation mask to the image.
        """
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1
        perturbed_image = img * mask[:, :, np.newaxis]
        return perturbed_image

# Main Class for Image Explanations
class ImageExplainer:
    def __init__(self, model, kernel_width=0.25):
        self.model = model
        self.kernel_width = kernel_width

    def explain(self, X_input, perturbations, num_perturb=150):
        """
        Explain model predictions for a given image.
        """
        superpixels = skimage.segmentation.quickshift(X_input, kernel_size=4, max_dist=200, ratio=0.2)
        num_superpixels = np.unique(superpixels).shape[0]

        predictions, weights, wd_distances = [], [], []
        for pert in perturbations:
            perturbed_img = ImagePerturbation.perturb_image(X_input, pert, superpixels)
            pred = self.model.predict(perturbed_img[np.newaxis, :, :, :])
            wd_dist = WassersteinUtility.calculate_distance(
                X_input.flatten(), perturbed_img.flatten()
            )
            predictions.append(pred)
            wd_distances.append(wd_dist)

        predictions = np.array(predictions)
        wd_distances = np.array(wd_distances)

        weights = np.sqrt(np.exp(-(wd_distances ** 2) / self.kernel_width ** 2))

        preds = self.model.predict(X_input[np.newaxis, :, :, :])
        decode_predictions(preds)
        top_pred_classes = preds[0].argsort()[-5:][::-1]
        class_to_explain = top_pred_classes[0]

        simpler_model = LinearRegression()
        simpler_model.fit(
            X=perturbations, y=predictions[:, :, class_to_explain], sample_weight=weights
        )

        return simpler_model.coef_[0]

# Example Usage
# model = load_your_model_here
# explainer = ImageExplainer(model)
# coefficients = explainer.explain(X_input, perturbations)
