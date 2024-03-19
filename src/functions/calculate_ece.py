from sklearn.calibration import calibration_curve
import numpy as np

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE)
    
    Parameters:
    - y_true: array-like, true labels.
    - y_prob: array-like, predicted probabilities for the positive class.
    - n_bins: int, number of bins to use for the calibration curve.
    
    Returns:
    - ece: float, the expected calibration error.
    """
    # Find the maximum predicted probability for each sample
    y_prob_max = np.max(y_prob, axis=1)
    # Determine whether each prediction is correct
    y_pred = np.argmax(y_prob, axis=1)
    correct_predictions = (y_pred == y_true).astype(int)

    # Use calibration_curve to compute the true and predicted probabilities
    prob_true, prob_pred = calibration_curve(correct_predictions, y_prob_max, n_bins=n_bins, strategy='uniform')

    # Compute bin widths (uniform)
    bin_widths = 1.0 / n_bins

    # Calculate the ECE as the weighted sum of absolute differences between prob_true and prob_pred
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_widths)

    return ece