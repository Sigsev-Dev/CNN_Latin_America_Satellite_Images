import numpy as np
import rasterio
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

def assess_accuracy(predicted_map, ground_truth_map):
    """
    Perform accuracy assessment by comparing predicted and ground truth maps.

    Args:
        predicted_map (np.ndarray): Predicted classification map.
        ground_truth_map (np.ndarray): Ground truth classification map.

    Returns:
        dict: Accuracy metrics and confusion matrix.
    """
    # Flatten maps for pixel-wise comparison
    pred_flat = predicted_map.flatten()
    gt_flat = ground_truth_map.flatten()

    # Filter out invalid pixels (e.g., no-data)
    valid_mask = gt_flat >= 0  # Assuming negative values represent no-data
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]

    # Compute confusion matrix
    conf_matrix = confusion_matrix(gt_flat, pred_flat)
    overall_accuracy = accuracy_score(gt_flat, pred_flat)

    # Calculate per-class metrics
    user_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=0)  # Precision
    producer_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)  # Recall

    # Return metrics and confusion matrix
    return {
        "confusion_matrix": conf_matrix,
        "overall_accuracy": overall_accuracy,
        "user_accuracy": user_accuracy,
        "producer_accuracy": producer_accuracy,
    }

def plot_difference_map(predicted_map, ground_truth_map):
    """
    Plot the difference between predicted and ground truth maps to visualize errors.

    Args:
        predicted_map (np.ndarray): Predicted classification map.
        ground_truth_map (np.ndarray): Ground truth classification map.
    """
    difference = predicted_map - ground_truth_map
    plt.figure(figsize=(10, 10))
    plt.title("Difference Map (Prediction - Ground Truth)")
    plt.imshow(difference, cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Class Difference")
    plt.show()

# Main Script
if __name__ == "__main__":
    # Paths to predicted and ground truth maps
    predicted_map_path = "predicted_map_2000epochs_Teresina.tif"
    ground_truth_map_path = "../outputs/recolored_classification.tif"

    # Load predicted and ground truth maps
    with rasterio.open(predicted_map_path) as src:
        predicted_map = src.read(1)  # Read first band
    with rasterio.open(ground_truth_map_path) as src:
        ground_truth_map = src.read(1)  # Read first band

    # Perform accuracy assessment
    metrics = assess_accuracy(predicted_map, ground_truth_map)

    # Print metrics
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"Overall Accuracy: {metrics['overall_accuracy'] * 100:.2f}%")
    print("User's Accuracy (Precision) per Class:")
    print(metrics["user_accuracy"])
    print("Producer's Accuracy (Recall) per Class:")
    print(metrics["producer_accuracy"])

    # Visualize difference map
    plot_difference_map(predicted_map, ground_truth_map)
