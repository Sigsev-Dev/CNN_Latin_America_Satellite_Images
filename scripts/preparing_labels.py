import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Convert unsupervised classification map to one-hot encoded labels
def one_hot_encode_labels(labels, num_classes):
    flat_labels = labels.flatten()
    encoder = OneHotEncoder(sparse=False, categories="auto")
    one_hot_labels = encoder.fit_transform(flat_labels.reshape(-1, 1))
    return one_hot_labels.reshape(labels.shape[0], labels.shape[1], num_classes)

# Example usage
num_classes = len(np.unique("./classified_image.tif"))  # Number of classes in the unsupervised map
one_hot_labels = one_hot_encode_labels("./classified_image.tif", num_classes)
print("One-hot Encoded Labels Shape:", one_hot_labels.shape)
