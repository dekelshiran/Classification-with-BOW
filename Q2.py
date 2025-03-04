import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import os
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.cluster import KMeans

import Q1
from Q1 import folder_path, images_files


def read_images_labels_folder():
    print('START read_images_labels_folder')
    all_images = []
    all_labels = []

    for file in images_files:
        # Load an image (original size is maintained)
        file_path = os.path.join(folder_path, file)
        try:
            if not file.lower().endswith('.jpg'):
                continue
            with Image.open(file_path) as img:

                if img.format == 'JPEG':
                    img = img.convert("RGB")
                    all_images.append(img)
                    all_labels.append(file.split('_')[0])
                else:
                    print(f'the file {file} has wrong type, it has type {img.format}')

        except Exception as e:
            print(f"Error processing {file}: {e}")
    print('END read_images_labels_folder')
    return all_images, all_labels


def extract_features(images, batch_size):
    print('START extract_features')
    # Load a pre-trained vgg-16 model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Remove the classifier (fully connected layers) to get the feature map from the last conv layer
    feature_extractor = nn.Sequential(*list(model.children())[:-2])
    # Set the feature extractor to evaluation mode
    feature_extractor.eval()
    # Image transformation (maintain original size)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(  # Normalize to ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Forward pass through the feature extractor
    inputs_tensor = [transform(image).unsqueeze(0) for image in images]  # Add batch dimension

    # Batch processing
    feats_map = []
    with torch.no_grad():  # Disable gradient computation for faster processing
        for i in range(0, len(inputs_tensor), batch_size):
            batch = torch.cat(inputs_tensor[i:i + batch_size])  # Create batch
            features = feature_extractor(batch).cpu()  # Extract features and move back to CPU
            feats_map.extend(features)
    print('END extract_features')
    return feats_map


def extract_feature(image):
    # Load a pre-trained vgg-16 model
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Remove the classifier (fully connected layers) to get the feature map from the last conv layer
    feature_extractor = nn.Sequential(*list(model.children())[:-2])
    # Set the feature extractor to evaluation mode
    feature_extractor.eval()
    # Image transformation (maintain original size)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(  # Normalize to ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    # Forward pass through the feature extractor
    feat_map = feature_extractor(input_tensor)

    return feat_map


n_clusters = 1


def compute_k_means(tensors):
    print('START compute_k_means')
    kmeans = KMeans(n_clusters, random_state=42)
    # flatten all tensors
    flattened_tensors = [tensor.flatten().numpy() for tensor in tensors]
    feats_array = np.vstack(flattened_tensors)
    print(feats_array)

    kmeans.fit(feats_array)
    print('END compute_k_means')

    return kmeans


def build_histograms(images, kmeans):
    print('START build_histograms')
    histograms = []

    for img in images:
        img = extract_feature(img)
        flattened_image = img.flatten().detach().numpy()  # Convert to numpy array a tensor
        # Get the cluster labels for the image
        labels = kmeans.predict(flattened_image.reshape(1, -1))
        # יצירת היסטוגרמה עבור התמונה
        histogram = np.histogram(labels, bins=np.arange(n_clusters + 1))[0]

        # נורמליזציה של ההיסטוגרמה
        histogram = histogram.astype(float)
        histogram /= histogram.sum()

        histograms.append(histogram)
    print('END build_histograms')

    return np.array(histograms)


# Main:
if __name__ == '__main__':
    print('MAIN START')
    # extract images & labels
    images, labels = read_images_labels_folder()

    label_encoder, X_train, y_train, X_test, y_test, X_val, y_val = Q1.split_train_test_val(labels, images)
    train_feats_map = extract_features(X_train, 8)

    kmeans = compute_k_means(train_feats_map)

    # extract histograms
    train_histograms = build_histograms(X_train, kmeans)
    val_histograms = build_histograms(X_val, kmeans)
    test_histograms = build_histograms(X_test, kmeans)

    # Use SVM to classified training batch
    svm = SVC(kernel='linear')
    svm.fit(train_histograms, y_train)

    # Evaluate performance on validation
    val_pred = svm.predict(val_histograms)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Evaluate performance on Test
    test_pred = svm.predict(test_histograms)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    ##################
    # פרמטרים לחיפוש
    param_grid = {
        'C': [0.1, 1, 10, 100],  # פרמטר הרגולריזציה
        'kernel': ['linear', 'rbf', 'poly'],  # סוג ה-kernel
        'gamma': ['scale', 'auto']  # רק עבור kernel לא-לינארי
    }

    # Grid Search עם Cross-Validation
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_histograms, y_train)

    # פרמטרים אופטימליים
    print(f"Best Parameters: {grid_search.best_params_}")

    # מסווג עם פרמטרים אופטימליים
    best_svm = grid_search.best_estimator_

    ###
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt

    # חיזוי הסתברויות על סט הבדיקה
    y_test_prob = best_svm.decision_function(test_histograms)
    print('y_test_prob:', y_test_prob.shape)
    # ROC לכל מחלקה
    plt.figure(figsize=(10, 8))
    for i in range(len(label_encoder.classes_)):
        fpr, tpr, _ = roc_curve(y_test == i, y_test_prob[:, i])
        auc = roc_auc_score(y_test == i, y_test_prob[:, i])
        plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC = {auc:.2f})")

    plt.title("ROC Curves for Each Class")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    ####

    # חיזוי על סט הבדיקה
    y_test_pred = best_svm.predict(test_histograms)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Precision, Recall, F1 לכל מחלקה
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_encoder.classes_))

    # Precision-Recall לכל מחלקה
    plt.figure(figsize=(10, 8))
    for i in range(len(label_encoder.classes_)):
        precision, recall, _ = precision_recall_curve(y_test == i, y_test_prob[:, i])
        plt.plot(recall, precision, label=f"{label_encoder.classes_[i]}")

    plt.title("Precision-Recall Curves for Each Class")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
