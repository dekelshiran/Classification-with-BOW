import os
import cv2
import numpy as np

import sklearn
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn import datasets
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# getting a graph with the best KMeans clusters number
def elbow_method(all_descriptors):
    print('START ELBOW METHOD')
    km = KMeans(random_state=42)
    visualizer = KElbowVisualizer(km, k=(2, 20))

    visualizer.fit(all_descriptors)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure
    print('END ELBOW METHOD')


def read_images_labels_folder():
    print('START read_images_labels_folder')
    images = []
    labels = []

    for file in images_files:
        img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(file.split('_')[0])
    print('END read_images_labels_folder')
    return images, labels


def sift_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors


# function get images that have already been read
def create_descriptors(read_images):
    print('START create_descriptors')
    all_descriptors = []
    for image in read_images:
        descriptors = sift_descriptors(image)

        if descriptors is not None:
            all_descriptors.append(descriptors)

    all_descriptors = np.vstack(all_descriptors)
    print('END create_descriptors')

    return all_descriptors


def compute_k_means(descriptors_array):
    print('START compute_k_means')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(descriptors_array)
    print('END compute_k_means')

    return kmeans


def build_histograms(images, kmeans):
    print('START build_histograms')
    histograms = []

    for img in images:

        descriptors = sift_descriptors(img)
        if descriptors is not None:
            # קבלת תוויות הקלאסטרים עבור הדסקריפטורים של התמונה
            labels = kmeans.predict(descriptors)

            # יצירת היסטוגרמה עבור התמונה
            histogram = np.histogram(labels, bins=np.arange(n_clusters + 1))[0]

            # נורמליזציה של ההיסטוגרמה
            histogram = histogram.astype(float)
            histogram /= histogram.sum()

            histograms.append(histogram)
    print('END build_histograms')

    return np.array(histograms)


def split_train_test_val(labels, images):
    print('START split_train_test_val')
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    label_encoder.fit(labels)

    # splitting train- 60%, test- 20%, validation- 20%
    X_temp, X_test, y_temp, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    print('END split_train_test_val')

    return label_encoder, X_train, y_train, X_test, y_test, X_val, y_val


# images folder
folder_path = 'images'
images_files = os.listdir(folder_path)
n_clusters = 8


# Main:
if __name__ == '__main__':

    # extract images & labels
    images, labels = read_images_labels_folder()

    label_encoder, X_train, y_train, X_test, y_test, X_val, y_val = split_train_test_val(labels, images)

    # extract descriptors
    train_descriptors = create_descriptors(X_train)

    # elbow_method(train_descriptors)

    # extract kmeans
    kmeans = compute_k_means(train_descriptors)

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
