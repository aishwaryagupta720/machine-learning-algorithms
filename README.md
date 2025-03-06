## **ML Algorithms Implementations**

### **Repository Description:**
A collection of machine learning algorithms implemented from scratch, including regression, classification, clustering, and dimensionality reduction. Each project follows a structured approach with detailed explanations, visualizations, and performance evaluations. Implemented in Python using NumPy, Matplotlib, and Pandas, these projects demonstrate fundamental ML concepts without relying on high-level libraries like scikit-learn.

### **Projects Included:**
- Linear Regression with batch gradient descent (predicting house prices)
- Logistic Regression for binary classification (Iris dataset)
- Minimum Risk Bayes Classifier using Gaussian distributions
- K-Means Clustering for unsupervised learning
- Principal Component Analysis (PCA) for dimensionality reduction

## 📌 Projects Included

### 1️⃣ Linear Regression with Batch Gradient Descent
**Description:**  
Trains a linear regression model to predict house prices based on square footage using a dataset of real estate listings from San Luis Obispo County.

**Tasks:**
- Train the model using batch gradient descent.
- Plot MSE vs. number of epochs.
- Visualize the regression line over training data.
- Predict the price of a 5000-square-foot house.

📌 **Expected Output:**  
- Regression line plot  
- MSE vs. epochs graph  
- Predicted price output  

📂 **Directory:** `linear_regression/`

---

### 2️⃣ Logistic Regression with Batch Gradient Descent
**Description:**  
Implements logistic regression to classify a modified 4D, 2-class Iris dataset.

**Tasks:**
- Train the model with gradient descent for 2000 epochs.
- Plot the cost function vs. epochs.
- Evaluate classification accuracy on the test set.

📌 **Expected Output:**  
- Cost function vs. epochs plot  
- Predicted vs. actual class labels  
- Accuracy ≥ 90%  

📂 **Directory:** `logistic_regression/`

---

### 3️⃣ Minimum Risk Bayes Theoretic Classifier
**Description:**  
Implements a Minimum Risk Bayes classifier for the 2-class, 4D Iris dataset.

**Tasks:**
- Compute mean and covariance for Gaussian distributions.
- Predict class labels on validation data.
- Evaluate classification accuracy.

📌 **Expected Output:**  
- Printed actual vs. predicted labels  
- Accuracy ≥ 90%  

📂 **Directory:** `bayes_classifier/`

---

### 4️⃣ K-Means Clustering
**Description:**  
Implements the K-Means clustering algorithm on the Iris dataset.

**Tasks:**
- Cluster samples into groups.
- Visualize cluster assignments and centroids.
- Print number of iterations until convergence.

📌 **Expected Output:**  
- Clustering plot with centroids  
- Algorithm converges in ~5 iterations  

📂 **Directory:** `kmeans_clustering/`

---

### 5️⃣ Principal Component Analysis (PCA)
**Description:**  
Implements PCA using eigen decomposition on the Iris dataset to reduce dimensionality.

**Tasks:**
- Compute principal components.
- Reconstruct dataset (`X_hat`).
- Train an LDA classifier on `X_hat` and evaluate accuracy.

📌 **Expected Output:**  
- Reconstruction error for 1 PC and 4 PCs  
- LDA accuracy ≥ 0.90 (1 PC) and ≥ 0.98 (4 PCs)  

📂 **Directory:** `pca/`

