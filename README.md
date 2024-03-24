
# Breast Cancer Prediction Using Gene Expression Data

### Problem Statement and Description

#### Problem Statement

The current approach to cancer diagnosis relies heavily on manual examination of patient data, leading to delays, inconsistencies, and potential misinterpretations. Limited data analysis capabilities hinder the comprehensive assessment of gene expression data, crucial for early detection. Without sophisticated computational tools, extracting actionable insights from complex datasets is challenging. Consequently, there's a pressing need for an accurate, efficient, and automated model leveraging data science and machine learning techniques to analyze gene expression data. This model aims to improve diagnostic accuracy, enable early detection, and facilitate timely interventions, ultimately enhancing patient outcomes.

#### Problem Size

According to the World Health Organization (WHO), cancer is a leading cause of death worldwide, with an estimated 10 million deaths in 2020 alone. Breast, prostate, lung, and liver cancers are among the most prevalent types, impacting millions of lives each year. The incidence of cancer continues to rise, with an increasing burden on healthcare systems and communities worldwide.

### Solution

The primary function of the model is to predict various types of cancer, such as Breast, Prostate, Lung, and Liver Cancer, based on gene expression data. By analyzing patterns in gene expression profiles, the model can classify cancer samples into different categories, enabling early detection and diagnosis. The model generates output predictions for cancer types based on the input gene expression data, providing insights into the likelihood of different cancer types for each sample, and helping healthcare professionals in diagnosis and treatment planning.

#### Opportunity Size

With millions of people diagnosed with cancer each year, particularly breast, prostate, lung, and liver cancer, the potential impact of accurate and early detection cannot be overstated. By deploying this model widely, healthcare systems can potentially save countless lives, reduce treatment costs, and alleviate the burden on patients and their families. Additionally, advancements in cancer prediction can spur further research and development in oncology, leading to better treatments and outcomes for patients globally.


## Data Set

The original data can be found [here](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq).

### Data Description

The model utilizes RNA-seq gene expression data, where each row represents a cancer sample, and each column represents gene count values. This data provides valuable information about the activity levels of genes in cancerous tissues, which can be indicative of specific cancer types.

## Brief Description of the Three Methods Used

To address this problem statement, I have employed three different models: Logistic Regression, Decision Tree, Random Forest, and SVM.

### Logistic Regression

Logistic regression is the appropriate analysis to conduct when the dependent variable is dichotomous (binary). Like all regression analyses, logistic regression is predictive. It is used to describe data and explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval, or ratio-level independent variables.

### Decision Tree

A decision tree is a flowchart-like tree structure where each internal node denotes a feature, branches denote rules, and leaf nodes denote the result of the algorithm. It is a versatile supervised machine-learning algorithm used for both classification and regression problems.

### Support Vector Machine

A support vector machine (SVM) is a machine learning algorithm that uses supervised learning models to solve complex classification, regression, and outlier detection problems by performing optimal data transformations that determine boundaries between data points based on predefined classes, labels, or outputs. SVMs are widely adopted across disciplines such as healthcare, natural language processing, signal processing applications, and speech & image recognition fields.


## Experimental Results

### Data Analysis

#### Correction Map

A correction map is commonly used in predictive modeling tasks, particularly in machine learning, to assess the relationship between different variables and the target variable of interest. In this case, the target variable is whether a tumor is benign (not causing cancer) or malignant (causing cancer). So, it's crucial to understand the correlations between various features (such as tumor size, shape, texture, etc.) and the tumor's classification. A correction map helps visualize these correlations by representing them with colors. Lighter colors typically signify a high correlation, while darker colors indicate a weaker or negative correlation.


#### Scatter Plot Matrix

It is a graphical representation that displays pairwise relationships between variables in a dataset. It consists of multiple scatter plots arranged in a grid format, where each plot shows the relationship between two variables. In this context, the scatter plot matrix will focus on the "mean" columns of the dataset, specifically in relation to the diagnosis column, which likely represents whether a tumor is benign or malignant.

**Observation**: Notable linear patterns are observed in the scatter plot matrix, particularly in the first row's third and fourth graphs, and the third row's first graph, indicating high correlation. The nearly perfect linear relationships between radius, perimeter, and area attributes suggest multicollinearity, implying redundancy in the information they provide. Similarly, concavity, concave points, and compactness exhibit strong linear associations, potentially indicating multicollinearity among these variables. Addressing multicollinearity is crucial for accurate model interpretation and predictive performance.


### Logistic Regression

#### Performance Metrics:

- True Positive (TP): 110 cases are correctly identified.
- True Negative (TN): 2 are correctly rejected.
- False Negative (FN): 54 are incorrectly rejected.
- False Positive (FP): 5 are incorrectly identified.

**Observation**: The algorithm predicts 110 true cases, 5 false cases, and 2 cases are incorrectly rejected, and 54 cases are correctly rejected.
**Accuracy Score**: 95.90%.

### Decision Tree

#### Performance Metrics:

- True Positive (TP): 105 cases are correctly identified.
- True Negative (TN): 9 are correctly rejected.
- False Negative (FN): 47 are incorrectly rejected.
- False Positive (FP): 10 are incorrectly identified.

**Observation**: Accuracy score is 88.88%, which is less. Hence, the decision tree is not a good model for this prediction.

### SVM

#### Performance Metrics:

- True Positive (TP): 112 cases are correctly identified.
- True Negative (TN): 53 are correctly rejected.
- False Negative (FN): 3 are incorrectly rejected.
- False Positive (FP): 3 are incorrectly identified.

**Observation**: We achieved the best accuracy with SVM, which is 96.4%. SVM performing best compared to the other two.


## Discussion of Results

### Performance Analysis

#### Model Evaluation

The following metrics are used to evaluate the model: accuracy, precision, recall, F1 score, confusion matrix, and ROC Curve.

#### Observations

- Each model's performance is evaluated based on its ability to correctly classify instances of benign and malignant tumors. The metrics used include True Positive (TP), True Negative (TN), False Negative (FN), and False Positive (FP). These metrics provide insights into how well the model is distinguishing between the two classes.

- The accuracy score of each model is calculated as the percentage of correctly classified instances out of the total instances. It provides a general overview of the model's performance on the test dataset.

- By comparing the accuracy scores and performance metrics of different models, we can determine which model performs best for the given prediction task. In this case, SVM achieves the highest accuracy score of 96.4%, indicating that it outperforms Logistic Regression and Decision Tree models in predicting tumor classification.

**Support Vector Machines (SVMs)**

SVMs offer several advantages that make them a popular choice for various machine-learning tasks:

- They perform well even when dealing with data that has a large number of features, making them suitable for tasks like text classification or image recognition where the data is complex.
- They remain effective even when there are more features than samples in the dataset, which is not the case for some other algorithms.
- They are memory-efficient because they only use a subset of training points called support vectors, which helps save computational resources.
- They provide flexibility in choosing different kernel functions, allowing them to handle different types of data and learn complex decision boundaries.

#### Results

Support Vector Machine (SVM) achieved an overall accuracy of 96%, with precision and recall.

**Author**: Shilpa Gopal
**Copyright**: © 2023 Shilpa Gopal. All Rights Reserved.

  
