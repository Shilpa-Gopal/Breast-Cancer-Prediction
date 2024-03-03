# Predictive Testing for Breast Cancer Diagnosis: A Machine Learning Approach

## Index: 

1. [Model Overview: Predictive Testing for Breast Cancer Diagnosis](#model-overview-predictive-testing-for-breast-cancer-diagnosis)
2. [Problem Statement](#problem-statement)
3. [Problem Size](#problem-size)
4. [Solution](#solution)
5. [Opportunity Size](#opportunity-size)
6. [Data Set](#data-set)
7. [Data Description](#data-description)
8. [Classification Techniques](#classification-techniques)
9. [Model Selection](#model-selection)
10. [Technology Stack](#technology-stack)
11. [Performance Analysis](#performance-analysis)
12. [Results](#results)
13. [Benefits of the Breast Cancer Classification Model](#benefits-of-the-breast-cancer-classification-model)


## Model Overview: Predictive Testing for Breast Cancer Diagnosis

### Problem Statement:
Breast cancer is the most common cancer among women in Wisconsin, regardless of race, accounting for nearly one-third of all cancers diagnosed among women. Identifying breast cancers in their early stages is crucial to controlling their occurrence. Manual methods by radiologists often fail due to the similarity in appearance of breast masses and microcalcifications with background segmentation, making accurate diagnosis challenging. Automated systems are needed for early detection to assist radiologists in accurate diagnosis and subsequent patient treatment.

### Problem Size:
According to the CDC, about 240,000 women in the US are diagnosed with breast cancer each year, along with about 2,100 men. Approximately 42,000 women and 500 men in the US die from breast cancer annually. The 5-year relative survival rate for women in the US with non-metastatic invasive breast cancer is 91%, while the 10-year relative survival rate is 85%.

### Solution:
The aim of AI models in breast cancer classification, particularly distinguishing between Benign (B) and Malignant (M) tumors, is to provide accurate and reliable diagnoses. For Benign tumors (B), the model aims to correctly identify them as non-cancerous growths to ensure patients receive appropriate follow-up care without unnecessary anxiety or invasive treatments. For Malignant tumors (M), the model aims to accurately detect and classify them as cancerous, crucial for timely intervention and treatment planning, significantly impacting patient outcomes and survival rates.

### Opportunity Size:
The opportunity size for addressing the problem of breast cancer classification is substantial. Breast cancer is one of the most common cancers globally, affecting millions of individuals each year. Accurate diagnosis and timely treatment are critical for improving patient outcomes and reducing mortality rates. By developing AI models capable of accurately classifying breast tumors, we have the opportunity to positively impact the healthcare industry and patient care on a significant scale.

## Data Set:
### Breast Cancer Wisconsin data set 
This link provides a detail overview of the Data Point, its columns: [Breast Cancer Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

### Data Description:
The Breast Cancer Wisconsin dataset contains information about breast tumor samples, including various features used for classification. Key data points include ID (a unique identifier for each tumor sample), Diagnosis (classification of the tumor as Malignant (M) or Benign (B)), and various numerical features representing tumor characteristics such as Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Symmetry, and Fractal Dimension.

## Classification Techniques
### Model Selection:
Classification is performed using various algorithms including Logistic Regression, Random Forest, Decision Tree, K Nearest Neighbors (KNN), Support Vector Machines (SVM), and Naive Bayes. The task is to predict Malignant (M) and Benign (B) tumors.

### Technology Stack:
Python, Jupyter Notebook, Pandas, NumPy, Matplotlib

## Performance Analysis

### Results:
SVM achieved the highest accuracy of 97%, with recall and precision also at 97%, outperforming other algorithms.

## Benefits of the Breast Cancer Classification Model

This model can be incredibly helpful in several ways:
- Early Detection
- Treatment Planning
- Resource Optimization
- Patient Counseling
- Research and Development
