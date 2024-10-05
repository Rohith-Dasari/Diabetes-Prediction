# Diabetes Prediction Project

This project is a component of a larger initiative aimed at gaining insights from Electronic Health Records (EHR). Specifically, this segment focuses on predicting diabetes outcomes using a dataset containing various health-related features. The analysis includes data preprocessing, feature scaling, and model evaluation to enhance the prediction accuracy of diabetes cases.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing Steps](#preprocessing-steps)
- [Data Visualization](#data-visualization)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Project Overview

The primary objective of this project is to build a predictive model that can accurately determine whether an individual has diabetes based on a set of input features. This project employs K-Nearest Neighbors (KNN) for classification and aims to achieve high accuracy through thorough data analysis and preprocessing techniques.

## Dataset

The dataset used for this project is the **Diabetes Prediction Dataset**, which includes the following features:
- Age
- BMI (Body Mass Index)
- HbA1c Level
- Blood Glucose Level
- Hypertension
- Heart Disease
- Smoking History
- Gender
- Diabetes (Target Variable)

## Preprocessing Steps

The preprocessing steps involved in this project include:

1. **Handling Missing Values**: Missing values in the dataset were addressed using the KNN imputation technique, ensuring that the dataset remained robust for analysis.
  
2. **Encoding Categorical Variables**: Categorical variables such as gender and smoking history were converted into numerical values using Label Encoding to facilitate model training.

3. **Feature Scaling**: Standardization of the feature set was performed using `StandardScaler`. This transformation improved the model’s accuracy from **95% to 96%**, significantly enhancing the performance of the KNN classifier.

4. **Splitting Data**: The dataset was divided into training and testing sets using an 80-20 split to ensure effective model validation.

## Data Visualization

Throughout the analysis, multiple visualizations were created to gain insights into the dataset and the relationships between features:
- **Scatter Plots**: To explore relationships between continuous variables and the target variable.
- **Correlation Heatmaps**: To visualize the correlation between different features, helping identify significant relationships.
- **Box Plots**: To examine the distribution of various health metrics across diabetes and non-diabetes groups, revealing potential outliers and trends.

## Model Training and Evaluation

The KNN model was trained on the preprocessed training dataset. Various values of K (from 1 to 20) were evaluated to determine the optimal K value for the best performance. 

- **Confusion Matrix**: A confusion matrix was generated for the best K value, providing insights into the model's true positive and false positive predictions.
  
- **ROC Curve**: The Receiver Operating Characteristic (ROC) curve was plotted to illustrate the trade-off between sensitivity and specificity for the KNN model.

- **Accuracy Visualization**: A graph displaying the accuracy of the KNN classifier across different K values was plotted, enabling a clear visualization of model performance.

## Results

The analysis yielded the following key results:
- The KNN classifier achieved an optimal accuracy of **96%** after feature scaling.
- The confusion matrix indicated the model's proficiency in distinguishing between diabetes and non-diabetes cases.
- The ROC curve highlighted the model’s effectiveness, showcasing a high area under the curve (AUC) score.

## Conclusion

The project successfully demonstrated the importance of data preprocessing and scaling in improving the performance of machine learning models. The KNN classifier achieved a high accuracy rate, showcasing its effectiveness in predicting diabetes outcomes based on various health metrics.

## Future Work

Future improvements to this project could include:
- Exploring other classification algorithms to compare performance.
- Implementing hyperparameter tuning for the KNN model to further enhance accuracy.
- Expanding the dataset to include more diverse samples for better generalization.
