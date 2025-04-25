# Cars-Dataset-Regression-Analysis-and-Model-Selection
This repository contains code and data for comparing various regression models to predict car prices using a dataset from the YallaMotor website. The dataset includes multiple features such as car name, price, engine capacity, cylinder count, horsepower, top speed, number of seats, brand, and country of sale. The goal is to evaluate the effectiveness of different regression techniques in predicting car prices and performance metrics.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
  - [Cleaning](#cleaning)
  - [Encoding](#encoding)
  - [Scaling](#scaling)
- [Models](#models)
- [Results](#results)
- [Conclusion](#Conclusion)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Technical Report](#technical-report)

## Table of Contents
Introduction
Dataset
Preprocessing
Cleaning
Encoding
Scaling
Models
Results
Conclusion
Usage
Dependencies
Technical Report


## Introduction
This project aims to compare various regression models to predict car prices using a comprehensive dataset from the YallaMotor website. The analysis evaluates the performance of different regression techniques and identifies the most suitable models for accurate car price prediction.

## Dataset
The dataset contains the following columns:

- Car Name: The name of the car.
- Price: The price of the car.
- Engine Capacity: The car's engine capacity.
- Cylinder: The car's cylinder count.
- Horse Power: The car's horsepower.
- Top Speed: The car's top speed.
- Seats: The number of seats in the car.
- Brand: The car's brand.
- Country: The country where the site sells this car.

## Preprocessing

The dataset underwent several preprocessing steps to ensure data quality and suitability for modeling. The preprocessing is divided into three main subsections: Cleaning, Encoding, and Scaling.

### Cleaning
Data cleaning is a critical step in the preprocessing pipeline to ensure the quality and reliability of the dataset. This process involved:

- **Handling Missing Values**: Replacing missing data with appropriate values, such as the mean, median, or a placeholder, to avoid introducing biases into the analysis.
- **Correcting Erroneous Entries**: Identifying and rectifying inconsistent or inaccurate data points, such as outliers or invalid values.
- **Standardizing Data Formats**: Ensuring all data is in a consistent format (e.g., date formats, numerical precision) to facilitate seamless analysis and modeling.

### Encoding
Encoding was applied to convert categorical variables into a numerical format that is suitable for machine learning models. This process included:

- **One-Hot Encoding**: Creating binary columns for each category in the dataset to preserve the categorical information without introducing ordinal relationships.
- **Label Encoding**: Assigning unique numerical labels to categories where applicable, ensuring the encoded values align with the problem context.

### Scaling
Scaling is an essential preprocessing step to normalize the dataset and ensure all features contribute equally to the model. This process included:

- **Feature Normalization**: Bringing features to a common scale to prevent larger numerical ranges from dominating the training process.
- **Standardization**: Transforming features to have a mean of zero and a standard deviation of one, which is particularly important for machine learning algorithms sensitive to feature magnitudes, such as gradient descent-based optimizers.
  
## Models
The following regression models were compared:

- Linear Regression
- Lasso Regression
- Ridge Regression
- Support Vector Regression (SVR)

## Results
The results of the regression models were evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) score. The models were compared to determine the most effective regression technique for predicting car prices.

## Conclusion
The analysis demonstrated that certain regression models outperformed others in predicting car prices. The results indicate that appropriate preprocessing and the choice of regression model are crucial for achieving accurate predictions. This comparison provides valuable insights into the strengths of different regression techniques when applied to complex datasets in the automotive domain.


## Usage
To run the code in this repository, follow these steps:

1. Clone the repository:
   ```bash

    git clone https://github.com/yazan6546/Regression-Analysis-and-Model-Selection.git
    cd Regression-Analysis-and-Model-Selection

2. Install the required dependencies:
   ```bash

    pip install -r requirements.txt

3. Run the data cleaning notebook:
   ```bash
    jupyter notebook cleaning.ipynb
Or open using your favorite IDE such as vscode.

4. Run the main analysis notebook:

jupyter notebook main.ipynb
This notebook will load the cleaned dataset, implement various regression models, and display the results.

## Dependencies
The following Python libraries are required to run the code:

- pandas
- numpy
- scikit-learn
- category_encoders
- matplotlib
- IPython
- Jupyter Notebook

##  Technical Report
A detailed technical report is available in this repository, providing an in-depth analysis of the data preprocessing steps, model selection, and evaluation metrics. The report includes visualizations and explanations of the results, offering valuable insights into the regression analysis and model selection process. You can check the technical report here..
