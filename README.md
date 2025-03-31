 Exploratory Data Analysis (EDA) on Mobile Dataset

Overview

This project performs Exploratory Data Analysis (EDA) on a mobile dataset using Python. The goal is to understand the dataset, detect patterns, and gain insights that can be useful for further analysis or machine learning models.

Dataset

The dataset should contain details about mobile specifications, such as:

Brand

Model

Price

RAM

Storage

Battery Capacity

Camera Specifications

Processor

Screen Size

Requirements

Before running the analysis, install the required dependencies:

pip install pandas numpy matplotlib seaborn

Steps for EDA

1. Load the Dataset

Ensure the dataset is in CSV format and load it using Pandas:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("mobile_data.csv")  # Update with the correct file path

2. Understand the Data

Get a basic understanding of the dataset:

print(df.head())  # First few rows
print(df.info())  # Column details
print(df.describe())  # Summary statistics

3. Handle Missing Values & Duplicates

print(df.isnull().sum())  # Check for missing values
df.drop_duplicates(inplace=True)  # Remove duplicate rows
df.fillna(df.median(), inplace=True)  # Fill missing values

4. Univariate Analysis (Single Variable)

plt.figure(figsize=(10, 5))
sns.histplot(df['price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.show()

5. Bivariate Analysis (Relationships Between Features)

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['ram'], y=df['price'])
plt.title("RAM vs Price")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

6. Categorical Analysis

sns.countplot(x=df['brand'])
plt.title("Mobile Brands Count")
plt.xticks(rotation=45)
plt.show()

Additional Insights

Outlier Detection: sns.boxplot(x=df['price'])

Feature Engineering: df['price_per_GB'] = df['price'] / df['storage']

Advanced Visualizations: Pairplots, violin plots, and more

Conclusion

This EDA helps in understanding the trends in the mobile dataset, identifying key factors affecting pricing, and preparing the dataset for further predictive modeling.
 
