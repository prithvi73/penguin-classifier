# Palmer Penguins Classifier 🐧

A Supervised Machine Learning classification pipeline that predicts the exact species of a penguin (Adelie, Chinstrap, or Gentoo) based on physical biometric measurements.

## Overview
This project demonstrates foundational Machine Learning operations using the classic Palmer Penguins dataset. The pipeline ingests raw CSV data, preprocesses numeric features, splits the data to prevent algorithmic memorization, and trains a classification model to identify the mathematical boundaries between species.

## Technical Flow
* **Data Preprocessing (Pandas):** Ingests remote CSV data, drops missing values, and isolates numeric features (bill length, bill depth, flipper length, body mass) from the target label (species).
* **Data Splitting (Scikit-learn):** Utilizes `train_test_split` to partition the dataset into an 80% training matrix and a 20% testing matrix.
* **Model Training:** Implements a `DecisionTreeClassifier` fitted to the training data.
* **Evaluation & Inference:** Evaluates the model against the hidden test dataset (achieving 100% accuracy) and successfully classifies synthetic biometric data during final inference.

## Tech Stack
* Python
* Pandas
* Scikit-learn