Project Overview

- This repository contains the complete workflow for predicting diabetes using supervised machine-learning techniques. The analysis starts with rigorous data cleaning and preprocessing, then trains and evaluates three classifiers—Random Forest, Decision Tree, and Support Vector Machine—on a 70 / 30 train–test split. Key performance indicators are accuracy, precision, recall, F1 score, and specificity. A second round of training fine-tunes the tree-based models with a grid search over their most influential hyper-parameters.

Data

- The dataset (sourced from Kaggle) was purged of missing or implausible values—such as extreme ages or body-mass indices—and split into categorical and numerical subsets. Categorical features were converted with one-hot encoding; numerical features were normalised with Min–Max scaling.

Modelling Pipeline

- Each model is fitted on the same preprocessed feature matrix to ensure a fair comparison. Predictions on the held-out test set are scored with the scikit-learn classification report and a custom specificity function. Core metrics before hyper-parameter tuning are:

Model	Accuracy	Precision	Recall	F1	Specificity
Random Forest	0.969	0.939	0.681	0.789	0.996
Decision Tree	0.949	0.681	0.738	0.708	0.968
SVM	0.962	0.967	0.563	0.712	0.998

Random Forest offers the most balanced performance, combining high overall accuracy, the best F1 score, and near-perfect specificity.

Hyper-parameter Optimisation
A grid search using GridSearchCV explores depth, split criteria, and minimum sample thresholds for the tree-based learners. The tuned estimators achieve:

Model (tuned)	Accuracy	Precision	Recall	F1	Specificity
Random Forest	0.972	1.000	0.665	0.799	1.000
Decision Tree	0.971	0.970	0.679	0.799	0.998

Running the Code
Clone the repo, create a Python 3.11 environment, install dependencies from requirements.txt, and execute python main.py. The script reproduces all preprocessing, model training, evaluation, and hyper-parameter tuning steps, and prints a concise report to the console.

Interpretation of Grid-Search Gains
Grid search refines parameters that control model complexity and class weighting:

Random Forest – Raising the number of trees while restricting maximum depth reduces variance on minority-class predictions. Precision and specificity reach 100 %, meaning every positive and negative prediction on the test set is correct. The modest drop in recall (from 0.681 to 0.665) reflects a stricter decision boundary that sacrifices a handful of true positives in exchange for zero false positives.

Decision Tree – Allowing greater depth and a lower minimum split size captures subtler patterns, pushing precision from 0.681 to 0.970—an enormous reduction in false positives—while recall eases only slightly. Overall accuracy and F1 score rise accordingly, and specificity nearly matches that of the tuned forest.

In short, grid search finds a better bias-variance trade-off for both tree-based models, sharply improving their certainty in positive predictions without meaningfully diminishing their ability to find cases of diabetes.

References: scikit-learn user guide (v1.5) for model and tuning utilities; Kaggle for the original diabetes dataset.
