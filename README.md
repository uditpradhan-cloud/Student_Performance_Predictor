# AI-Driven Student Performance Prediction System

This project is an AI-driven machine learning system developed to predict a student's final mathematics grade (`G3`) using demographic, educational, behavioral, family, and lifestyle-related information. The project was developed as a submission for the IBM PBEL program.

The system uses the **Student Performance** dataset and explores multiple regression-based machine learning approaches to determine how well student-related attributes can be used to estimate final academic performance. The primary prediction workflow intentionally excludes the earlier academic grades (`G1` and `G2`) so that the model focuses on other student characteristics rather than relying directly on previous grades.

The project includes exploratory data analysis, data preprocessing, categorical feature encoding, machine learning model comparison, hyperparameter experimentation, cross-validation, feature-importance analysis, and final model training.

The final trained model is saved as a reusable `.pkl` file and is accompanied by a Python command-line application that accepts student information interactively and produces a predicted final grade along with a performance-level classification.

The model is intended as a **predictive estimation tool**, not as a guaranteed measure of a student's future academic result. The feature-importance analysis indicates which variables the trained model relied on most within the dataset, but it does not establish that those variables directly cause changes in student performance.

## Index

| File                                                                                         | Description                                                                                                                                                                                                                    |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`README.md`](README.md)                                                                     | Project documentation containing an overview of the system, repository guide, methodology, results, setup instructions, limitations, and future improvements.                                                                  |
| [`student_performance_prediction(1).ipynb`](student_performance_prediction%281%29.ipynb)     | Main Jupyter Notebook containing data loading, exploratory data analysis, preprocessing, feature engineering/encoding, model experiments, evaluation, cross-validation, feature-importance analysis, and final model training. |
| [`student-mat(2).csv`](student-mat%282%29.csv)                                               | Student mathematics performance dataset used as the source data for the machine learning experiments and final model.                                                                                                          |
| [`predict_student.py`](predict_student.py)                                                   | Command-line prediction program that loads the final trained model, collects student information interactively, predicts the final grade (`G3`), and assigns a performance level.                                              |
| [`student_performance_model_baseline(2).pkl`](student_performance_model_baseline%282%29.pkl) | Serialized baseline trained machine learning model produced during the project's model-development process.                                                                                                                    |
| [`student_performance_model_v2_g1_g2.pkl`](student_performance_model_v2_g1_g2.pkl)           | Serialized Version 2 model based on the experimental prediction approach that includes `G1` and `G2` as input features for predicting `G3`.                                                                                    |
| [`student_performance_model_final.pkl`](student_performance_model_final.pkl)                 | Serialized final Random Forest regression pipeline used by `predict_student.py` to generate student performance predictions.                                                                                                   |

---

## Project Objective

The primary objective of this project is to develop a machine learning system capable of estimating a student's final mathematics grade (`G3`) from available student-related information.

The project focuses on determining whether demographic, educational, behavioral, family-support, lifestyle, and other non-final-grade attributes can provide useful predictive information about final academic performance.

A secondary objective is to compare different regression approaches, evaluate their predictive performance, investigate feature importance, and package the resulting model into a reusable prediction application.
