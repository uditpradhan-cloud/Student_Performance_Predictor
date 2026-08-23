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

## Dataset

The project uses the **Student Performance** mathematics dataset stored in `student-mat(2).csv`.

The dataset contains information about students' demographic background, family circumstances, educational environment, study habits, social activities, lifestyle, and academic performance.

### Dataset Overview

| Property             | Details                                    |
| -------------------- | ------------------------------------------ |
| Dataset              | Student Performance — Mathematics          |
| Records              | 395 students                               |
| Original columns     | 33                                         |
| Prediction target    | `G3`                                       |
| Target meaning       | Final mathematics grade                    |
| Grade scale          | 0–20                                       |
| Primary model inputs | 30 features                                |
| `G1` / `G2`          | Excluded from the primary prediction model |

The final grade, `G3`, is treated as the regression target. The primary model does not use `G1` or `G2` as input features, allowing the project to investigate prediction based on other student characteristics rather than directly using earlier-period grades.

## Features

The original dataset contains several categories of information about each student.

### Demographic and Personal Information

* `school` — Student's school
* `sex` — Student's gender
* `age` — Student's age
* `address` — Urban or rural residence
* `famsize` — Family size
* `Pstatus` — Parent cohabitation status

### Parent and Family Information

* `Medu` — Mother's education level
* `Fedu` — Father's education level
* `Mjob` — Mother's job
* `Fjob` — Father's job
* `guardian` — Student's guardian
* `famrel` — Quality of family relationships

### School and Academic-Related Information

* `reason` — Reason for choosing the school
* `traveltime` — Travel time to school
* `studytime` — Weekly study time
* `failures` — Number of past class failures
* `schoolsup` — Extra educational support
* `famsup` — Family educational support
* `paid` — Extra paid classes
* `higher` — Desire for higher education

### Activities and Lifestyle Information

* `activities` — Participation in extracurricular activities
* `nursery` — Attendance at nursery school
* `internet` — Internet access at home
* `romantic` — Romantic relationship status
* `freetime` — Amount of free time
* `goout` — Frequency of going out
* `Dalc` — Workday alcohol consumption
* `Walc` — Weekend alcohol consumption
* `health` — Current health status
* `absences` — Number of school absences

### Target and Excluded Academic Grades

* `G1` — First-period grade
* `G2` — Second-period grade
* `G3` — Final grade and prediction target

For the **primary prediction system**, `G1` and `G2` are excluded from the input features. They were explored separately in a Version 2 experiment and are represented by the `student_performance_model_v2_g1_g2.pkl` model artifact.

## Feature Types

The notebook separates the 30 primary input features into:

* **17 categorical features**
* **13 numerical features**

Categorical features are encoded before being passed to the machine learning models, while numerical features are retained as numerical values.

After categorical encoding, the 30 original input features are transformed into **56 processed features** for model training.

This preprocessing approach allows the regression models to work with both numerical and categorical student information within a unified machine learning pipeline.
