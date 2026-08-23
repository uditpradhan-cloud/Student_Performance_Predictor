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

## Methodology

The project follows a structured machine learning workflow, beginning with exploratory analysis of the student dataset and ending with a serialized prediction model that can be used through the command-line application.

### Machine Learning Workflow

```text
Student Performance Dataset
            │
            ▼
   Exploratory Data Analysis
            │
            ▼
   Feature / Target Selection
            │
            ▼
      Train-Test Split
            │
            ▼
   Data Preprocessing
   ┌──────────────────────┐
   │ Numerical Features   │
   │ Categorical Features │
   └──────────────────────┘
            │
            ▼
   Categorical Encoding
            │
            ▼
     Model Comparison
            │
            ▼
    Random Forest Model
            │
            ▼
 Model Evaluation & CV
            │
            ▼
 Hyperparameter Experiments
            │
            ▼
 Feature Importance Analysis
            │
            ▼
 Final Model Training
            │
            ▼
student_performance_model_final.pkl
            │
            ▼
     predict_student.py
            │
            ▼
     Predicted G3 Grade
```

## Exploratory Data Analysis

The notebook begins by examining the structure and characteristics of the dataset before model training.

The analysis includes inspection of:

* Dataset dimensions
* Data types
* Numerical and categorical variables
* Missing-value information
* Descriptive statistics
* Target variable distribution
* Relationships between relevant variables
* Student performance patterns
* Feature relationships and correlations

The exploratory phase is used to understand the dataset and guide subsequent preprocessing and modelling decisions.

## Target and Feature Selection

The prediction target is:

```text
G3
```

where `G3` represents the student's final mathematics grade.

For the primary model, the earlier-period grades `G1` and `G2` are excluded from the input feature set. This results in a prediction task based on the student's demographic, educational, behavioral, family, support, and lifestyle information.

The resulting primary modelling setup consists of:

* **30 input features**
* **1 target variable (`G3`)**

A separate Version 2 experiment was also performed using `G1` and `G2` as additional predictors. This experiment is kept separate from the primary model because it represents a different prediction scenario.

## Data Splitting

The primary modelling workflow uses a train-test split to create separate data for model development and evaluation.

The training data is used to fit the preprocessing and regression models, while the held-out test data is used to measure predictive performance on previously unseen examples.

The project also uses cross-validation during model evaluation to obtain a broader estimate of model performance across multiple data splits.

## Preprocessing

The dataset contains both numerical and categorical variables, so the notebook uses separate preprocessing strategies for each type.

### Numerical Features

Numerical features are passed through the preprocessing pipeline without categorical encoding.

### Categorical Features

Categorical features are transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

This converts categorical values into numerical representations that can be processed by the machine learning algorithms.

The preprocessing is incorporated into a `ColumnTransformer`, allowing the numerical and categorical transformations to be applied consistently.

The 30 original input features result in **56 processed features** after categorical encoding.

## Model Development

Several regression algorithms were evaluated during the project to determine which approach was most suitable for predicting `G3`.

The models explored include:

* Linear Regression
* Random Forest Regression
* Gradient Boosting Regression
* Tuned Random Forest
* Extra Trees Regression
* HistGradientBoosting Regression

Additional experiments were also conducted using:

* A Version 2 Random Forest model incorporating `G1` and `G2`
* A behavior-focused Random Forest approach
* A two-stage approach combining classification and regression for students with zero final grades

The initial model comparison identified **Random Forest Regression** as the strongest of the first three baseline models evaluated.

### Initial Model Comparison

| Model             |        MAE |       RMSE |         R² |
| ----------------- | ---------: | ---------: | ---------: |
| Linear Regression |     3.3953 |     4.1957 |     0.1415 |
| **Random Forest** | **2.9936** | **3.7485** | **0.3147** |
| Gradient Boosting |     3.1077 |     3.9299 |     0.2468 |

Lower MAE and RMSE indicate smaller prediction errors, while a higher R² indicates that more of the variation in the target is explained by the model.

Based on these initial results, Random Forest was selected for further experimentation.

## Hyperparameter Tuning

The Random Forest model was subsequently tuned using `GridSearchCV`.

The search evaluated **108 parameter combinations**.

The best configuration identified during the search was:

```text
n_estimators = 300
max_depth = 15
min_samples_split = 5
min_samples_leaf = 1
```

However, the tuned model did not improve upon the original Random Forest on the held-out test set.

| Model                  |        MAE |       RMSE |         R² |
| ---------------------- | ---------: | ---------: | ---------: |
| Original Random Forest | **2.9936** | **3.7485** | **0.3147** |
| Tuned Random Forest    |     3.0042 |     3.7582 |     0.3112 |

Therefore, hyperparameter tuning was retained as part of the experimentation process rather than being presented as an improvement over the original model.

## Cross-Validation

The project also evaluates model performance using **5-fold cross-validation**.

For each fold, the model is trained using a portion of the available data and evaluated on the remaining portion. The resulting scores are then averaged to provide a more robust estimate of generalization performance.

For the Random Forest model, the cross-validation results reported by the project are:

* **Mean MAE:** 2.99 marks
* **Mean RMSE:** 4.00 marks
* **Mean R²:** 0.216
* **Predictions within ±2 marks:** 46.33%

These values are also displayed by the project's command-line prediction script.

## Feature Importance Analysis

After training the Random Forest model, feature importance was examined to identify which variables contributed most strongly to the model's predictions.

The highest-ranked features included:

| Rank | Feature      |
| ---: | ------------ |
|    1 | `absences`   |
|    2 | `failures`   |
|    3 | `health`     |
|    4 | `goout`      |
|    5 | `age`        |
|    6 | `studytime`  |
|    7 | `freetime`   |
|    8 | `traveltime` |
|    9 | `Walc`       |
|   10 | `Fedu`       |

The analysis indicates that `absences` and `failures` were particularly influential features in the trained Random Forest model.

These rankings should be interpreted as **model feature importance**, not causal relationships. A high feature-importance value means that the model relied more heavily on that feature when making predictions; it does not demonstrate that changing the feature would directly cause a change in a student's final grade.

## Final Model

After the experimentation and evaluation stages, a final Random Forest regression pipeline was trained and serialized as:

```text
student_performance_model_final.pkl
```

The final model is trained using the complete available feature/target dataset after preprocessing, allowing the saved model to be reused for predictions.

The accompanying `predict_student.py` application loads this model with `joblib` and uses it to generate predictions from newly entered student information.

This separation between **model development in the notebook** and **model inference through the Python script** makes the trained model reusable without requiring the notebook to be executed each time a prediction is required.

## Evaluation & Results

Several regression models were explored during the project to identify a suitable approach for predicting the final grade (`G3`).

### Models Evaluated

The project experimented with:

- Linear Regression
- Random Forest Regression
- Gradient Boosting Regression
- Extra Trees Regression
- HistGradientBoosting Regression
- Tuned Random Forest Regression

Additional experimental approaches were also explored, including a Version 2 Random Forest using `G1` and `G2`, and a two-stage classification + regression approach.

### Model Comparison

The initial model comparison produced the following results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 3.3953 | 4.1957 | 0.1415 |
| **Random Forest** | **2.9936** | **3.7485** | **0.3147** |
| Gradient Boosting | 3.1077 | 3.9299 | 0.2468 |

Among the initial models, **Random Forest Regression** achieved the lowest MAE and RMSE and the highest R², making it the selected approach for the primary prediction system.

### Cross-Validation

The selected Random Forest model was further evaluated using 5-fold cross-validation:

- **MAE:** 2.99 marks
- **RMSE:** 4.00 marks
- **R²:** 0.216
- **Predictions within ±2 marks:** 46.33%

The results indicate useful but limited-to-moderate predictive performance. Predictions should therefore be treated as estimates rather than guaranteed final grades.

## Technologies Used

- **Python** — Core programming language
- **Jupyter Notebook** — Data analysis and model development
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Seaborn** — Exploratory data visualization
- **Scikit-learn** — Preprocessing, model training, evaluation, and cross-validation
- **Joblib** — Saving and loading trained machine learning models

## Project Structure

```text
Student_Performance_Predictor/
│
├── README.md
├── student_performance_prediction(1).ipynb
├── student-mat(2).csv
├── predict_student.py
│
├── student_performance_model_baseline(2).pkl
├── student_performance_model_v2_g1_g2.pkl
└── student_performance_model_final.pkl
```

## How to Run

### 1. Clone the Repository

````markdown
```bash
git clone <your-repository-url>
cd Student_Performance_Predictor
````

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn jupyter
```

### 3. Run the Prediction Program

The trained final model is already included in the repository. Run:

```bash
python predict_student.py
```

The program will interactively ask for student information such as demographic details, education, family support, study habits, lifestyle, health, and absences.

It then displays:

* Predicted final grade (`G3`) out of 20
* Performance level

The prediction script loads the saved `student_performance_model_final.pkl` model automatically. 

### 4. Explore the Notebook

To reproduce or explore the model-development workflow:

```bash
jupyter notebook
```

Open:

```text
student_performance_prediction(1).ipynb
```

The notebook contains the data analysis, preprocessing, model experiments, evaluation, and final model training workflow.