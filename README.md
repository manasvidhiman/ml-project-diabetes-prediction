# Diabetes Risk Prediction

A machine learning project that estimates a person's risk of diabetes from eight health measurements. It includes a complete pipeline (data cleaning, preprocessing, model training, testing) and a small web app where anyone can enter their values and see an estimated risk percentage.

> **Disclaimer:** This is a student learning project. It is **not** a medical tool and must not be used to diagnose or treat any condition. Please see a doctor for real medical advice.

**Author:** [Manasvi Dhiman]
**Course / Institution:** [CSE,ABES Engineering College]
**Date:** [1st Oct 2026]

link: https://ml-project-diabetes-prediction-2.streamlit.app/

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
5. [Results](#results)
6. [Installation](#installation)
7. [How to Run](#how-to-run)
8. [The Web App](#the-web-app)
9. [Exporting the Model to XML (PMML)](#exporting-the-model-to-xml-pmml)
10. [Limitations](#limitations)
11. [Technologies Used](#technologies-used)

---

## Project Overview

The goal is to predict whether a person is likely to have diabetes (a yes/no classification problem) using common health measurements such as glucose level, BMI, and age.

The project covers the full machine learning workflow:

1. Exploring and cleaning the data
2. Visualising patterns with charts
3. Preprocessing the data for modelling
4. Training and comparing three models
5. Testing the best model in several ways
6. Finding which factors matter most
7. Building an app that uses the trained model

---

## Dataset

**Pima Indians Diabetes Database**, originally from the National Institute of Diabetes and Digestive and Kidney Diseases. It is available on Kaggle and the UCI Machine Learning Repository.

- **768 records**, one per person
- **8 input features** and **1 target column**
- Class balance: 500 without diabetes (0), 268 with diabetes (1)

| Column | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | A score of diabetes risk from family history |
| Age | Age in years |
| **Outcome** | **Target: 0 = no diabetes, 1 = diabetes** |

**Data quality note:** In this dataset, a value of 0 in Glucose, BloodPressure, SkinThickness, Insulin and BMI means "not measured", not a real reading. These zeros are treated as missing values and replaced with the column median.

---

## Project Structure

```
mlproject/
├── diabetes.csv              # The dataset
├── preprocessing.py          # Cleans, splits and scales the data
├── train_models.py           # Trains and compares three models
├── test_model.py             # Tests the best model
├── feature_importance.py     # Shows which features matter most
├── export_pmml.py            # Exports the model to an XML (PMML) file
├── app.py                    # Streamlit web app
├── all_in_one.py             # Optional: the whole pipeline in a single script
├── .streamlit/
│   └── config.toml           # App theme settings
├── results/                  # Saved tables, charts and reports
├── best_model.pkl            # Saved trained model (created by the scripts)
├── scaler.pkl                # Saved scaler (created by preprocessing.py)
├── diabetes_model.pmml       # Model exported as XML (created by export_pmml.py)
└── README.md
```

Files such as `best_model.pkl`, `scaler.pkl`, the `X_*.csv` / `y_*.csv` files and the `results/` folder are created when you run the scripts.

---

## How It Works

### 1. Cleaning and preprocessing (`preprocessing.py`)

- Removes duplicate rows
- Converts the "not measured" zeros into missing values
- Splits the data into **80% training** and **20% test** sets, keeping the same share of diabetes cases in both (stratified split)
- Fills missing values with the **median of the training data**
- Scales all columns with `StandardScaler`, fitted on the training data only to avoid leaking information from the test set

### 2. Model training (`train_models.py`)

Three classification models are trained and compared:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

All use `class_weight="balanced"` because the classes are uneven. The best model is selected by **recall**, because in disease detection, missing a real case is worse than a false alarm.

### 3. Testing (`test_model.py`)

- Accuracy, precision, recall and F1 on the hidden test data
- ROC-AUC and ROC curve
- 5-fold cross-validation to check that the result is stable
- A prediction for a brand-new person to confirm the full pipeline works

### 4. Feature importance (`feature_importance.py`)

A Random Forest is used to rank which columns matter most for the prediction, shown as a bar chart.

---

## Results

Results on the hidden test set (Logistic Regression, the selected model):

| Metric | Score |
|---|---|
| Accuracy | 0.73 |
| Precision | 0.60 |
| Recall | 0.70 |
| F1 score | 0.65 |
| ROC-AUC | 0.81 |

5-fold cross-validation gives an average accuracy of about 0.75 and an average recall of about 0.70, with similar scores across all rounds, so the result is stable.

**Most important features** (from the Random Forest): Glucose is the strongest predictor, followed by BMI, Age and the Diabetes Pedigree Function.

These results are in line with published results for this dataset, where accuracy of roughly 75 to 80% is typical. Exact numbers may differ slightly from run to run or between library versions.

---

## Installation

**Requirements:** Python 3.10 or newer.

1. Download or clone this project folder.
2. (Recommended) Create and activate a virtual environment.
3. Install the libraries:

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

4. Make sure `diabetes.csv` is in the project folder.

---

## How to Run

Run the scripts **in this order**, because each one needs the output of the one before it:

```
python preprocessing.py
python train_models.py
python test_model.py
```

Optional extras:

```
python feature_importance.py
python export_pmml.py
```

**Shortcut:** `all_in_one.py` runs cleaning, splitting, training and saving in one go:

```
python all_in_one.py
```

Use either the step-by-step route or `all_in_one.py`, not a mix of both, so that `best_model.pkl` and `scaler.pkl` always come from the same run.

---

## The Web App

The app lets a user enter the eight health values and shows an estimated diabetes risk as a percentage, labelled low (under 30%), medium (30 to 60%) or high (over 60%).

To start it, run this in the terminal from the project folder:

```
streamlit run app.py
```

Then open the link shown in the terminal (usually `http://localhost:8501`). Run the pipeline first, so that `best_model.pkl` and `scaler.pkl` exist.

---

## Exporting the Model to XML (PMML)

`export_pmml.py` saves the trained Logistic Regression model as `diabetes_model.pmml`, an XML file in the standard PMML format. The scaling is built into the model's weights, so the file takes original values (for example Glucose = 150) directly.

```
python export_pmml.py
```

This export supports the Logistic Regression model only.

---

## Limitations

- **Small dataset:** only 768 records, so the test set is just 154 rows and scores can vary by a few percent between splits.
- **Missing measurements:** many Insulin and SkinThickness values were not recorded and had to be filled in with estimates.
- **Narrow population:** the data comes from a specific group of women of Pima Indian heritage aged 21 and over, so the model may not be accurate for other groups.
- **Risk estimate, not a diagnosis:** the model outputs a probability based on 8 measurements and cannot replace a medical test.

---

## Technologies Used

- Python
- pandas and NumPy for data handling
- scikit-learn for preprocessing, models and evaluation
- matplotlib and seaborn for charts
- joblib for saving the model
- Streamlit for the web app
