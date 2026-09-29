# Air Quality Prediction using GA-KELM

A Machine learning project for **air-pollutant prediction** using a **Genetic Algorithm-based Kernel Extreme Learning Machine (GA-KELM)**. The project uses Ahmedabad air-quality data and evaluates GA-KELM against **Support Vector Regression (SVR)** as a baseline. It also includes an experimental 10-day AQI forecasting demo.

This repository is developed as an **academic project** demonstrating data preprocessing, regression modelling, optimization, and model evaluation on environmental data.

## Project Overview

### Workflow

1. Load the Ahmedabad air-quality dataset (`Dataset.csv`).
2. Drop rows where the target (PM2.5) is missing.
3. Automatically exclude empty columns and select the pollutant measurements as input features.
4. Fill missing feature values using linear interpolation over time.
5. Wrap every model in a pipeline that min-max scales inputs and target (fitted on training data only).
6. Compare GA-KELM with a mean baseline, Ridge, and a **tuned** SVR using repeated 5-fold and time-series cross-validation (mean ± std).
7. Run an 80/20 hold-out test with predicted-vs-actual plots.
8. Report RMSE, MAE and R² in original units (µg/m³).
9. (Experimental) Validate and produce a 10-day AQI forecast, compared against a persistence baseline.

## Models

### GA-KELM

GA-KELM combines a **Kernel Extreme Learning Machine (KELM)** with a **Genetic Algorithm (GA)**. KELM performs nonlinear regression with an RBF kernel, and the GA searches for the best kernel width (`gamma`) and regularization constant (`C`) using an internal validation split of the training data (the test set is never used for tuning).

### SVR

**Support Vector Regression (SVR)**, with `C`, `gamma` and `epsilon` tuned by grid search, is used as the main baseline. A mean-value baseline and Ridge regression are also included.

## Dataset

The project uses an **Ahmedabad air-quality dataset**: 312 daily records from **2015-01-01 to 2015-11-08**.

### Target

- **PM2.5** (279 usable rows after removing missing targets)

### Input Features (9)

- NO
- NO2
- NOx
- CO
- SO2
- O3
- Benzene
- Toluene
- Xylene
- PM10 and NH3 are **not usable**: both columns are completely empty in this dataset and are excluded from modelling.

> The `AQI` and `AQI_Bucket` columns are excluded from the input features because they are derived from the pollutant values (target leakage). `AQI` is only used in the experimental forecasting section.

### Data Notes

- CO is identical to NO in about 98% of rows, which suggests a data-quality issue in the source data.
- AQI reaches values above 1000 in places, well beyond the usual 0-500 scale.

## Results

Predicting **PM2.5** (original units, µg/m³). Cross-validation values are mean ± std.

| Model | Repeated 5-fold RMSE | Time-series CV RMSE | Hold-out RMSE | Hold-out R² |
|---|---:|---:|---:|---:|
| Mean baseline | 46.13 ± 6.00 | 49.61 ± 9.02 | - | - |
| Ridge | 29.27 ± 2.78 | 33.07 ± 7.09 | - | - |
| SVR (tuned) | 29.03 ± 2.63 | 32.44 ± 9.22 | 26.93 | 0.666 |
| GA-KELM | 31.66 ± 7.03 | 33.82 ± 8.70 | 26.90 | 0.666 |

### Interpreting the results

- All learned models are far better than the mean baseline.
- On the hold-out split, GA-KELM and the tuned SVR are essentially tied.
- In cross-validation, GA-KELM is **not better** than a tuned SVR or Ridge, and it varies more between folds. On this small dataset GA-KELM is competitive but shows no clear advantage. Ridge matching the kernel models suggests the relationship is largely linear.
- Earlier "GA-KELM beats SVR" results came from an untuned SVR on a single split.

## Experimental: 10-Day AQI Forecast

The notebook also trains GA-KELM on 14 lagged AQI values, 7- and 14-day rolling means, and seasonal (sine/cosine) features, then forecasts 10 days past the end of the data (2015-11-09 onward).

**Validation:** the notebook tests the forecast on the last 40 readings against a persistence baseline ("tomorrow = today"). Errors are large (RMSE about 200 AQI points one day ahead, R² near 0) because AQI is very volatile in this period. The models beat persistence at most horizons, but only modestly, and GA-KELM does not beat Ridge.

**Limitations:** the forecast is recursive (each prediction feeds the next), so errors accumulate. Treat the future forecast as a demonstration only, not as a reliable prediction.

## Project Structure

~~~text
Air-Quality-Index-using--GA-KELM/
│
├── dataset/
│   └── Dataset.csv
│
├── models/                      # legacy files, not used by the notebook
│   ├── elm.npy
│   └── extension_weights.hdf5
│
├── notebooks/
│   └── AirQuality.ipynb
│
├── src/
│   └── GAKELM.py
│
├── .gitignore
├── requirements.txt
└── README.md
~~~

## Technologies Used

- Python 3.11
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Genetic Algorithm
- Kernel Extreme Learning Machine
- Support Vector Regression

## Installation

### 1. Clone the repository

~~~bash
git clone https://github.com/omer-farooq28/Air-Quality-Index-using--GA-KELM.git
cd Air-Quality-Index-using--GA-KELM
~~~

### 2. Create a Python 3.11 virtual environment

~~~bash
py -3.11 -m venv venv
~~~

Activate the environment on Windows:

~~~bash
venv\Scripts\activate
~~~

### 3. Install dependencies

~~~bash
py -3.11 -m pip install -r requirements.txt
~~~

### 4. Launch Jupyter Notebook

~~~bash
jupyter notebook
~~~

Open:

~~~text
notebooks/AirQuality.ipynb
~~~

Run the notebook cells in sequence to reproduce the preprocessing, training, evaluation, comparison, and forecasting workflow. The notebook expects the folder layout shown above (`../src` and `../dataset`).

## Evaluation Metrics

### Mean Squared Error (MSE)

MSE measures the average squared difference between actual and predicted target values.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE and expresses prediction error on the same scale as the evaluated target values.

The notebook reports both metrics on the normalized PM2.5 target, plus RMSE converted back to original units.

## Important Files

| File | Description |
|---|---|
| `dataset/Dataset.csv` | Ahmedabad air-quality dataset (Jan-Nov 2015) |
| `notebooks/AirQuality.ipynb` | Main notebook: cross-validated evaluation, hold-out test, validated forecast |
| `src/GAKELM.py` | GA-KELM implementation |
| `models/` | Legacy model files, not loaded by the current code |
| `requirements.txt` | Python dependencies |

## Limitations

- Small dataset: a single city and under one year of data (Jan-Nov 2015).
- GA-KELM shows no clear advantage over a tuned SVR or Ridge on this data.
- The forecasting section is a demo; its accuracy is low.

## Future Improvements

- Larger GA search (population, generations) and other kernels for GA-KELM
- Feature-selection experiments
- Larger, multi-year and multi-city datasets
- Comparison with additional regression and deep-learning models
- Development of an AQI prediction web application
- Deployment of the trained model through an API

## Academic Project

This repository is developed as an **academic machine learning project** focused on **air-pollutant prediction using GA-KELM**.

It demonstrates the practical application of data preprocessing, regression modelling, optimization techniques, and performance evaluation to an environmental data prediction problem.

## Author

**Mohammed Omer Farooq**


## License

This project is intended for academic and educational purposes.