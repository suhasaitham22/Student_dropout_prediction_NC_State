# Student Dropout Prediction (NC State)

Predicting student dropout risk with machine learning, using county-level school data from North Carolina. The notebook explores what drives dropout rates across the state's 100 counties and trains regression models to predict them.

## Dataset

`School data 2020-21 (1).csv` — one row per NC county (100 rows) with 23 columns covering:

- Socioeconomics: median household income, poverty estimates, effective county tax rate, per capita income, total county revenue, relative effort as percent of revenue per student, local appropriations
- Schools: total students enrolled, teacher attrition rates, crime acts per 1000 students
- Dropouts by group: total dropout count plus breakdowns by gender and race/ethnicity (female, male, Asian, Black, Hispanic, American Indian, White)

The target is **Dropout Rate**, computed as total dropout count divided by total students enrolled.

## Approach

The notebook (`ML Project.ipynb`) works through:

1. **Imputations.** Handle missing values across the county features.
2. **Visualizations.** Explore distributions and relationships between socioeconomic factors and dropout rates.
3. **Clustering.** K-Means with 3 clusters groups the counties by profile (73 / 25 / 2 counties per cluster).
4. **Regression modeling.** Predict the dropout rate from the county features:
   - Random Forest regression
   - XGBoost regression

## Results

From the notebook's model outputs:

- Random Forest: MSE 8644.32, RMSE 92.97 (scaled), with 5-fold cross-validation scores of 0.371, 0.542, 0.251, 0.860, 0.977
- XGBoost: MSE 2907.23, RMSE 53.92, R-squared 0.84

XGBoost explained about 84% of the variance in county dropout rates, clearly the stronger of the two models.

## How to run

```bash
git clone https://github.com/suhasaitham22/Student_dropout_prediction_NC_State.git
cd Student_dropout_prediction_NC_State
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook "ML Project.ipynb"
```

The notebook expects `School data 2020-21 (1).csv` next to it, which is already in the repo.

## Tech stack

Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter Notebook.
