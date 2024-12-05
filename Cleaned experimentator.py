# Databricks notebook source
# MAGIC %md
# MAGIC

# COMMAND ----------

# first install missing dependencies
!pip install -q mlflow plotly tqdm plotnine scikit-learn pandas 

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleaned version of main experiments pipeline
# MAGIC Meant for combination spectral and non spectral features and tracking everything with mlflow

# COMMAND ----------

from utils.queries import baseline_year, baseline_12_months
from utils.helpers import run_experiment
from utils.models import PLSPreProc
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tqdm.auto import tqdm
import gc


# COMMAND ----------

# this is needed to autoreload the modules
%load_ext autoreload
%autoreload 2

# COMMAND ----------

# load the data
data_df = spark.sql(baseline_year).toPandas() 
# data_df = spark.sql(baseline_12_months).toPandas() # this is for 12 months averages

# COMMAND ----------

# load the data and remove possible redundant columns
data_df = data_df.iloc[:, ~data_df.columns.duplicated()]

# COMMAND ----------

feature_columns = [
    # include here list of columns to be used as features
]

y_predictors = [
    # a list of predictors to be used
]

features_to_exclude = {
    # a disctionary of sets to be excluded from the feature_columns for each predictor
    'y_predictor': {
        'feature1',
        'feature2',
    }
}

spectrum_idxs = np.array([...]) # spectrum Idxs to be used

# COMMAND ----------

for y_predicter in tqdm(y_predictors):
  for model1, model2 in [
    [PLSPreProc(), None],
    [None, LinearRegression()],
    [None, GradientBoostingRegressor()],
    [PLSPreProc(), LinearRegression()],
    [PLSPreProc(), GradientBoostingRegressor()],
  ]:
    run_experiment( # run and log the experiments
      data_df,
      spectrum_idxs,
      y_predicter=y_predicter,
      test_name='improved_v4',
      spectral_model_pipeline = False,
      spectral_model = model1,
      stage1='preds',
      feature_columns = feature_columns,
      features_to_exclude = features_to_exclude,
      residual_model = model2,
    )

    # collect the garbage to prevent memory leak
    gc.collect()
