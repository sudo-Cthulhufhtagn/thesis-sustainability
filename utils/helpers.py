from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score, KFold, cross_validate, StratifiedGroupKFold, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import plotly.express as px
# import xgboost as xgb
import mlflow
import time
import pickle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone as clone_estimator
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils.models import PLSPreProc
from sklearn.preprocessing import StandardScaler
from utils.plotting import plot_preds, plot_y_predicter, plot_vip, plot_vip_spectrum, plot_spectrum, plot_stage2_importances, plot_residuals, plot_features_corr
from joblib import Memory

# Initialize memory cache
memory = Memory('local_cache', verbose=0)

@memory.cache # Cache the function to save compute time
def optimize_components(X, y, gscv):
    """
    Optimize the number of components for PLS regression.

    Parameters
    ----------
    X: np.array
    The input features
    y: np.array
    The target variable
    gscv: GridSearchCV object
    The grid search cross-validation object

    Returns
    -------
    GridSearchCV object
    The optimized grid search cross-validation object
    """
    gscv.fit(X, y)
    return gscv

def run_experiment(data_df: pd.DataFrame,
                   X_idx: list,
                   y_predicter: str, 
                   n_splits: int = 8,
                   random_state: int = 42, 
                   test_name: str = '24',
                   suffix: str = '',
                   month: int = None,
                   feature_columns: list[str] = None,
                   spectral_model_pipeline = False, 
                   spectral_model = None, 
                   features_to_exclude: dict[str:list] = {},
                   stage1: str = None, # 'preds' for predictions or 'components' for components
                   residual_model = None, 
                   ):
    """
    Run the experiment with the given parameters and log results to MLflow.

    Parameters
    ----------
    data_df : pd.DataFrame
        The input data frame containing features and target variable.
    X_idx : list
        List of feature column indices.
    y_predicter : str
        The name of the target variable column.
    n_splits : int, optional
        Number of splits for cross-validation (default is 8).
    random_state : int, optional
        Random state for reproducibility (default is 42).
    test_name : str, optional
        Name of the test/experiment (default is '24').
    feature_columns : list[str], optional
        List of feature column names (default is None).
    spectral_model_pipeline : bool, optional
        Whether to use a spectral model pipeline (default is False).
    spectral_model : optional
        The spectral model to use (default is None).
    features_to_exclude : list[str], optional
        List of feature column names to exclude (default is None).
    stage1 : str, optional
        Stage 1 type: 'preds' for predictions or 'components' for components (default is None).
    residual_model : optional
        The residual model to use (default is None).

    Returns
    -------
    None
    """
    
    # Set up MLflow experiment
    mlflow.set_experiment(f"/experiments/{test_name+suffix}")
    with mlflow.start_run(run_name=f"{y_predicter}") as run:
        # Log parameters to MLflow
        mlflow.log_param("spectral-model", str(spectral_model))
        mlflow.log_param("spectral-model-pipeline", spectral_model_pipeline)
        mlflow.log_param("stage1", stage1)
        mlflow.log_param("residual_model", residual_model)
        mlflow.log_param("y_predicter", y_predicter)
        mlflow.log_param("month", month)

        # Initialize cross-validation strategies
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        gkf = GroupKFold(n_splits=n_splits)
        cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)

        # Copy data and filter based on quantiles
        data_df_v2 = data_df.copy()
        percentile_to_cut = 0.005
        mlflow.log_param("percentile_to_cut", percentile_to_cut)

        try:
            quantiles = {
                "lower": data_df_v2[y_predicter].quantile(percentile_to_cut),
                "upper": data_df_v2[y_predicter].quantile(1 - percentile_to_cut),
            }
        except:
            print("No quantiles for", y_predicter)

        filtered_data_df = data_df_v2[
            (data_df_v2[y_predicter] > quantiles["lower"])
            & (data_df_v2[y_predicter] < quantiles["upper"])
        ]

        filtered_data_df.reset_index(drop=True, inplace=True)
        mlflow.log_param("data_size_full", len(data_df_v2))
        mlflow.log_param("data_size", len(filtered_data_df))
        mlflow.log_param("X_idx", X_idx)
        mlflow.log_param("X_idx_len", len(X_idx))

        # Log feature correlations if feature columns and residual model are provided
        if feature_columns and residual_model is not None:
            feature_columns_corr = filtered_data_df[(feature_columns + [y_predicter])].corr()
            mlflow.log_param("feature_columns_len", len(feature_columns))
            feature_columns_selected = [
                f for f in list(set(feature_columns) - set(features_to_exclude.get(y_predicter, {})))
                if (f not in y_predicter and f[:9] not in y_predicter)
            ]

            mlflow.log_param("feature_columns_selected", feature_columns_selected)
            fig = plot_features_corr(feature_columns_corr)
            mlflow.log_figure(fig, "feature_columns_corr.png")

        # Prepare input features and log spectrum
        X = filtered_data_df[X_idx]
        X_mean = -X.mean(axis=0)
        X_std = X.std(axis=0)
        fig = plot_spectrum(X_mean)
        mlflow.log_figure(fig, "spectrum.png")

        # Log target variable distribution
        y = data_df_v2[y_predicter]
        fig = plot_y_predicter(y, y_predicter, quantiles)
        mlflow.log_figure(fig, "histogram.png")

        y = filtered_data_df[y_predicter]
        fig = plot_y_predicter(y, y_predicter)
        mlflow.log_figure(fig, "histogram - filtered.png")

        mlflow.log_param("na_y", y.isna().sum())
        if y.isna().sum():
            raise Exception("NA detected")

        # Start timing the experiment
        time_start = time.time()
        if spectral_model:
            # Define parameter grid for spectral model
            n_components_name = "preproc__n_components" if spectral_model_pipeline else "n_components"
            parameters = (
                {n_components_name: range(5, 71, 5)}
                if not spectral_model_pipeline
                else {n_components_name: range(50, 95, 5)}
            )
            mlflow.log_param("parameters_grid", parameters)

            # Initialize GridSearchCV for spectral model
            gscv = GridSearchCV(
                spectral_model,
                parameters,
                cv=cv,
                n_jobs=-1,
                scoring="neg_mean_squared_error",
            )

            # Optimize components and log best parameters
            gscv = optimize_components(X, y, gscv=gscv)
            n_components_best = gscv.best_params_[n_components_name]
            mlflow.log_param("best_n_components", n_components_best)
            mlflow.log_metric("best_score", gscv.best_score_)

        # Initialize predictions dictionary
        predictions = {
            "r2_test": [],
            "r2_train": [],
            "mse_test": [],
            "mse_train": [],
            'mape_test': [],
            'mape_train': [],
            "true_test": [],
            "true_train": [],
            "predicted_test": [],
            "predicted_train": [],
            "mfc": [],
            'importances': [],
            'vip_scores': []
        }

        # Perform cross-validation
        for train_idx, test_idx in sgkf.split(
                X,
                pd.qcut(filtered_data_df[y_predicter], 10, labels=False, duplicates='drop'), # Ensure balance in CV folds
                groups=filtered_data_df.milc_farm_code.astype("category"),
            ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            if spectral_model:
                spectral_model.set_params(**{n_components_name: n_components_best})

                # Fit and predict using spectral model
                if stage1 == "preds":
                    spectral_model.fit(X_train, y_train)
                    y_train_pred_stage1 = spectral_model.predict(X_train)
                    y_test_pred_stage1 = spectral_model.predict(X_test)
                    if residual_model is None:
                        y_test_preds = y_test_pred_stage1
                        y_train_preds = y_train_pred_stage1

                # Transform data using spectral model components
                elif stage1 == "components":
                    y_train_pred_stage1 = spectral_model.fit_transform(X_train, y_train)
                    y_test_pred_stage1 = spectral_model.transform(X_test, y_test)

                # Log feature importances
                fig = plot_vip(spectral_model.feature_importances_)
                mlflow.log_figure(fig, "feature_importances_stage1.png")
                fig = plot_vip_spectrum(X_mean, spectral_model.feature_importances_)
                mlflow.log_figure(fig, "feature_importances_spectrum_stage1.png")
                predictions["vip_scores"].append(spectral_model.feature_importances_)

            # Fit and predict using residual model
            if not residual_model is None:
                scaler1 = StandardScaler()
                scaler2 = StandardScaler()
                scaler1.set_output(transform='pandas')
                scaler2.set_output(transform='pandas')

                if not spectral_model is None:
                    if len(y_train_pred_stage1.shape) == 1:
                        y_train_pred_stage1 = y_train_pred_stage1.to_frame()
                        y_test_pred_stage1 = y_test_pred_stage1.to_frame()
                    
                    X_train_residual = pd.concat(
                        [
                            scaler1.fit_transform(y_train_pred_stage1),
                            scaler2.fit_transform(filtered_data_df.loc[
                                        train_idx,
                                        feature_columns_selected
                                    ].reset_index(drop=True)
                                ),
                        ],
                        axis=1,
                    )
                    X_test_residual = pd.concat(
                        [
                            scaler1.transform(y_test_pred_stage1),
                            scaler2.transform(filtered_data_df.loc[
                                        test_idx,
                                        feature_columns_selected
                                        ].reset_index(drop=True)
                                                )
                            ],
                            axis=1
                        )
                    residual_model.fit(X_train_residual, y_train)
                    y_train_preds = residual_model.predict(X_train_residual)
                    y_test_preds = residual_model.predict(X_test_residual)
                        
                else:
                    X_train_residual = scaler2.fit_transform(filtered_data_df.loc[
                                        train_idx,
                                        feature_columns_selected
                                    ].reset_index(drop=True)
                                )
                    X_test_residual = scaler2.transform(filtered_data_df.loc[
                                    test_idx,
                                    feature_columns_selected
                                    ].reset_index(drop=True))
                    residual_model.fit(X_train_residual, y_train)
                    y_train_preds = residual_model.predict(X_train_residual)
                    y_test_preds = residual_model.predict(X_test_residual)

                column_names = X_train_residual.columns.tolist()
                if hasattr(residual_model, "feature_importances_"):
                    predictions["importances"].append(residual_model.feature_importances_)
                else:
                    predictions["importances"].append(residual_model.coef_)

            # Calculate and store performance metrics
            r2_test = r2_score(y_test, y_test_preds)
            r2_train = r2_score(y_train, y_train_preds)
            mse_test = mean_squared_error(y_test, y_test_preds)
            mse_train = mean_squared_error(y_train, y_train_preds)
            mape_test = mean_absolute_percentage_error(y_test, y_test_preds)
            mape_train = mean_absolute_percentage_error(y_train, y_train_preds)
            predictions["r2_test"].append(r2_test)
            predictions["r2_train"].append(r2_train)
            predictions["mse_test"].append(mse_test)
            predictions["mse_train"].append(mse_train)
            predictions["mape_test"].append(mape_test)
            predictions["mape_train"].append(mape_train)
            predictions["true_test"].extend(y_test.tolist())
            predictions["true_train"].extend(y_train.tolist())
            predictions["predicted_test"].extend(y_test_preds.tolist())
            predictions["predicted_train"].extend(y_train_preds.tolist())
            predictions["mfc"].extend(filtered_data_df.milc_farm_code.iloc[test_idx].tolist())

        # Log feature importances if available
        if predictions["importances"]:
            coefs_df = pd.DataFrame(predictions["importances"], columns=column_names)
            fig = plot_stage2_importances(coefs_df)
            mlflow.log_param("stage2_importances", predictions["importances"])
            mlflow.log_param("stage2_columns", column_names)
            mlflow.log_figure(fig, "feature_importances.png")

        if predictions["vip_scores"]:
            mlflow.log_param("stage2_vip_scores", predictions["vip_scores"])

        # Log performance metrics to MLflow
        for metric in [
            "r2_test",
            "r2_train",
            "mse_test",
            "mse_train",
            "mse_train",
            "mape_test",
            "mape_train",
            ]:
            for value in predictions[metric]:
                mlflow.log_metric(f"{metric}", value)

        mlflow.log_metric("r2_test_mean", np.mean(predictions["r2_test"]))
        mlflow.log_metric("r2_test_std", np.std(predictions["r2_test"]))
        mlflow.log_metric("r2_train_mean", np.mean(predictions["r2_train"]))
        mlflow.log_metric("r2_train_std", np.std(predictions["r2_train"]))
        mlflow.log_metric("mse_test_mean", np.mean(predictions["mse_test"]))
        mlflow.log_metric("mse_test_std", np.std(predictions["mse_test"]))
        mlflow.log_metric("mse_train_mean", np.mean(predictions["mse_train"]))
        mlflow.log_metric("mse_train_std", np.std(predictions["mse_train"]))
        mlflow.log_metric("mape_test_mean", np.mean(predictions["mape_test"]))
        mlflow.log_metric("mape_test_std", np.std(predictions["mape_test"]))
        mlflow.log_metric("mape_train_mean", np.mean(predictions["mape_train"]))
        mlflow.log_metric("mape_train_std", np.std(predictions["mape_train"]))

        # Calculate and log overall performance metrics
        y_test_true = np.array(predictions["true_test"])
        y_test_pred = np.array(predictions["predicted_test"])
        y_train_true = np.array(predictions["true_train"])
        y_train_pred = np.array(predictions["predicted_train"])

        mlflow.log_metric("r2_test_mean_cv", r2_score(y_test_true, y_test_pred))
        mlflow.log_metric("r2_train_mean_cv", r2_score(y_train_true, y_train_pred))
        mlflow.log_metric("mse_test_mean_cv", mean_squared_error(y_test_true, y_test_pred))
        mlflow.log_metric("mse_train_mean_cv", mean_squared_error(y_train_true, y_train_pred))
        mlflow.log_metric("mape_test_mean_cv", mean_absolute_percentage_error(y_test_true, y_test_pred))
        mlflow.log_metric("mape_train_mean_cv", mean_absolute_percentage_error(y_train_true, y_train_pred))

        # Plot and log actual vs predicted values
        pred_true_test_plot = plot_preds(y_test_true, y_test_pred, predictions["mfc"])
        mlflow.log_figure(pred_true_test_plot, "ActualVsPredicted.html")

        # Save and log predictions
        with open('./predictions.pkl', 'wb') as file:
            pickle.dump(predictions, file)
        mlflow.log_artifact("predictions.pkl")

        # Plot and log residuals
        if len(y_test_pred.shape) == 2:
            y_test_pred = y_test_pred[:, 0]
        vals = y_test_pred - y_test_true
        fig = plot_residuals(vals)
        mlflow.log_figure(fig, "Residual histogram.png")

        mlflow.log_metric("mean_residual", np.mean(vals))
        mlflow.log_metric("std_residual", np.std(vals))