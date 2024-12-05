import seaborn as sns
import plotly.express as px
import matplotlib.collections as collections
import matplotlib.pyplot as plt

def plot_preds(y, y_cv, mfc):
    """
    Plot actual vs predicted values

    Parameters
    ----------
    y: array-like of shape (n_samples,)
    Actual values.
    y_cv: array-like of shape (n_samples,)
    Predicted values.
    mfc: array-like of shape (n_samples,)
    The color of the markers, milk_farm_code

    Returns
    -------
    fig: plotly.graph_objects.Figure
    """
    if len(y_cv.shape) == 2:
        y_cv = y_cv[:, 0]
    
    # Create a scatter plot with trendline
    fig = px.scatter(
                x=y,
                y=y_cv,
                trendline="ols",
                template="seaborn",
                trendline_color_override="red",
                opacity=0.8,
                hover_data={"mfc": mfc},
                labels=dict(x="Actual", y="Predicted"),
                title=f"Actual vs Predicted",
            )

    # Update axes for equal scaling
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    # Add diagonal reference line
    fig.update_layout(
        width=700,
        height=700,
        shapes=[
            dict(
                type="line",
                x0=min(y),
                y0=min(y),
                x1=max(y),
                y1=max(y),
                line=dict(color="black", width=2, dash="dash"),
            )
        ],
    )

    return fig

def plot_y_predicter(y, y_predicter, quantiles = None):
    """
    Plot histogram of predicted values

    Parameters
    ----------
    y: array-like of shape (n_samples,)
    Values.
    y_predicter: str
    Predicter name.
    quantiles: dict
    Quantiles.

    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(y, bins=100)
    ax.set_title(f"Histogram of {y_predicter}")
    ax.set_xlabel(y_predicter)
    ax.set_ylabel("Frequency")
    
    # Add quantile lines if provided
    if quantiles:
        ax.axvline(quantiles["lower"], linestyle="--", color="r")
        ax.axvline(quantiles["upper"], linestyle="--", color="r")

    plt.close()
    return fig

def plot_vip(importances):
    """
    Plot variable importance VIP plot. VIP score > 1 is a sign of a good feature.

    Parameters
    ----------
    importances: array-like of shape (n_features,)
    Importance scores.

    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(range(len(importances)), importances, "b", linestyle='--', marker="o", linewidth=.5)
    
    # Add horizontal line to indicate VIP threshold
    plt.axhline(y=1, color='r', linestyle='-')
    ax.set_title(f"Variable importance VIP Plot")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    plt.close()
    return fig

def plot_vip_spectrum(X_mean, importances):
    """
    Plot variable importance VIP plot for spectrum. Red bands on the figure are the features with VIP > 1.

    Parameters
    ----------
    X_mean: array-like of shape (n_samples,)
    Mean spectrum.
    importances: array-like of shape (n_features,)
    Importance scores.

    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(X_mean, "b",)

    # Highlight regions with VIP > 1
    collection = collections.BrokenBarHCollection.span_where(
        range(len(X_mean)), ymin=X_mean.min(), ymax=X_mean.max(), where=importances > 1, facecolor='red', alpha=0.5,)
    
    plt.xlabel('Wavenumber cm$^{-1}$')
    plt.ylabel('Intensity a.u.')
    ax.add_collection(collection)
    plt.close()

    return fig

def plot_spectrum(X):
    """
    Plot the spectrum.

    Parameters
    ----------
    X: array-like of shape (n_samples,)
    Spectrum data.

    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(X, "b", linestyle='--', marker="o", linewidth=2)
    ax.set_title(f"Spectrum")
    plt.close()
    return fig

def plot_stage2_importances(coefs_df):
    """
    Plot the feature importances and variability.
    The boxplot shows the variability of the feature importances.
    
    Parameters
    ----------
    coefs_df: pd.DataFrame
    The data frame containing the feature importances.

    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create boxplot for feature importances
    sns.boxplot(ax=ax, data=coefs_df, orient="h", color="blue", saturation=.5)
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_title("Feature Importances and Variability")
    plt.close()
    return fig

def plot_features_corr(feature_columns_corr):
    """
    Plot the correlation between features.
    Parameters
    ----------
    feature_columns_corr: pd.DataFrame
    The correlation matrix.

    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create heatmap for feature correlation matrix
    sns.heatmap(feature_columns_corr, ax=ax, cmap="RdBu_r", annot=True, fmt=".2f", square=True, vmin=-1, vmax=1)
    plt.close()
    return fig

def plot_residuals(residual):
    """
    Plot the residuals.
    Parameters
    ----------
    residual: array-like of shape (n_samples,)
    The residuals.

    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(residual, bins=100)
    ax.set_xlabel("Predicted - Actual")
    ax.set_ylabel("Count")
    ax.set_title("Residual histogram")
    plt.close()
    return fig