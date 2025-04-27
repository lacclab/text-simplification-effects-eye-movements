from loguru import logger
import pandas as pd
import numpy as np
import scipy.stats
import time

import juliapkg
juliapkg.require_julia("=1.10.7")
juliapkg.resolve()

from juliacall import Main as jl, convert as jlconvert  # noqa: E402, F401

###############################################################################
# Julia Setup
###############################################################################
def setup_julia():
    """
    Install and import the needed Julia packages.
    """
    print(jl.seval("VERSION"))
    jl.seval("import Pkg")
    # jl.seval('Pkg.add("GLM")')
    # jl.seval('Pkg.add("MixedModels")')
    # jl.seval('Pkg.add("DataFrames")')
    # jl.seval('Pkg.add("Distributions")')
    jl.seval("using MixedModels")
    jl.seval("using DataFrames")
    jl.seval("using Distributions")
    jl.seval("using GLM")
    logger.info("Julia environment is set up with MixedModels and DataFrames.")
    
###############################################################################
# Helpers
###############################################################################
def add_CIs_to_coef_df(coef_df: pd.DataFrame, dof: int) -> pd.DataFrame:
    """
    Adds 95% CI columns (l_conf, u_conf) to a coefficient table.

    Parameters
    ----------
    coef_df : pd.DataFrame
        The coefficient table from MixedModels.
    dof : int
        Degrees of freedom, used to compute the t-quantile.

    Returns
    -------
    pd.DataFrame
        The original coef_df with additional columns for lower and upper CI.
    """
    coef_df_copy = coef_df.copy()
    t_quantile = scipy.stats.t(df=dof).ppf(0.975)
    coef_df_copy["l_conf"] = coef_df_copy["Coef."] - t_quantile * coef_df_copy["Std. Error"]
    coef_df_copy["u_conf"] = coef_df_copy["Coef."] + t_quantile * coef_df_copy["Std. Error"]
    coef_df_copy["dof"] = dof
    coef_df_copy["t_quantile"] = t_quantile
    return coef_df_copy

def choose_link_dist(df, outcome_variable: str, all_normal=False):
    """

    Args:
        outcome_variable (str): can be: ['TF', 'SkipTotal', 'SkipFirstPass', 'IsReg', 'RegCountTotal', 'RegCountFirstPass']

    Raises:
        ValueError: if the outcome_variable is not recognized

    Returns:
        jl.Distributions: the appropriate link distribution for the given outcome variable
    """
    if all_normal:
        return jl.Distributions.Normal()
    if outcome_variable in ["SkipTotal", "SkipFirstPass", "IsReg"]:
        # validate that this col is binary
        unique_vals = df[outcome_variable].unique()
        if len(unique_vals) != 2 or set(unique_vals) != {0, 1}:
            raise ValueError(f"Column '{outcome_variable}' is not binary.")
        # Bernoulli distribution
        link_dist = jl.Distributions.Bernoulli()
    elif outcome_variable in [
        "RegCountTotal",
        "RegCountFirstPass",
    ]:
        # validate that this col is integer
        if not df[outcome_variable].dtype == int:
            raise ValueError(f"Column '{outcome_variable}' is not integer.")
        # check if there aren't values above 1
        if not (df[outcome_variable] > 1).any():
            logger.warning(f"Column '{outcome_variable}' has no values above 1.")
        # Poisson distribution
        link_dist = jl.Distributions.Poisson()
    elif outcome_variable in ['TF']:
        # Normal distribution
        link_dist = jl.Distributions.Normal()
    else:
        raise ValueError(f"Unknown outcome variable: {outcome_variable}")
    return link_dist

def _validate_nulls(df):
    # log n rows with nulls in any column
    n_nulls = df.isnull().any(axis=1).sum()
    if n_nulls > 0:
        logger.warning(f"Dropping {n_nulls} rows with null values")
    
    # Drop nulls
    df = df.dropna()
    if df.empty:
        raise ValueError("No valid rows remain after dropping NA.")
    return df

def _pass_df_to_julia(df: pd.DataFrame):
    # Pass dataframe to Julia
    jl.seval("global j_df = 0") # need to define before assigning
    jl.j_df = jlconvert(jl.PyTable, df)
    
def _pass_convert_df_to_julia():
    # Define a Julia function to convert Julia tables to pandas DataFrames
    # This function uses PythonCall to facilitate the conversion
    jl.seval(
    """
        function table_to_pd(x)
            PythonCall.Compat.pytable(x)
        end
    """
    )

def validate_unique_vals(df, predict_col, formula):
    # checks that pred_col is not constant
    if len(df[predict_col].unique()) == 1:
        logger.warning(f"Column '{predict_col}' is constant. Skipping model fit.")
        return False
    return True

###############################################################################
# Models
###############################################################################
def fit_mixed_effects_model(df: pd.DataFrame, predict_col: str, formula: str, silent: bool = False) -> pd.DataFrame:
    """
    Fit a mixed effects model to the given data.
    """
    df = _validate_nulls(df)
    _pass_df_to_julia(df)
    _pass_convert_df_to_julia()
    ok_flag = validate_unique_vals(df, predict_col, formula)
    if not ok_flag:
        return "skipped"
    
    # Create formula in Julia
    jl.seval(f"j_formula = @formula({formula})")
    # Choose link distribution
    jl.seval("global link_dist = 0")
    link_dist = choose_link_dist(df, predict_col, all_normal=True)
    jl.link_dist = link_dist
    # Fit the model
    if not silent:
        logger.info("Fitting model...")
    # save time for logging
    start_time = time.time()
    jl.seval("model_res = fit(MixedModel, j_formula, j_df, link_dist, progress=false)")
    model_res_name = "model_res"
    # log time in minutes and seconds
    elapsed_time = time.time() - start_time
    if not silent:
        logger.info(f"Model fit in {elapsed_time//60:.0f} minutes, {elapsed_time%60:.0f} seconds")
    # Extract coefficient table and degrees of freedom
    mm_coeftable = jl.table_to_pd(
        jl.MixedModels.coeftable(getattr(jl, model_res_name))
    )
    # convert coef by link dist
    mm_coeftable['link_dist'] = str(link_dist)
    # Add confidence intervals
    mm_dof = jl.MixedModels.dof(getattr(jl, model_res_name))
    mm_coeftable = add_CIs_to_coef_df(mm_coeftable, mm_dof)
    # Add formula
    mm_coeftable["formula"] = formula
    return mm_coeftable

def fit_linear_model(df: pd.DataFrame, predict_col: str, formula: str, silent: bool = False) -> pd.DataFrame:
    """
    Fit a mixed effects model to the given data.
    """
    df = _validate_nulls(df)
    _pass_df_to_julia(df)
    _pass_convert_df_to_julia()
    
    # Create formula in Julia
    jl.seval(f"j_formula = @formula({formula})")
    # Fit the model
    if not silent:
        logger.info("Fitting model...")
    jl.seval("model_res = lm(j_formula, j_df)")
    model_res_name = "model_res"
    # Extract coefficient table and degrees of freedom
    mm_coeftable = jl.table_to_pd(
        jl.GLM.coeftable(getattr(jl, model_res_name))
    )
    # Add confidence intervals
    mm_dof = jl.GLM.dof(getattr(jl, model_res_name))
    mm_coeftable = add_CIs_to_coef_df(mm_coeftable, mm_dof)
    
    # Retrieve predictions from Julia => pass them back to Python
    #    so we can compute correlation
    preds_jl = jl.seval("predict(model_res, j_df)")  # Julia Vector of predictions
    fitted_y = np.array(preds_jl, dtype=float).ravel()     # convert to numpy

    # Compute Pearson r between the actual outcome & fitted predictions
    actual_y = df[predict_col].values.astype(float).ravel()
    r_val, p_val = scipy.stats.pearsonr(actual_y, fitted_y)

    mm_coeftable["Pearson_r"] = r_val
    mm_coeftable["Pearson_pval"] = p_val
    
    # Add formula
    mm_coeftable["formula"] = formula
    return mm_coeftable