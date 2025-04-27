import os
import pandas as pd
from src.constants import PROCESSED_EYE_METRICS_PATH
from src.utils import replace_results_in_file
from src.utils_stats import add_p_val_symbols
from src.Julia_models import setup_julia, fit_mixed_effects_model
from loguru import logger

import juliapkg
juliapkg.require_julia("=1.10.7")
juliapkg.resolve()

from juliacall import Main as jl, convert as jlconvert  # noqa: E402, F401

def get_df_by_col(reading_regime, col, has_preview, reread, L1_or_L2):
    if 'is_correct' in col:
        scores_path = f"src/Reading_Comprehension/data/{L1_or_L2}/comprehension_scores.csv"
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Comprehension scores file not found: {scores_path}")
        scores = pd.read_csv(scores_path)
        # filter by has_preview and reread
        scores = scores[(scores["has_preview"] == has_preview) & (scores["reread"] == reread)]
        if col == "is_correct_adv":
            # keep only Adv comprehension scores
            scores = scores[scores['level'] == 'Adv']
            # rename comprehension score col
            scores = scores.rename(columns={'is_correct': 'is_correct_adv'})
        return scores, False
    elif col == 'words_per_sec_based_P_RT':
        speed_df = f"src/Eye_metrics/data/{L1_or_L2}/reading_speed_by=subject_id_text_level.csv"
        # filter by has_preview and reread
        speed_df = pd.read_csv(speed_df)
        speed_df = speed_df[(speed_df["has_preview"] == has_preview) & (speed_df["reread"] == reread)]
        return speed_df, False
    else:
        df, file_not_found = get_df_by_RT_col(reading_regime, col, has_preview, reread, L1_or_L2)
        return df, file_not_found
        
        
def get_df_by_RT_col(reading_regime, rt_col, has_preview, reread, L1_or_L2):    
    if rt_col in ['GD', 'FF', 'NF']:
        RT_file = 'other_RT_df.csv'
    elif rt_col in ['FirstPassGD', 'FirstPassFF', 'HigherPassFixation']:
        RT_file = 'other_FirstPassRT_df.csv'
    elif 'QA_RT' in rt_col:
        RT_file = 'QA_RT_df.csv'
    else:
        RT_file = f"{rt_col}_df.csv"
    
    # Build path to CSV
    data_path = os.path.join(
        PROCESSED_EYE_METRICS_PATH,
        L1_or_L2,
        reading_regime,
        RT_file
    )

    # Check if the file exists
    if not os.path.isfile(data_path):
        logger.warning(f"File not found for {reading_regime}, {rt_col}: {data_path}")
        return pd.DataFrame(), True

    # Load CSV
    df = pd.read_csv(data_path)
    # Filter df by conditions has_preview and reread
    df = df[(df["has_preview"] == has_preview) & (df["reread"] == reread)]
    # log the number of rows
    logger.info(f"Loaded {df.shape[0]} rows for {reading_regime}, {rt_col}")
    return df, False

def _update_coef_table(coef_table, df, has_preview, reread, pred_col):
    """
    Adds metadata columns to a coefficient table.
    """
    coef_table["has_preview"] = has_preview
    coef_table["reread"] = reread
    coef_table["pred_col"] = pred_col
    coef_table["n_rows"] = df.shape[0]
    coef_table["n_subjects"] = df["subject_id"].nunique()
    coef_table["n_texts"] = df["text_id"].nunique()
    coef_table = add_p_val_symbols(coef_table, 'Pr(>|z|)')
    return coef_table

def _save_results(regime_results, reading_regime, L1_or_L2, file_name, replace_in_file):
    # If we got any results for this reading regime, save to CSV
    if regime_results:
        final_results_df = pd.concat(regime_results, ignore_index=True)

        # Build target output folder:
        saving_dir = os.path.abspath(
            os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", reading_regime)
        )
        os.makedirs(saving_dir, exist_ok=True)

        # Save
        out_path = os.path.join(saving_dir,file_name)
        # if replace_in_file -> replace rows of these RT cols
        if replace_in_file:
            final_results_df = replace_results_in_file(out_path, final_results_df)
        else:
            final_results_df.to_csv(out_path, index=False)
        logger.info(f"Saved results to {out_path}")
    else:
        logger.info(f"No analysis results for reading_regime={reading_regime}")

###############################################################################
# Main analysis loop
###############################################################################
def calc_Ele_effect_on_cols(
    L1_or_L2: str,
    pred_cols: list,
    file_name: str,
    replace_in_file: bool
) -> None:
    """
    Iterates over conditions (has_preview, reread), and RT columns, then
    fits a mixed-effects model for each scenario. Saves results to:
        src/Eye_metrics/{L1_or_L2}/mixed_effects/{reading_regime}/Ele_effect_on_RT.csv

    Parameters
    ----------
    L1_or_L2 : str
        Indicator of language group or reading context (e.g., "L1" or "L2").
    processed_eye_metrics_path : str
        Base directory path where the processed eye metrics are located.
    pred_cols : list
        List of reading-time metric columns to analyze.
    """
    # Iterate over conditions
    for has_preview in ["Gathering", "Hunting"]:
        for reread in [0, 1]:
            # Skip analysis for reread == 1
            if reread == 1:
                logger.info(f"Skipping analysis for reread=1, has_preview={has_preview}")
                continue

            reading_regime = f"{has_preview}{reread}"
            logger.info(f"Processing reading_regime: {reading_regime}")

            # Collect results for this reading regime in a list
            regime_results = []

            # Iterate over each reading-time metric
            for pred_col in pred_cols:
                
                df, file_not_found = get_df_by_col(reading_regime, pred_col, has_preview, reread, L1_or_L2)
                if file_not_found:
                    continue
                
                # run the analysis where Adv is the reference level
                df["is_Ele"] = (df["level"] == "Ele").astype(int)
                ele_effect_formula = f"{pred_col} ~ 1 + is_Ele + (1+is_Ele|subject_id) + (1+is_Ele|text_id)"
                # Fit model & get coefficient table
                coef_table = fit_mixed_effects_model(df, pred_col, ele_effect_formula)
                # Add metadata columns
                _update_coef_table(coef_table, df, has_preview, reread, pred_col)
                coef_table['reference_level'] = 'Adv'
                coef_table['effect_level'] = 'Ele'
                regime_results.append(coef_table)
                
                # run the analysis where Ele is the reference level
                df["is_Adv"] = (df["level"] == "Adv").astype(int)
                adv_effect_formula = f"{pred_col} ~ 1 + is_Adv + (1+is_Adv|subject_id) + (1+is_Adv|text_id)"
                # Fit model & get coefficient table
                coef_table = fit_mixed_effects_model(df, pred_col, adv_effect_formula)
                # Add metadata columns
                _update_coef_table(coef_table, df, has_preview, reread, pred_col)
                coef_table['reference_level'] = 'Ele'
                coef_table['effect_level'] = 'Adv'
                regime_results.append(coef_table)

            _save_results(regime_results, reading_regime, L1_or_L2, file_name, replace_in_file)

if __name__ == "__main__":
    # 1) Initialize Julia environment
    setup_julia()

    # 2) Define parameters
    L1_or_L2 = "L1"
    RT_cols = [
        'TF', 'GD', 'FirstPassGD', 'FF', 
        'FirstPassFF', 'NF', 'SkipTotal', 'SkipFirstPass', 
        'IsReg', 'RegCountTotal', 'RegCountFirstPass', 
        'nonzero_TF', 'nonzero_GD', 'nonzero_FF', 'HigherPassFixation']
    replace_in_file = True

    # 3) Run analysis
    calc_Ele_effect_on_cols(
        L1_or_L2=L1_or_L2,
        pred_cols=RT_cols,
        file_name="Ele_effect_on_RT.csv",
        replace_in_file=replace_in_file
        )
    calc_Ele_effect_on_cols(
        L1_or_L2=L1_or_L2, 
        pred_cols=["is_correct"], 
        file_name="Ele_effect_on_comprehension_score.csv",
        replace_in_file=replace_in_file
        )
    calc_Ele_effect_on_cols(
        L1_or_L2=L1_or_L2, 
        pred_cols=['QA_RT'], 
        file_name="Ele_effect_on_QA_RT.csv",
        replace_in_file=replace_in_file
        )
    calc_Ele_effect_on_cols(
        L1_or_L2=L1_or_L2, 
        pred_cols=['words_per_sec_based_P_RT'], 
        file_name="Ele_effect_on_words_per_sec_based_P_RT.csv",
        replace_in_file=replace_in_file
        )
    calc_Ele_effect_on_cols(
        L1_or_L2=L1_or_L2, 
        pred_cols=['norm_QA_RT'], 
        file_name="Ele_effect_on_norm_QA_RT.csv",
        replace_in_file=replace_in_file
        )
    
