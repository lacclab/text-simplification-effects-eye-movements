import os
import pandas as pd
from src.Julia_models import setup_julia, fit_mixed_effects_model
from src.Eye_metrics.fit_mixed_effects_julia import get_df_by_col
from src.utils import replace_results_in_file
from src.utils_stats import add_p_val_symbols
from loguru import logger
from tqdm import tqdm
import juliapkg
juliapkg.require_julia("=1.10.7")
juliapkg.resolve()

from juliacall import Main as jl, convert as jlconvert  # noqa: E402, F401

def _update_coef_table(coef_table, df, has_preview, reread, pred_col, id, per_col):
    """
    Adds metadata columns to a coefficient table.
    """
    if per_col == "subject_id":
        coef_table['subject_id'] = id
    elif per_col == "text_id":
        coef_table['text_id'] = id
    coef_table["has_preview"] = has_preview
    coef_table["reread"] = reread
    coef_table["pred_col"] = pred_col
    coef_table["n_rows"] = df.shape[0]
    coef_table["n_subjects"] = df["subject_id"].nunique()
    coef_table["n_texts"] = df["text_id"].nunique()
    coef_table['mean_pred_col'] = df[pred_col].mean()
    coef_table['std_pred_col'] = df[pred_col].std()
    coef_table = add_p_val_symbols(coef_table, 'Pr(>|z|)')
    return coef_table

def _save_results(regime_results, reading_regime, L1_or_L2, file_name, per, replace_in_file):
    if regime_results:
        final_results_df = pd.concat(regime_results, ignore_index=True)
        saving_dir = os.path.abspath(
            os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", reading_regime)
        )
        os.makedirs(saving_dir, exist_ok=True)

        out_path = os.path.join(saving_dir,file_name)
        # if replace_in_file -> replace rows of these RT cols
        if replace_in_file:
            final_results_df = replace_results_in_file(out_path, final_results_df)
        else:
            final_results_df.to_csv(out_path, index=False)
        logger.info(f"Saved {per}-specific results to {out_path}")
    else:
        logger.info(f"No analysis results for reading_regime={reading_regime}.")
                
###############################################################################
# Main analysis loop
###############################################################################
def calc_Ele_effect_on_cols_per_subject_text(
    per: str,
    L1_or_L2: str,
    pred_cols: list,
    file_name: str,
    replace_in_file: bool
) -> None:
    """
    For each reading_regime, and each pred column, we fit **a separate MixedModel for each subject**.
    That is, we subset the data to each subject and run the model once per subject.

    The final CSV will have the usual columns plus an additional 'subject_id' column so
    you can identify which subject's single-subject model each row pertains to.
    """
    per_col = f"{per}_id"
    # Iterate over conditions
    for has_preview in ["Hunting", "Gathering"]:
        for reread in [0, 1]:
            # Skip analysis for reread == 1
            if reread == 1:
                logger.info(f"Skipping analysis for reread=1, has_preview={has_preview}")
                continue

            reading_regime = f"{has_preview}{reread}"
            logger.info(f"Processing reading_regime: {reading_regime}")

            # We'll store results from all subjects in this list
            regime_results = []

            for pred_col in pred_cols:
                df, file_not_found = get_df_by_col(reading_regime, pred_col, has_preview, reread, L1_or_L2)
                if file_not_found:
                    continue

                # Now, we loop over each subject in this data
                for id, df_sub in tqdm(df.groupby(per_col)):
                    
                    # run the analysis where Adv is the reference level
                    df_sub["is_Ele"] = (df_sub["level"] == "Ele").astype(int)
                    ele_effect_formula = f"{pred_col} ~ 1 + is_Ele + (1+is_Ele|{per_col})"
                    if df_sub[pred_col].nunique() == 1:
                        logger.info(f"Not fitting for {pred_col} as only one value is present.")
                        coef_table = pd.DataFrame({
                            'Name': ['(Intercept)', 'is_Ele'],
                            'Coef.': [None, 0.0],
                            'Std. Error': [None, 0.0],
                            'z': [None, None],
                            'Pr(>|z|)': [None, 1.0],
                            'l_conf': [None, 0.0],
                            'u_conf': [None, 0.0],
                            'link_dist': [None, None],
                            'dof': [None, None],
                            't_quantile': [None, None]
                        })
                    else:
                        # Fit model & get coefficient table
                        coef_table = fit_mixed_effects_model(df_sub, pred_col, ele_effect_formula, silent=True)
                    # Add metadata columns
                    _update_coef_table(coef_table, df_sub, has_preview, reread, pred_col, id, per_col)
                    coef_table['reference_level'] = 'Adv'
                    coef_table['effect_level'] = 'Ele'
                    regime_results.append(coef_table)
                    
                    # run the analysis where Ele is the reference level
                    df_sub["is_Adv"] = (df_sub["level"] == "Adv").astype(int)
                    adv_effect_formula = f"{pred_col} ~ 1 + is_Adv + (1+is_Adv|text_id)"
                    if df_sub[pred_col].nunique() == 1:
                        logger.info(f"Not fitting for {pred_col} as only one value is present.")
                        coef_table = pd.DataFrame({
                            'Name': ['(Intercept)', 'is_Adv'],
                            'Coef.': [None, 0.0],
                            'Std. Error': [None, 0.0],
                            'z': [None, None],
                            'Pr(>|z|)': [None, 1.0],
                            'l_conf': [None, 0.0],
                            'u_conf': [None, 0.0],
                            'link_dist': [None, None],
                            'dof': [None, None],
                            't_quantile': [None, None]
                        })
                    else:
                        # Fit model & get coefficient table
                        coef_table = fit_mixed_effects_model(df_sub, pred_col, adv_effect_formula, silent=True)
                    # Add metadata columns
                    _update_coef_table(coef_table, df_sub, has_preview, reread, pred_col, id, per_col)
                    coef_table['reference_level'] = 'Ele'
                    coef_table['effect_level'] = 'Adv'
                    regime_results.append(coef_table)
            
            _save_results(regime_results, reading_regime, L1_or_L2, file_name, per, replace_in_file)


###############################################################################
# Example usage
###############################################################################
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

    # 3) Run per-subject analysis
    calc_Ele_effect_on_cols_per_subject_text(
        per="subject",
        L1_or_L2=L1_or_L2,
        pred_cols=RT_cols,
        file_name="Ele_effect_on_RT_per_subject.csv",
        replace_in_file=replace_in_file
    )
    calc_Ele_effect_on_cols_per_subject_text(
        per="subject",
        L1_or_L2=L1_or_L2,
        pred_cols=["is_correct", "QA_RT", "words_per_sec_based_P_RT", "norm_QA_RT"],
        file_name="Ele_effect_on_comprehension_per_subject.csv",
        replace_in_file=replace_in_file
    )
    # 4) Run per-text analysis
    calc_Ele_effect_on_cols_per_subject_text(
        per="text",
        L1_or_L2=L1_or_L2,
        pred_cols=RT_cols,
        file_name="Ele_effect_on_RT_per_text.csv",
        replace_in_file=replace_in_file
    )
    calc_Ele_effect_on_cols_per_subject_text(
        per="text",
        L1_or_L2=L1_or_L2,
        pred_cols=["is_correct", "QA_RT", "words_per_sec_based_P_RT", "norm_QA_RT"],
        file_name="Ele_effect_on_comprehension_per_text.csv",
        replace_in_file=replace_in_file
    )