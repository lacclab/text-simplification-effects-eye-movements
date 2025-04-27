import os
import pandas as pd
from src.constants import EYE_BY_WORD_DF_L1_ALIGNED_PATH, EYE_BY_WORD_DF_L2_ALIGNED_PATH
from src.utils import replace_results_in_file
from src.utils_stats import add_p_val_symbols
from src.Eye_metrics.eye_df_utils import get_eye_df, preprocess_eye_df
from src.Julia_models import setup_julia, fit_mixed_effects_model
from loguru import logger
from src.Eye_metrics.calc_eye_metrics import _add_HigherPassFixation, _add_SkipTotal, _add_RegCountTotal, _get_nonzero_GD_or_FF, _get_nonzero_TF, _preprocess_TF_df

import juliapkg
juliapkg.require_julia("=1.10.7")
juliapkg.resolve()

from juliacall import Main as jl, convert as jlconvert  # noqa: E402, F401
        
def get_df(L1_or_L2):
    if L1_or_L2 == "L1":
        data_path = EYE_BY_WORD_DF_L1_ALIGNED_PATH
    elif L1_or_L2 == "L2":
        data_path = EYE_BY_WORD_DF_L2_ALIGNED_PATH
    else:
        raise ValueError(f"Invalid L1_or_L2: {L1_or_L2}")
    
    eye_df = get_eye_df(data_path, L1_or_L2)
    return eye_df

def _update_coef_table(coef_table, df, has_preview, reread, pred_col, surp_col):
    """
    Adds metadata columns to a coefficient table.
    """
    coef_table["has_preview"] = has_preview
    coef_table["reread"] = reread
    coef_table["pred_col"] = pred_col
    coef_table["n_rows"] = df.shape[0]
    coef_table["n_subjects"] = df["subject_id"].nunique()
    coef_table["n_texts"] = df["text_id"].nunique()
    coef_table["n_levels"] = df["level"].nunique()
    coef_table["level"] = str(df["level"].unique().tolist())
    coef_table["surp_col"] = surp_col
    coef_table["mean_surp"] = df["surp"].mean()
    coef_table["mean_prev_surp"] = df["prev_surp"].mean()
    coef_table["mean_len"] = df["len"].mean()
    coef_table["mean_prev_len"] = df["prev_len"].mean()
    coef_table["mean_freq"] = df["freq"].mean()
    coef_table["mean_prev_freq"] = df["prev_freq"].mean()
    coef_table["mean_RT"] = df[pred_col].mean()
    coef_table["std_surp"] = df["surp"].std()
    coef_table["std_prev_surp"] = df["prev_surp"].std()
    coef_table["std_len"] = df["len"].std()
    coef_table["std_prev_len"] = df["prev_len"].std()
    coef_table["std_freq"] = df["freq"].std()
    coef_table["std_prev_freq"] = df["prev_freq"].std()
    coef_table["std_RT"] = df[pred_col].std()
    coef_table = add_p_val_symbols(coef_table, 'Pr(>|z|)')
    return coef_table

def _save_results(regime_results, reading_regime, L1_or_L2, file_name, replace_in_file):
    # If we got any results for this reading regime, save to CSV
    if regime_results:
        final_results_df = pd.concat(regime_results, ignore_index=True)

        # Build target output folder:
        saving_dir = os.path.abspath(
            os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", reading_regime, "RT_response")
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

def _norm_features(df, features):
    new_df = df.copy()
    for feature in features:
        new_df[feature] = (new_df[feature] - new_df[feature].mean()) / new_df[feature].std()
    return new_df

def _norm_RT(df, RT_col):
    new_df = df.copy()
    new_df[RT_col] = (new_df[RT_col] - new_df[RT_col].mean()) / new_df[RT_col].std()
    return new_df

###############################################################################
# Main analysis loop
###############################################################################
def calc_response_to_linguistic_features(
    L1_or_L2: str,
    rt_cols: list,
    surp_cols: list,
    formula_x: str,
    level_formula_x: str,
    norm_RT: bool,
    file_name: str,
    replace_in_file: bool,
    norm_features: bool = False
) -> None:
    # Load CSV
    original_eye_df = get_df(L1_or_L2)
    full_surp_cols = [col for col in original_eye_df.columns if 'surp' in col]
    freq_cols = [col for col in original_eye_df.columns if 'freq' in col]
    len_cols = [col for col in original_eye_df.columns if 'len' in col]
    # Iterate over conditions
    for has_preview in ["Gathering", "Hunting"]:
        for reread in [0, 1]:
            # Skip analysis for reread == 1
            if reread == 1:
                logger.info(f"Skipping analysis for reread=1, has_preview={has_preview}")
                continue

            reading_regime = f"{has_preview}{reread}"
            logger.info(f"Processing reading_regime: {reading_regime}")
            regime_df = preprocess_eye_df(original_eye_df, reading_regime, filter_TF=False)
            regime_df = regime_df.reset_index(drop=True)

            # Collect results for this reading regime in a list
            regime_results = []

            # Iterate over each reading-time metric
            for rt_col in rt_cols:
                
                if rt_col == "TF":
                    eye_df = _preprocess_TF_df(regime_df)
                elif rt_col == "nonzero_TF":
                    eye_df = _get_nonzero_TF(regime_df)
                elif rt_col == "nonzero_GD":
                    eye_df = _get_nonzero_GD_or_FF(regime_df, col='GD')
                elif rt_col == "nonzero_FF":
                    eye_df = _get_nonzero_GD_or_FF(regime_df, col='FF')
                elif rt_col == "HigherPassFixation":
                    eye_df = _add_HigherPassFixation(regime_df, filter_nulls=True)
                elif rt_col == "SkipTotal":
                    eye_df = _add_SkipTotal(regime_df)
                elif rt_col == "RegCountTotal":
                    eye_df = _add_RegCountTotal(regime_df)
                else:
                    eye_df = regime_df.copy()
                    
                if rt_col not in eye_df.columns:
                    raise ValueError(f"Column {rt_col} not found in eye_df")
                select_columns = ["subject_id", "text_id", "level", "has_preview", "reread"] + [rt_col] + full_surp_cols + freq_cols + len_cols
                eye_df = eye_df[select_columns].reset_index(drop=True)
                
                for surp_col in surp_cols:
                    logger.info(f"Processing {rt_col} response to {surp_col} surprisal")
                    eye_df['surp'] = eye_df[f"{surp_col}_surprisal"]
                    eye_df['prev_surp'] = eye_df[f"prev_{surp_col}_surprisal"]
                    
                    eye_df['len'] = eye_df['word_length_no_punctuation']
                    eye_df['prev_len'] = eye_df['prev_word_length_no_punctuation']
                    
                    eye_df['freq'] = eye_df['wordfreq_frequency']
                    eye_df['prev_freq'] = eye_df['prev_wordfreq_frequency']
                    
                    formula = f"{rt_col} ~ {formula_x}"
                    
                    select_cols =['subject_id', 'text_id', 'level', 'has_preview', 'reread', rt_col, 'surp', 'prev_surp', 'len', 'prev_len', 'freq', 'prev_freq']
                    curr_df = eye_df[select_cols].copy()
                    
                    # drop na from rt_col
                    # log n rows dropped, percentage of rows dropped
                    n_rows = curr_df.shape[0]
                    curr_df = curr_df.dropna(subset=[rt_col])
                    n_rows_dropped = n_rows - curr_df.shape[0]
                    percentage_dropped = n_rows_dropped/n_rows
                    if n_rows_dropped > 0:
                        logger.info(f"RT col: {rt_col} | Dropped {n_rows_dropped} rows, {percentage_dropped:.2%} of rows")
                    
                    # run the analysis seperatly for each level
                    for _, level_df in curr_df.groupby("level"):
                        if norm_RT:
                            level_df = _norm_RT(level_df, rt_col)
                        if norm_features:
                            level_df = _norm_features(level_df, features=['surp', 'prev_surp', 'len', 'prev_len', 'freq', 'prev_freq'])
                            
                        # Fit model & get coefficient table                        
                        coef_table = fit_mixed_effects_model(level_df, rt_col, formula)
                        # Add metadata columns
                        _update_coef_table(coef_table, level_df, has_preview, reread, rt_col, surp_col)
                        regime_results.append(coef_table)
                    
                    # fit again for statistical test on the coeff of level
                    logger.info("Running statistical test on the coeff of level")
                    # add is_Ele col to df
                    curr_df['is_Ele'] = curr_df['level'].apply(lambda x: 1 if x == 'Ele' else 0)
                    if norm_RT:
                        curr_df = _norm_RT(curr_df, rt_col)
                    if norm_features:
                        curr_df = _norm_features(curr_df, features=['surp', 'prev_surp', 'len', 'prev_len', 'freq', 'prev_freq'])
                    
                    level_formula = f"{rt_col} ~ {level_formula_x}"
                    
                    coef_table = fit_mixed_effects_model(curr_df, rt_col, level_formula)
                    # Add metadata columns
                    _update_coef_table(coef_table, curr_df, has_preview, reread, rt_col, surp_col)
                    regime_results.append(coef_table)
                    _save_results(regime_results, reading_regime, L1_or_L2, file_name, replace_in_file)

            _save_results(regime_results, reading_regime, L1_or_L2, file_name, replace_in_file)

def _build_formula_str(interaction, add_prev, random_effects):
    formula = "1 + "
    level_formula = "1 + is_Ele * surp + is_Ele * len + is_Ele * freq + "
    
    base = "freq + len + surp"
    
    if interaction == "all":
        formula += "freq * len * surp + " 
    elif interaction == "only_freq_len":
        formula += "freq * len + surp + "
    elif interaction == "base":
        formula += f"{base} + "
    else:
        raise ValueError(f"Invalid interaction: {interaction}")
    
    if add_prev:
        formula += "prev_freq + prev_len + prev_surp + "
        level_formula += "is_Ele * prev_freq + is_Ele * prev_len + is_Ele * prev_surp + "
    
    if random_effects == "base":
        formula += f"(1 + {base} |subject_id) + (1 + {base} |text_id)"
        level_formula += f"(1 + is_Ele + {base} |subject_id) + (1 + is_Ele + {base} |text_id)"
    elif random_effects == "with_prev":
        prev = "prev_freq + prev_len + prev_surp"
        formula += f"(1 + {base} + {prev}|subject_id) + (1 + {base} + {prev}|text_id)"
        level_formula += f"(1 + is_Ele + {base} + {prev} |subject_id) + (1 + is_Ele + {base} + {prev} |text_id)"   
    
    return formula, level_formula

if __name__ == "__main__":
    # 1) Initialize Julia environment
    setup_julia()

    # 2) Define parameters
    L1_or_L2 = "L1"
    RT_cols = ['nonzero_TF', 'nonzero_FF', 'nonzero_GD', 'FirstPassFF', 'FirstPassGD', 'SkipTotal', 'RegCountTotal']
    replace_in_file=True
    surp_cols = ["pythia70m"]

    formulas = {
        "6": (_build_formula_str(interaction="base", add_prev=True, random_effects="base"))
    }

    # 3) Run analysis
    formula_version = "6"
    logger.info(f"Running analysis for formula version {formula_version}")
    for norm_RT in [False, True]:
        calc_response_to_linguistic_features(
            L1_or_L2=L1_or_L2,
            rt_cols=RT_cols,
            surp_cols=surp_cols,
            formula_x=formulas[formula_version][0],
            level_formula_x=formulas[formula_version][1],
            norm_RT=norm_RT,
            file_name=f"RT_response_formula={formula_version}_norm={norm_RT}.csv",
            replace_in_file=replace_in_file
        )
