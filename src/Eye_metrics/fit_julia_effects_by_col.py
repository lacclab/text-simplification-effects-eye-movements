import os
import pandas as pd
from src.Julia_models import setup_julia, fit_linear_model
from src.utils import replace_results_in_file
from src.utils_stats import add_p_val_symbols
from loguru import logger
from pathlib import Path
from src.constants import (
    _get_by_col_type, _get_per_type,
    BY_COMPREHENSION_COLS, BY_METADATA_COLS, BY_SPEED_COLS,
    BY_CONTEXT_COLS, BY_LINGUISTIC_COLS, BY_SIMPLIFICATION_COLS
    )
import juliapkg
juliapkg.require_julia("=1.10.7")
juliapkg.resolve()

from juliacall import Main as jl, convert as jlconvert  # noqa: E402, F401

def _get_Ele_effects(per, effect_on, L1_or_L2, has_preview, reread):
    effects_file = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/{has_preview}{reread}/Ele_effect_on_{effect_on}_per_{per}.csv"
    
    # Load the two per-subject result files
    if not os.path.exists(effects_file):
        raise FileNotFoundError(f"per-{per} file not found: {effects_file}")

    df = pd.read_csv(effects_file)
    
    # Get Ele effects
    Ele_effects = df[(df['effect_level'] == 'Ele') & (df['Name'] == 'is_Ele')]
    
    # rename effect col
    Ele_effects = Ele_effects.rename(columns={'Coef.': 'Ele_effect'})
    
    return Ele_effects

def _merge_with_comprehension_scores(effect_on, effects_df, comprehension_col, L1_or_L2, has_preview, reread):
    if comprehension_col == "comprehension_score":
        scores_path = f"src/Reading_Comprehension/data/{L1_or_L2}/comprehension_scores_by=subject_id.csv"
    elif comprehension_col == "comprehension_score_adv":
        scores_path = f"src/Reading_Comprehension/data/{L1_or_L2}/comprehension_scores_by=subject_id_and_level.csv"
    else:
        raise ValueError(f"Invalid comprehension_col: {comprehension_col}")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Comprehension scores file not found: {scores_path}")
    
    scores = pd.read_csv(scores_path)
    if comprehension_col == "comprehension_score_adv":
        # keep only Adv comprehension scores
        scores = scores[scores['level'] == 'Adv']
        # rename comprehension score col
        scores = scores.rename(columns={'comprehension_score': 'comprehension_score_adv'})
    
    # merge
    effects_df = effects_df.merge(scores[['subject_id', comprehension_col]], on='subject_id', how='left')
    # save
    select_cols = ['subject_id', 'has_preview', 'reread', comprehension_col, 'pred_col', 'Ele_effect', 'Std. Error', 'l_conf', 'u_conf', 'Pr(>|z|)', 'Pr(>|z|)_symbol', 'effect_level', 'Name']
    save_to_dir = src_path / f"Eye_metrics/{L1_or_L2}/mixed_effects/{has_preview}{reread}/individual_effect_comprehension"
    os.makedirs(save_to_dir, exist_ok=True)
    effects_df[select_cols].to_csv(save_to_dir / f"per_{per}_ele_effects_on_{effect_on}_with_{comprehension_col}.csv", index=False)
    return effects_df

def _merge_with_metadata(effect_on, effects_df, metadata_col, has_preview, reread):
    # load metadata    
    subjects_metadata_file = f"src/Participants_Metadata/data/{L1_or_L2}/participant_metadata_processed.csv"
    subjects_df = pd.read_csv(subjects_metadata_file)
    
    # add participant_id
    effects_df['participant_id'] = effects_df['subject_id'].str.split('_').str[1].astype(int)
    # merge
    effects_df = effects_df.merge(subjects_df[['participant_id', metadata_col]], on='participant_id' ,how='left')
    # save
    select_cols = ['subject_id', 'has_preview', 'reread', metadata_col, 'pred_col', 'Ele_effect', 'Std. Error', 'l_conf', 'u_conf', 'Pr(>|z|)', 'Pr(>|z|)_symbol', 'effect_level', 'Name']
    save_to_dir = src_path / f"Eye_metrics/{L1_or_L2}/mixed_effects/{has_preview}{reread}/individual_effect_metadata"
    os.makedirs(save_to_dir, exist_ok=True)
    effects_df[select_cols].to_csv(save_to_dir / f"per_{per}_ele_effects_on_{effect_on}_with_{metadata_col}.csv", index=False)
    return effects_df

def _merge_with_speed(effect_on, effects_df, speed_col, L1_or_L2, has_preview, reread):
    if "adv" not in speed_col:
        path = f"src/Eye_metrics/data/{L1_or_L2}/reading_speed_by=subject_id.csv"
    elif "adv" in speed_col:
        path = f"src/Eye_metrics/data/{L1_or_L2}/reading_speed_by=subject_id_level.csv"
    else:
        raise ValueError(f"Invalid speed_col: {speed_col}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reading Speed file not found: {path}")
    
    speeds = pd.read_csv(path)
    if 'adv' in speed_col:
        # keep only Adv rows
        speeds = speeds[speeds['level'] == 'Adv']
        # rename speed col
        col_without_adv = speed_col.replace('_adv', '')
        speeds = speeds.rename(columns={col_without_adv: speed_col})
    
    # filter by has_preview and reread
    speeds = speeds[(speeds["has_preview"] == has_preview) & (speeds["reread"] == reread)]
    
    # merge
    effects_df = effects_df.merge(speeds[['subject_id', speed_col]], on='subject_id', how='left')
    # save
    select_cols = ['subject_id', 'has_preview', 'reread', speed_col, 'pred_col', 'Ele_effect', 'Std. Error', 'l_conf', 'u_conf', 'Pr(>|z|)', 'Pr(>|z|)_symbol', 'effect_level', 'Name']
    save_to_dir = src_path / f"Eye_metrics/{L1_or_L2}/mixed_effects/{has_preview}{reread}/individual_effect_speed"
    os.makedirs(save_to_dir, exist_ok=True)
    effects_df[select_cols].to_csv(save_to_dir / f"per_{per}_ele_effects_on_{effect_on}_with_{speed_col}.csv", index=False)
    return effects_df

def _load_linguistic_data():
    ling_df = pd.read_csv(src_path / "readability_metrics/data/paragraphs_metrics_cleaned.csv")
    return ling_df

def _merge_with_linguistic_data(effect_on, effects_df, linguistic_col, has_preview, reread):
    ling_df = _load_linguistic_data()
    
    if 'adv' in linguistic_col:
        ling_df, _ = _get_adv_metric_df(ling_df, [linguistic_col])
    if 'diff' in linguistic_col:
        ling_df, _ = _get_diff_metric_df(ling_df, [linguistic_col])
    else:
        # select cols
        ling_df = ling_df[['text_id', 'level', linguistic_col]].sort_values(by=["text_id"])
        # agg levels Ele and Adv
        ling_df = ling_df.groupby('text_id').agg({linguistic_col: 'mean'}).reset_index()
    
    # merge
    effects_df = effects_df.merge(ling_df[['text_id', linguistic_col]], on='text_id', how='left')
    # save
    select_cols = ['text_id', 'has_preview', 'reread', linguistic_col, 'pred_col', 'Ele_effect', 'Std. Error', 'l_conf', 'u_conf', 'Pr(>|z|)', 'Pr(>|z|)_symbol', 'effect_level', 'Name']
    save_to_dir = src_path / f"Eye_metrics/{L1_or_L2}/mixed_effects/{has_preview}{reread}/text_effect_linguistic"
    os.makedirs(save_to_dir, exist_ok=True)
    effects_df[select_cols].to_csv(save_to_dir / f"per_{per}_ele_effects_on_{effect_on}_with_{linguistic_col}.csv", index=False)
    return effects_df

def _get_adv_metric_df(metrics_df, cols):
    clean_cols = [col.replace('_adv', '') for col in cols if 'adv' in col]
    clean_cols = [col for col in clean_cols if col in metrics_df.columns]
    cols_with_adv = {col: f"{col}_adv" for col in clean_cols}
    # keep only Adv rows
    adv_df = metrics_df[metrics_df['level'] == 'Adv']
    # select cols
    adv_df = adv_df[['text_id', 'level'] + clean_cols].sort_values(by=["text_id"])
    adv_df = adv_df.rename(columns=cols_with_adv)
    # assert half of the rows from metrics_df
    assert adv_df.shape[0] == metrics_df.shape[0] / 2
    return adv_df, list(cols_with_adv.values())

def _get_diff_metric_df(metrics_df, cols):
    clean_cols = [col.replace('diff_', '') for col in cols if 'diff' in col]
    clean_cols = [col for col in clean_cols if col in metrics_df.columns]
    diff_cols = {col: f"diff_{col}" for col in clean_cols}
    # select cols
    diff_df = metrics_df[['text_id', 'level'] + clean_cols].sort_values(by=["text_id"])
    # pivot df by text_id to calc diff Adv - Ele
    all_df = diff_df['text_id'].drop_duplicates().reset_index()
    for clean_col, diff_col in diff_cols.items():
        pivot_df = diff_df.pivot(index='text_id', columns='level', values=clean_col).reset_index()
        pivot_df[diff_col] = pivot_df['Adv'] - pivot_df['Ele']
        all_df = all_df.merge(pivot_df[['text_id', diff_col]], on='text_id', how='left')
    
    # assert half of the rows from metrics_df
    assert all_df.shape[0] == metrics_df.shape[0] / 2
    return all_df, list(diff_cols.values())

def _load_context_data(context_col, has_preview, reread):
    if 'gpt2' in context_col:
        file_name = 'gpt2_surprisal_df.csv'
    elif 'pythia' in context_col:
        file_name = 'pythia70m_surprisal_df.csv'
    else:
        raise ValueError(f"Invalid context_col: {context_col}")
    # complete
    resolution = 'paragraph'
    context_df = pd.read_csv(src_path / f"Context_Metrics/data/{L1_or_L2}/{has_preview}{reread}/{resolution}_{file_name}")
    return context_df

def _merge_with_context_data(effect_on, effects_df, context_col, has_preview, reread=0):
    context_df = _load_context_data(context_col, has_preview, reread)
    
    if 'adv' in context_col:
        context_df, _ = _get_adv_metric_df(context_df, [context_col])
    if 'diff' in context_col:
        context_df, _ = _get_diff_metric_df(context_df, [context_col])
    else:
        # select cols
        context_df = context_df[['text_id', 'level', context_col]]
        # agg levels Ele and Adv
        context_df = context_df.groupby('text_id').agg({context_col: 'mean'}).reset_index()
    
    # merge
    effects_df = effects_df.merge(context_df[['text_id', context_col]], on='text_id', how='left')
    # save
    select_cols = ['text_id', 'has_preview', 'reread', context_col, 'pred_col', 'Ele_effect', 'Std. Error', 'l_conf', 'u_conf', 'Pr(>|z|)', 'Pr(>|z|)_symbol', 'effect_level', 'Name']
    save_to_dir = src_path / f"Eye_metrics/{L1_or_L2}/mixed_effects/{has_preview}{reread}/text_effect_context"
    os.makedirs(save_to_dir, exist_ok=True)
    effects_df[select_cols].to_csv(save_to_dir / f"per_{per}_ele_effects_on_{effect_on}_with_{context_col}.csv", index=False)
    return effects_df

def _merge_with_simplification_data(effect_on, effects_df, edit_col, has_preview, reread):
    # complete
    res_path = src_path / 'Alignment_Sentences/simplification_types/human_annotation/20250113/simp_per_text_id.csv'
    simp_df = pd.read_csv(res_path)
    # merge
    effects_df = effects_df.merge(simp_df[['text_id', edit_col]], on='text_id', how='left')
    # save
    select_cols = ['text_id', 'has_preview', 'reread', edit_col, 'pred_col', 'Ele_effect', 'Std. Error', 'l_conf', 'u_conf', 'Pr(>|z|)', 'Pr(>|z|)_symbol', 'effect_level', 'Name']
    save_to_dir = src_path / f"Eye_metrics/{L1_or_L2}/mixed_effects/{has_preview}{reread}/text_effect_simplification"
    os.makedirs(save_to_dir, exist_ok=True)
    effects_df[select_cols].to_csv(save_to_dir / f"per_{per}_ele_effects_on_{effect_on}_with_{edit_col}.csv", index=False)
    return effects_df

def _merge_with_multivar_data(effect_on, effects_df, multivar_name, multivar_cols, has_preview, reread):
    ling_df = _load_linguistic_data()
    context_df = _load_context_data("pythia70m_surprisal", has_preview, reread)
    
    # diff cols
    diff_cols = [col for col in multivar_cols if 'diff' in col]
    adv_cols = [col for col in multivar_cols if 'adv' in col]
    
    # init merge
    all_df = context_df['text_id'].drop_duplicates().reset_index()
    # merge
    for metric_df in [ling_df, context_df]:
        if len(adv_cols) > 0:
            adv_df, new_cols = _get_adv_metric_df(metric_df, adv_cols)
            all_df = all_df.merge(adv_df[['text_id'] + new_cols], on='text_id', how='left')
        if len(diff_cols) > 0:
            diff_df, new_cols = _get_diff_metric_df(metric_df, diff_cols)
            all_df = all_df.merge(diff_df[['text_id'] + new_cols], on='text_id', how='left')
    
    # add ratio_added, ratio_deleted metrics
    text_diff_stats_path =  src_path / "Alignment_Sentences/data/Text_differences_paragraph_stats.csv"
    text_diff_stats = pd.read_csv(text_diff_stats_path)
    all_df = all_df.merge(text_diff_stats[['text_id', 'ratio_added', 'ratio_deleted', 'n_deleted_from_adv' ,'n_added_to_ele']], on='text_id', how='left')
    
    # assert that n rows is the same for ling_df, context_df, all_df
    assert ling_df.shape[0] == context_df.shape[0] == (all_df.shape[0] * 2)
    # merge with effects_df
    effects_df = effects_df.merge(all_df[['text_id'] + multivar_cols], on='text_id', how='left')
    # save
    select_cols = (
        ['text_id', 'has_preview', 'reread'] + 
        multivar_cols + 
        ['pred_col', 'Ele_effect', 'Std. Error', 'l_conf', 'u_conf', 'Pr(>|z|)', 'Pr(>|z|)_symbol', 'effect_level', 'Name']
    ) 
    save_to_dir = src_path / f"Eye_metrics/{L1_or_L2}/mixed_effects/{has_preview}{reread}/text_effect_simplification"
    os.makedirs(save_to_dir, exist_ok=True)
    effects_df[select_cols].to_csv(save_to_dir / f"per_{per}_ele_effects_on_{effect_on}_with_{multivar_name}.csv", index=False)
    return effects_df

def _merge_with_by_col_data(effect_on, Ele_effects, by_col, has_preview, multivar_cols):
    if by_col in BY_COMPREHENSION_COLS:
        df = _merge_with_comprehension_scores(effect_on, Ele_effects, by_col, L1_or_L2, has_preview, reread=0)
    elif by_col in BY_METADATA_COLS:
        df = _merge_with_metadata(effect_on, Ele_effects, by_col, has_preview, reread=0)
    elif by_col in BY_SPEED_COLS:
        df = _merge_with_speed(effect_on, Ele_effects, by_col, L1_or_L2, has_preview, reread=0)
    elif by_col in BY_LINGUISTIC_COLS:
        df = _merge_with_linguistic_data(effect_on, Ele_effects, by_col, has_preview, reread=0)
    elif by_col in BY_CONTEXT_COLS:
        df = _merge_with_context_data(effect_on, Ele_effects, by_col, has_preview, reread=0)
    elif by_col in BY_SIMPLIFICATION_COLS:
        df = _merge_with_simplification_data(effect_on, Ele_effects, by_col, has_preview, reread=0)
    elif 'multivar' in by_col:
        df = _merge_with_multivar_data(effect_on, Ele_effects, by_col, multivar_cols, has_preview, reread=0)
    else:
        raise ValueError(f"Invalid by_col: {by_col}")
    return df

def _save_results(regime_results, reading_regime, L1_or_L2, by_col, effect_on, per, replace_in_file):
    by_col_type = _get_by_col_type(by_col)
    per_type = _get_per_type(per)
    
    if regime_results:
        final_results_df = pd.concat(regime_results, ignore_index=True)
        saving_dir = os.path.abspath(
            os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", reading_regime, f"{per_type}_effect_{by_col_type}")
        )
        os.makedirs(saving_dir, exist_ok=True)

        out_path = os.path.join(saving_dir, f"per_{per}_Ele_effect_on_{effect_on}_by_{by_col}.csv")
        # if replace_in_file -> replace rows of these RT cols
        if replace_in_file:
            replace_results_in_file(out_path, final_results_df)
        else:
            final_results_df.to_csv(out_path, index=False)
        logger.info(f"Saved {per}-specific results to {out_path}")
    else:
        logger.info(f"No analysis results for reading_regime={reading_regime}.")


###############################################################################
# Main analysis loop
###############################################################################
def analyze_Ele_effect_per_subject_text_on_col_by_col(
    per: str,
    L1_or_L2: str,
    pred_cols: list,
    by_col: str,
    effect_on: str,
    replace_in_file: bool,
    x_formula: str = None,
    multivar_cols: list = None
) -> None:
    """
    For each reading_regime, and each pred column, we fit **a separate MixedModel for each subject**.
    That is, we subset the data to each subject and run the model once per subject.

    The final CSV will have the usual columns plus an additional 'subject_id' column so
    you can identify which subject's single-subject model each row pertains to.
    """
    per_col = f"{per}_id"
    
    def _update_coef_table(coef_table, df, has_preview, reread, pred_col, per_col):
        """
        Adds metadata columns to a coefficient table.
        """
        coef_table["has_preview"] = has_preview
        coef_table["reread"] = reread
        coef_table["pred_col"] = pred_col
        coef_table["n_rows"] = df.shape[0]
        
        if per_col == "subject_id":
            coef_table["n_subjects"] = df["subject_id"].nunique()
        elif per_col == "text_id":
            coef_table["n_texts"] = df["text_id"].nunique()
            
        coef_table = add_p_val_symbols(coef_table, 'Pr(>|t|)')
        return coef_table
    
    # Iterate over conditions
    for has_preview in ["Gathering", "Hunting"]:
        Ele_effects = _get_Ele_effects(per, effect_on, L1_or_L2, has_preview, reread=0)
        # merge with by_col data
        df = _merge_with_by_col_data(effect_on, Ele_effects, by_col, has_preview, multivar_cols)
        
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
                df_sub = df[df['pred_col'] == pred_col]
                if 'multivar' in by_col:
                    select_cols = [
                        'Ele_effect', 
                        per_col, 'has_preview', 'reread', 
                        'pred_col', *multivar_cols
                    ]
                else:
                    select_cols = [
                        'Ele_effect', 
                        per_col, 'has_preview', 'reread', 
                        'pred_col', by_col
                    ]
                df_sub = df_sub[select_cols]

                if x_formula:
                    formula = f"Ele_effect ~ 1 + {x_formula}"
                    logger.warning(f"Running model with formula: {formula}")
                else:
                    formula = f"Ele_effect ~ 1 + {by_col}"
                    
                # Fit model & get coefficient table
                coef_table = fit_linear_model(df_sub, 'Ele_effect', formula)
                # Add metadata columns
                _update_coef_table(coef_table, df_sub, has_preview, reread, pred_col, per_col)
                regime_results.append(coef_table)                

            _save_results(regime_results, reading_regime, L1_or_L2, by_col, effect_on, per, replace_in_file)

###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    # src Path
    src_path = Path.cwd() / Path("src")
    
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

    # 3) Ele_effect_on_RT ~ reading_comprehension
    
    # for run_type in ["subject", "text_univariate", "text_multivariate"]:
    for run_type in ["subject"]:
        if run_type == "subject":
            # cols_to_run = (BY_METADATA_COLS + BY_COMPREHENSION_COLS + BY_SPEED_COLS)
            cols_to_run = (['is_student', 'is_secondary', 'is_undergrad', 'is_postgrad', 'edu_level_num'])
        elif run_type == "text_univariate":
            cols_to_run = (BY_SIMPLIFICATION_COLS + BY_LINGUISTIC_COLS + BY_CONTEXT_COLS)
        elif run_type == "text_multivariate":
            run_dict = {
                # "multivar_diff":
                #     ("diff_mean_pythia70m_surprisal + diff_wordFreq_frequency + diff_word_length",
                #      ['diff_mean_pythia70m_surprisal', 'diff_wordFreq_frequency', 'diff_word_length']),
                # "multivar_adv":
                #     ("mean_pythia70m_surprisal_adv + wordFreq_frequency_adv + word_length_adv",
                #      ['mean_pythia70m_surprisal_adv', 'wordFreq_frequency_adv', 'word_length_adv']),
                # "multivar_all":
                #     (("mean_pythia70m_surprisal_adv + wordFreq_frequency_adv + word_length_adv + "
                #     "diff_mean_pythia70m_surprisal + diff_wordFreq_frequency + diff_word_length"),
                #      ['mean_pythia70m_surprisal_adv', 'wordFreq_frequency_adv', 'word_length_adv',
                #       'diff_mean_pythia70m_surprisal', 'diff_wordFreq_frequency', 'diff_word_length']),
                # "multivar_all_and_ratio_add_del":
                #     (("mean_pythia70m_surprisal_adv + wordFreq_frequency_adv + word_length_adv + "
                #     "diff_mean_pythia70m_surprisal + diff_wordFreq_frequency + diff_word_length + "
                #     "ratio_added + ratio_deleted"),
                #      ['mean_pythia70m_surprisal_adv', 'wordFreq_frequency_adv', 'word_length_adv',
                #       'diff_mean_pythia70m_surprisal', 'diff_wordFreq_frequency', 'diff_word_length', 
                #       "ratio_added", "ratio_deleted"]),
                # "multivar_all_and_n_del_add":
                #     (("mean_pythia70m_surprisal_adv + wordFreq_frequency_adv + word_length_adv + "
                #     "diff_mean_pythia70m_surprisal + diff_wordFreq_frequency + diff_word_length + "
                #     "n_deleted_from_adv + n_added_to_ele"),
                #      ['mean_pythia70m_surprisal_adv', 'wordFreq_frequency_adv', 'word_length_adv',
                #       'diff_mean_pythia70m_surprisal', 'diff_wordFreq_frequency', 'diff_word_length', 
                #       "n_deleted_from_adv", "n_added_to_ele"]),
            }
            cols_to_run = run_dict.keys()
            
        for by_col in cols_to_run:
            if run_type == "subject":
                x_formula = None
                multivar_cols = None
                per = "subject"
            elif run_type == "text_univariate":
                x_formula = None
                multivar_cols = None
                per = "text"
            elif run_type == "text_multivariate":
                x_formula = run_dict[by_col][0]
                multivar_cols = run_dict[by_col][1]
                per = "text"
            
            analyze_Ele_effect_per_subject_text_on_col_by_col(
                per=per,
                L1_or_L2=L1_or_L2,
                pred_cols=RT_cols,
                by_col=by_col,
                effect_on="RT",
                replace_in_file=replace_in_file,
                x_formula=x_formula,
                multivar_cols=multivar_cols
                )
            analyze_Ele_effect_per_subject_text_on_col_by_col(
                per=per,
                L1_or_L2=L1_or_L2,
                pred_cols=['is_correct', 'QA_RT', 'words_per_sec_based_P_RT', 'norm_QA_RT'],
                by_col=by_col,
                effect_on="comprehension",
                replace_in_file=replace_in_file,
                x_formula=x_formula,
                multivar_cols=multivar_cols
                )
