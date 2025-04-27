import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns
from typing import List, Union
from loguru import logger
import matplotlib.patches as mpatches
import textwrap
from src.Eye_metrics.fit_julia_effects_by_col import _get_by_col_type, _get_per_type
from src.constants import (
    SIGNIFICANCE_COLORS, SIGNIGICANCE_MARKERSIZE,
    LEVEL_LABELS, LEVEL_COLORS, LEVEL_TEXT_COLORS, LEVEL_HATCH, 
    EFFECT_COLOR, PRED_COLS_FULL_LABELS, PRED_COLS_SHORT_LABELS, BASE_COLOR, BASE_COLOR_2,
    SIMPLIFICATION_TYPES_SHORT_LABELS, REGIME_LABELS, SUBJECT_TEXT_LABELS,
    BY_COL_LABELS, SELF_REPORTED_STR,
    _get_label_linguistic_col
    )
from src.utils import add_significance_bracket, add_significance_legend
from src.utils_stats import add_p_val_symbols, get_mean_ci

REPO_PATH = Path("/data/home/gkeren/6771003ed4bc1f42fbf477f9")
OVERLEAF_PATH = REPO_PATH / "Plots" / "all"

MEANS_PLOT_WIDTH = 3
MEANS_PLOT_HIGHT = 2
MEANS_PLOT_TITLE_SIZE = 12

REG_PLOT_HEIGHT = 4
REG_PLOT_WIDTH = 6

def _save_latex_figure(
    overleaf_dir: Path,
    pdf_file_name: str,
    plot_type: str,
    reading_regimes: Union[List[str], str],
    rt_cols: List[str]=None,
    comp_cols: List[str]=None,
    pred_cols: List[str]=None,
    reader_type: str=None,
    per: str=None,
    effect_on: str=None,
    by_col: str=None,
    by_col_type: str=None,
):
    """
    Creates a .tex file in 'overleaf_dir' referencing 'pdf_file_name'.
    The figure caption is built using 'resolution', 'rt_cols', and 'text_cols'.
    """

    # 1) Construct the .tex file name by replacing .pdf with .tex
    latex_file_name = pdf_file_name.replace(".pdf", ".tex")
    latex_file_name = f"tex_{latex_file_name}"
    latex_path = overleaf_dir / latex_file_name
    
    # if file already exists and dont_edit_latex is True, skip
    if latex_path.exists() and dont_edit_latex:
        logger.info(f"File {latex_path} already exists. Skipping.")
        return

    # 2) Build a caption referencing the user data = 
    
    if plot_type == "Ele_effects_grid":
        plot_title = "Simplification effect on Reading Times and Reading Comprehension Metrics. "
        additional_notes = f"Comparison of mean values between {LEVEL_LABELS['Adv']} and {LEVEL_LABELS['Ele']} texts. "
        add_rt_labels = True
        add_comp_labels = True
        add_sign_str = True
        sign_of = "simplification effect"
        marked_on = " on each pair of bars"
        main_plot = True
    elif plot_type == "Ele_effects_on_RT":
        plot_title = "Simplification effect on Reading Times. "
        additional_notes = f"Comparison of mean values between {LEVEL_LABELS['Adv']} and {LEVEL_LABELS['Ele']} texts. "
        add_rt_labels = True
        add_comp_labels = False
        add_sign_str = True
        sign_of = "simplification effect"
        marked_on = " on each pair of bars"
        main_plot = True
    elif plot_type == "Ele_effect_on_reading_comprehension":
        plot_title = "Simplification effect on Reading Comprehension Metrics. "
        additional_notes = f"Comparison of mean values between {LEVEL_LABELS['Adv']} and {LEVEL_LABELS['Ele']} texts. "
        add_rt_labels = False
        add_comp_labels = True
        add_sign_str = True
        sign_of = "simplification effect"
        marked_on = " on each pair of bars"
        main_plot = True
    elif plot_type == "RT_response":
        plot_title = "Linguistic Effects on Reading Times"
        additional_notes = "'prev' - Previous words metrics. '&' - interaction between metrics. The full formula appears at the bottom of the plot."
        add_rt_labels = True
        add_comp_labels = False
        add_sign_str = True
        sign_of = "effect"
        marked_on = " on each pair of bars"
        main_plot = True  
    elif (plot_type == "Ele_effect_on_comprehension_per_subject"
          or plot_type == "Ele_effect_on_comprehension_per_text"):
        per_str = SUBJECT_TEXT_LABELS[per]
        plot_title = rf"Simplification effect on \textit{{Reading Comprehension}} per {{{per_str}}}. "
        additional_notes = ""
        add_rt_labels = False
        add_comp_labels = True
        comp_cols = pred_cols
        add_sign_str = True
        sign_of = "simplification effect"
        marked_on = " on error bars"
        main_plot = False
    elif (plot_type == "Ele_effect_on_RT_per_subject"
          or plot_type == "Ele_effect_on_RT_per_text"):
        per_str = SUBJECT_TEXT_LABELS[per]
        plot_title = rf"Simplification effect on \textit{{Reading Times}} per {{{per_str}}}. "
        additional_notes = ""
        add_rt_labels = True
        rt_cols = pred_cols
        add_comp_labels = False
        add_sign_str = True
        sign_of = "simplification effect"
        marked_on = " on error bars"
        main_plot = False
    elif plot_type == "Ele_effects_grid_per_text_subject":
        subject_str = SUBJECT_TEXT_LABELS["subject"]
        text_str = SUBJECT_TEXT_LABELS["text"]
        plot_title = rf"Simplification effect on Reading Times and Reading Comprehension per \textit{{{subject_str}}} and per \textit{{{text_str}}}. "
        pred_cols_str = [PRED_COLS_FULL_LABELS[col] for col in pred_cols]
        pred_cols_str = ", ".join(pred_cols_str)
        additional_notes = f"Simplification Effect on\nmetrics: {pred_cols_str}. "
        add_rt_labels = False
        add_comp_labels = False
        add_sign_str = True
        sign_of = "simplification effect"
        marked_on = " on error bars"
        main_plot = True
    elif plot_type == "Ele_effects_grid_on_col_per_text_subject_SM":
        subject_str = SUBJECT_TEXT_LABELS["subject"]
        text_str = SUBJECT_TEXT_LABELS["text"]
        if effect_on == "RT":
            on_label = "Reading Times"
        elif effect_on == "comprehension":
            on_label = "Reading Comprehension"
        plot_title = rf"Simplification effect on {{{on_label}}} per \textit{{{subject_str}}} and per \textit{{{text_str}}}. "
        pred_cols_str = [PRED_COLS_FULL_LABELS[col] for col in pred_cols]
        pred_cols_str = ", ".join(pred_cols_str)
        additional_notes = f"Simplification Effect on\nmetrics: {pred_cols_str}. "
        add_rt_labels = False
        add_comp_labels = False
        add_sign_str = True
        sign_of = "simplification effect"
        marked_on = " on error bars"
        main_plot = False
    elif plot_type == "reg_plot_on_col_by_col":
        per_str = SUBJECT_TEXT_LABELS[per]
        if effect_on == "RT":
            on_label = "Reading Times"
        elif effect_on == "comprehension":
            on_label = "Reading Comprehension"
        by_col_label = BY_COL_LABELS.get(by_col, by_col)
        if SELF_REPORTED_STR in by_col_label:
            by_col_pre, by_col_post = by_col_label.split(SELF_REPORTED_STR)
            plot_title = rf"Simplification effect on {{{on_label}}} per \textit{{{per_str}}} by {SELF_REPORTED_STR}\textit{{{by_col_post}}}. "
        else:
            plot_title = rf"Simplification effect on {{{on_label}}} per \textit{{{per_str}}} by \textit{{{by_col_label}}}. "
        additional_notes = ""
        if effect_on == "RT":
            add_rt_labels = True
            rt_cols = pred_cols
            add_comp_labels = False
        elif effect_on == "comprehension":
            add_rt_labels = False
            add_comp_labels = True
            comp_cols = pred_cols
        add_sign_str = True
        sign_of = "simplification effect"
        marked_on = " on dots"
        main_plot = False
    elif plot_type == "reg_lines_grid":
        per_str = SUBJECT_TEXT_LABELS[per]
        if per == "subject":
            by_col_type_str = f"{per_str} Reading Speed, Reading Comprehension, and Metadata"
        elif per == "text":
            by_col_type_str = f"{per_str} Liguistic"
        plot_title = rf"Simplification effect on on Reading Times and Reading Comprehension per \textit{{{per_str}}} by \textit{{{by_col_type_str}}} Metrics. "
        pred_cols_str = [PRED_COLS_FULL_LABELS[col] for col in pred_cols]
        pred_cols_str = ", ".join(pred_cols_str)
        additional_notes = f"Simplification Effect on metrics: {pred_cols_str}. "
        add_rt_labels = False
        add_comp_labels = False
        add_sign_str = True
        sign_of = "simplification effect"
        marked_on = " on dots"
        main_plot = True
    elif plot_type == "per_text_R_bars_grid":
        plot_title = ("Correlation between effect size (on reading times or reading comprehension) and linguistic properties of the text."
                      " The Original Complexity Model includes frequency, length and surprisal of the original text."
                      " The Reduced Complexity Model includes the differences in frequency, length and surprisal between the original and simplified texts."
                      " The Combined Model includes both sets of features.")
        additional_notes = ""
        add_rt_labels = False
        add_comp_labels = False
        add_sign_str = False
        main_plot = True
        
    if add_sign_str:
        add_sign_str = rf"The \textbf{{significance of {sign_of}}} is marked{marked_on}. "
    else:
        add_sign_str = ""
    
    if add_rt_labels:
        relevant_rt_labels = [PRED_COLS_FULL_LABELS[col] for col in rt_cols]
        rt_list_str = ", ".join(relevant_rt_labels)
        rt_labels_str = f"Reading Time columns: {rt_list_str}. "
    else:
        rt_labels_str = ""
        
    if add_comp_labels:
        relevant_comp_labels = [PRED_COLS_FULL_LABELS[col] for col in comp_cols]
        comp_list_str = ", ".join(relevant_comp_labels)
        comp_labels_str = f"Reading Comprehension columns: {comp_list_str}. "
    else:
        comp_labels_str = ""
    
    if reading_regimes == "Hunting0":
        reading_regime_str = r"\textit{{Information Seeking}} Reading Regime. "
        main_plot = False
    else:
        reading_regime_str = ""
        
    if reader_type == "general_reader":
        reader_type_str = r" Reading Times are generated using EZ-Reader with the default hyperparameters. "
    else:
        reader_type_str = ""
        
    if 'SM' in pdf_file_name:
        main_plot = False
    
    caption_text = rf"{plot_title}{reader_type_str}"
    # Create a list of non-empty components
    caption_components = [
        additional_notes,
        reading_regime_str,
        comp_labels_str,
        rt_labels_str,
        add_sign_str
    ]
    # Filter out empty components
    caption_components = [component for component in caption_components if component.strip()]
    # Add non-empty components to caption_text, joined by newlines
    if caption_components:
        caption_text += "\n" + "\n".join(caption_components)
    
    # Figure size
    if main_plot:
        full_size_str = "figure"
        width = 0.4
        if plot_type == "RT_response" and main_plot:
            width = 0.3
    else:
        full_size_str = "figure*"
        width = 1
    
    # Figure label
    figure_label = pdf_file_name.replace(".pdf", "")  # e.g. "Ele_effects_on_RT"
    if reading_regimes == "Hunting0" and "Hunting0" not in figure_label:
        figure_label = f"{figure_label}_Hunting0"
    if reader_type == "general_reader":
        figure_label = f"{figure_label}_EZ_general"
        
    # Figure path
    sub_path = str(overleaf_dir).split("Plots/")[-1]
    figure_path = f"Plots/{sub_path}/{pdf_file_name}"

    # 3) Build a short LaTeX figure environment referencing the same PDF
    latex_content = rf"""\begin{{{full_size_str}}}[ht]
    \centering
    \includegraphics[width={{{width}}}\textwidth]{{{figure_path}}}
    \caption{{{caption_text}}}
    \label{{fig:{figure_label}}}
\end{{{full_size_str}}}
    """
    
    # 4) Write the .tex file
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

def _format_mean_values(data):
    """
    Given a DataFrame with 2 rows (Adv, Ele), each having a 'Mean' column,
    decide how to format them:
      - if the two means are the same up to 2 decimals, print them with 3 decimals
        and use smaller font (8).
      - otherwise, print them with 2 decimals and font size 10.
    Returns:
      [adv_str, ele_str])
    """
    adv_val = data['Mean'].iloc[0]
    ele_val = data['Mean'].iloc[1]
    
    adv_str = f"{adv_val:.3f}"
    ele_str = f"{ele_val:.3f}"
    return [adv_str, ele_str]

def _extract_means_with_CI(file_path, pred_col):
    """
    Extracts data for plotting from a mixed effects model results CSV file,
    using l_conf and u_conf to form the error bars.
    """
    df = pd.read_csv(file_path)
    df = df[df['pred_col'] == pred_col].copy()

    adv_row = df[(df['reference_level'] == 'Adv') & (df['Name'] == '(Intercept)')].iloc[0]
    ele_row = df[(df['reference_level'] == 'Ele') & (df['Name'] == '(Intercept)')].iloc[0]
    sign_ele_effect_row = df[(df['effect_level'] == 'Ele') & (df['Name'] == 'is_Ele')].iloc[0]

    mean_adv = adv_row['Coef.']
    mean_ele = ele_row['Coef.']

    # Lower/upper CI for adv
    l_adv = adv_row['l_conf']
    u_adv = adv_row['u_conf']

    # Lower/upper CI for ele
    l_ele = ele_row['l_conf']
    u_ele = ele_row['u_conf']

    significance = sign_ele_effect_row['Pr(>|z|)_symbol']

    # Convert confidence bounds to error offsets
    adv_err_down = mean_adv - l_adv
    adv_err_up   = u_adv   - mean_adv
    ele_err_down = mean_ele - l_ele
    ele_err_up   = u_ele   - mean_ele

    data = pd.DataFrame({
        'Level':       ['Adv', 'Ele'],
        'Mean':        [mean_adv, mean_ele],
        'ErrLow':      [adv_err_down, ele_err_down],
        'ErrHigh':     [adv_err_up,   ele_err_up],
        'Significance': [significance, significance]
    })
    return data

def _add_slope_star(slope_row, ax):
    # row where Name == comprehension_col
    slope_sig_symbol = slope_row.iloc[0]["Pearson_pval_symbol"]
    Pearson_r = slope_row.iloc[0]["Pearson_r"]
    # place star near the left or center
    # Get current limits
    xlims = ax.get_xlim()  # (x_min, x_max)
    ylims = ax.get_ylim()  # (y_min, y_max)
    # Choose top-right coordinates, e.g. 90% of each range
    x_star = xlims[0] + 0.7 * (xlims[1] - xlims[0])
    y_star = ylims[0] + 0.93 * (ylims[1] - ylims[0])
    # Add the star
    # ax.text(
    #     x_star, 
    #     y_star - 0.08 * (ylims[1] - ylims[0]),
    #     f"$significance$: {slope_sig_symbol}",
    #     color='black',
    #     fontsize=10, fontweight='bold',
    #     ha='left', va='bottom'
    # )
    # Add Pearson's r
    ax.text(
        x_star,
        y_star,
        f"$Pearson$ $r$ = {Pearson_r:.2f} ({slope_sig_symbol})",
        color='black',
        fontsize=10,
        fontweight='bold',
        ha='left',
        va='bottom'
    )

def _single_bar_plot(ax, csv_path, pred_col, pred_col_label, bar_color=BASE_COLOR):
    # 2) Extract the data for a bar chart
    data = _extract_means_with_CI(csv_path, pred_col=pred_col)

    x_levels = data["Level"].values           # e.g. ['Adv', 'Ele']
    # replace with LEVEL_LABELS
    x_levels = [LEVEL_LABELS.get(level, level) for level in x_levels]
    means    = data["Mean"].values           # two values: [meanAdv, meanEle]
    # Format the means
    means_str = _format_mean_values(data)
    yerr     = np.vstack([data["ErrLow"], data["ErrHigh"]])
    significance_str = data["Significance"].iloc[1]  # from Ele row
    
    if pred_col == "is_correct":
        # convert to percentage
        means = means * 100
        means_str = [f"{mean_val:.2f}" for mean_val in means]
        yerr = yerr * 100
    if pred_col == "QA_RT":
        # convert ms to seconds
        means = means / 1000
        yerr = yerr / 1000
        means_str = [f"{mean_val:.1f}" for mean_val in means]
    if "(ms" in pred_col_label:
        means_str = [f"{mean_val:.0f}" for mean_val in means]
    if "words_per_sec_based_P_RT" in pred_col:
        means_str = [f"{mean_val:.1f}" for mean_val in means]

    # 3) Plot bars
    ax.bar(
        x_levels,
        means,
        yerr=yerr,
        capsize=5,
        width=0.3,
        color=bar_color,
        hatch=[LEVEL_HATCH[level] for level in data['Level']],
    )
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 4) Place numeric labels above bars
    overall_min = (means - yerr[0]).min()
    overall_max = (means + yerr[1]).max()
    # Force y=0 into the axis range
    overall_min = min(overall_min, 0)
    overall_max = max(overall_max, 0)
    
    # Add some padding so bars / text aren’t on the very edge
    up_padding = 0.5 * (overall_max - overall_min)  # 5% padding
    # down_padding = 0.1 * (overall_max - overall_min)  # 5% padding
    y_min = overall_min
    y_max = overall_max + up_padding

    for j, _ in enumerate(x_levels):
        offset = 0.02 * (y_max - y_min)
        top = means[j] + yerr[1, j]
        ax.text(
            j,
            top + offset,
            means_str[j],
            color='black',
            ha='center', va='bottom',
            fontsize=8
        )

    # 5) Ensure axis range
    ax.set_xlim([-0.6, 1.6])
    ax.set_ylim([y_min, y_max])
    ax.axhline(0, color='grey', linewidth=1)

    # 6) Add significance bracket from x=0 to x=1
    bracket_y = y_max - 0.15*(y_max - y_min)
    add_significance_bracket(
        ax=ax,
        x1=0, x2=1,
        y=bracket_y,
        text=significance_str,
        bracket_height=0.02*(y_max-y_min)
    )
    
    # 7) Labels
    # wrapped_label = wrap_label(pred_col_label, line_width=20)
    wrapped_label = pred_col_label
    ax.set_ylabel(wrapped_label, fontsize=10)
    # _plot_subplot_with_level_legend(ax)
    
def wrap_label(label_text, line_width=10):
    wrapped = textwrap.wrap(label_text, width=line_width)
    return "\n".join(wrapped)
    
def plot_Ele_effects_on_RT_bars(
    L1_or_L2: str, 
    rt_cols: List[str], 
    output_file_name: str,
    reading_modes = ["Gathering0", "Hunting0"],
    ):
    """
    BAR PLOT
    One row per RT metric, two columns:
    - Left  => Ordinary Reading
    - Right => Information Seeking
    Each bar has error bars, significance bracket at the top.
    Safely handles cases where bars can be negative.
    """
    Hunting0_results_file = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/Hunting0/Ele_effect_on_RT.csv"
    Gathering0_results_file = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/Gathering0/Ele_effect_on_RT.csv"
    
    n_rows = len(rt_cols)
    n_cols = len(reading_modes)
    
    if (n_cols == 1) and (n_rows > 3): # make gird of pred cols
        reading_mode = reading_modes[0]
        # put half of the pred cols in the first column and the other half in the second column
        n_cols = 2
        # round up the half
        n_rows = (n_rows + 1) // 2
        RT_cols_1 = rt_cols[:n_rows]
        RT_cols_2 = rt_cols[n_rows:]
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(MEANS_PLOT_WIDTH * n_cols, MEANS_PLOT_HIGHT * n_rows), sharex=False)
        
        for j in range(2):
            new_rt_cols = RT_cols_1 if j == 0 else RT_cols_2
            for i, rt_col in enumerate(new_rt_cols):
                if reading_mode == "Gathering0":
                    results_file = Gathering0_results_file
                elif reading_mode == "Hunting0":
                    results_file = Hunting0_results_file
                else:
                    raise ValueError(f"Unknown reading mode: {reading_mode}")
                ax = axes[i, j] if n_cols > 1 else axes[i]
                _single_bar_plot(ax, results_file, rt_col, PRED_COLS_SHORT_LABELS[rt_col])
        
        # fill blank subplots with white
        for i in range(n_rows):
            for j in range(2):
                if i >= len(RT_cols_1):
                    axes[i, j].axis('off')
                
    else: # grid of RT X reading regime
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(MEANS_PLOT_WIDTH * n_cols, MEANS_PLOT_HIGHT * n_rows), sharex=False)
        
        # Titles
        for j, reading_mode in enumerate(reading_modes):
            if n_cols > 1:
                axes[0, j].set_title(REGIME_LABELS[reading_mode], fontsize=MEANS_PLOT_TITLE_SIZE, fontweight='bold')
        
        for i, rt_col in enumerate(rt_cols):
            for j, reading_mode in enumerate(reading_modes):
                if reading_mode == "Gathering0":
                    results_file = Gathering0_results_file
                elif reading_mode == "Hunting0":
                    results_file = Hunting0_results_file
                else:
                    raise ValueError(f"Unknown reading mode: {reading_mode}")
                ax = axes[i, j] if n_cols > 1 else axes[i]
                _single_bar_plot(ax, results_file, rt_col, PRED_COLS_SHORT_LABELS[rt_col])


    plt.tight_layout()

    saving_dir = Path(os.path.abspath(
        os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", "effect_on_RT")
    ))
    # mkdir
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / "effect_on_RT"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = overleaf_dir / output_file_name
    plt.savefig(output_pdf_path)
    plt.close()
    logger.info(f"Bar plot saved to {output_pdf_path}")
    
    # call save_latex_figure
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type="Ele_effects_on_RT",
        reading_regimes=reading_modes,
        rt_cols=rt_cols,
        reader_type=L1_or_L2
    )

def plot_Ele_effect_on_reading_comprehension(
    L1_or_L2: str, 
    output_file_name: str,
    reading_modes = ["Gathering0", "Hunting0"],
    ):
    """
    Creates a 2x2 grid of bar plots:

         (row=0) is_correct          (row=0) is_correct
          Gathering0  -- vs --        Hunting0

         (row=1) QA_RT              (row=1) QA_RT
          Gathering0  -- vs --        Hunting0
    """    
    # 2 rows:
    comp_cols = ['is_correct', 'QA_RT', 'words_per_sec_based_P_RT']
    
    n_rows = len(comp_cols)
    n_cols = len(reading_modes)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(MEANS_PLOT_WIDTH * n_cols, MEANS_PLOT_HIGHT * n_rows), sharex=False)

    # Titles
    for j, reading_mode in enumerate(reading_modes):
        if n_cols > 1:
            axes[0, j].set_title(REGIME_LABELS[reading_mode], fontsize=MEANS_PLOT_TITLE_SIZE, fontweight='bold')

    for i, pred_col in enumerate(comp_cols):
        for j, reading_mode in enumerate(reading_modes):
            ax = axes[i, j] if n_cols > 1 else axes[i]

            # 1) Build the CSV path 
            if pred_col == "is_correct":
                file_name = "comprehension_score"
            elif pred_col == "QA_RT":
                file_name = "QA_RT"
            elif pred_col == "words_per_sec_based_P_RT":
                file_name = "words_per_sec_based_P_RT"
            else:
                raise ValueError(f"Unknown pred_col: {pred_col}")
            csv_path = (
                f"src/Eye_metrics/{L1_or_L2}/mixed_effects/"
                f"{reading_mode}/Ele_effect_on_{file_name}.csv"
            )
            _single_bar_plot(ax, csv_path, pred_col, PRED_COLS_SHORT_LABELS[pred_col])

    plt.tight_layout()

    # Build saving path
    saving_dir = Path(
        os.path.abspath(os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", "effect_on_comprehension"))
    )
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / "effect_on_comprehension"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = overleaf_dir / output_file_name
    plt.savefig(output_pdf_path)
    plt.close() 
    logger.info(f"Ele effect on reading comprehension saved to {output_pdf_path}")
 
    # call save_latex_figure
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type="Ele_effect_on_reading_comprehension",
        reading_regimes=reading_modes,
        comp_cols=comp_cols,
        reader_type=L1_or_L2
    )

def plot_Ele_effects_bars_grid(
    L1_or_L2: str,
    RT_cols: List[str],
    output_file_name: str,
    reading_regime: str,
    ):
    grid_labels_columns = ['Online Measures', 'Offline Measures']
    reading_comp_cols = ['is_correct', 'QA_RT', 'words_per_sec_based_P_RT']
    
    n_rows = max(len(RT_cols), len(reading_comp_cols))
    n_cols = len(grid_labels_columns)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(MEANS_PLOT_WIDTH * n_cols, MEANS_PLOT_HIGHT * n_rows), sharex=False)

    # # Titles
    # for j, col_label in enumerate(grid_labels_columns):
    #     if n_cols > 1:
    #         axes[0, j].set_title(col_label, fontsize=MEANS_PLOT_TITLE_SIZE, fontweight='bold')

    for j, _ in enumerate(grid_labels_columns):
        if j == 0:
            pred_cols = RT_cols
        elif j == 1:
            pred_cols = reading_comp_cols
            
        for i, pred_col in enumerate(pred_cols):
            ax = axes[i, j] if n_cols > 1 else axes[i]

            # 1) Build the CSV path 
            if pred_col == "is_correct":
                file_name = "comprehension_score"
                bar_color = BASE_COLOR_2
            elif pred_col == "QA_RT":
                file_name = "QA_RT"
                bar_color = BASE_COLOR_2
            elif pred_col == "words_per_sec_based_P_RT":
                file_name = "words_per_sec_based_P_RT"
                bar_color = BASE_COLOR
            elif pred_col in RT_cols:
                file_name = "RT"
                bar_color = BASE_COLOR
            else:
                raise ValueError(f"Unknown pred_col: {pred_col}")
            csv_path = (
                f"src/Eye_metrics/{L1_or_L2}/mixed_effects/"
                f"{reading_regime}/Ele_effect_on_{file_name}.csv"
            )
            _single_bar_plot(ax, csv_path, pred_col, PRED_COLS_SHORT_LABELS[pred_col], bar_color)
    
    # # blank subplots
    # ax = axes[2, 1]
    # ax.axis('off')
    
    fig = add_significance_legend(fig, with_colors=False, add_base_colors_legend=True)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    
    # Build saving path
    saving_dir = Path(
        os.path.abspath(os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", "effect_main_grids"))
    )
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / "effect_main_grids"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    overleaf_path = overleaf_dir / output_file_name
    plt.savefig(overleaf_path)
    
    plt.close() 
    logger.info(f"Ele effects main grid saved to {output_pdf_path}")
    
    # call save_latex_figure
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type="Ele_effects_grid",
        reading_regimes=reading_regime,
        rt_cols=RT_cols,
        comp_cols=reading_comp_cols,
        reader_type=L1_or_L2
    )

def plot_Ele_effects_on_RT_scatter(L1_or_L2: str, rt_cols: List[str], output_file_name: str):
    """
    DOT + ERRORBAR PLOT
    One row per RT metric, two columns:
     - Left  => Ordinary Reading
     - Right => Information Seeking
    Points with confidence intervals, significance bracket, y-axis not forced to zero,
    and ensuring text doesn't overlap the border.
    """
    Hunting0_results_file = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/Hunting0/Ele_effect_on_RT.csv"
    Gathering0_results_file = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/Gathering0/Ele_effect_on_RT.csv"
    
    saving_dir = Path(os.path.abspath(
        os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", "effect_on_RT")
    ))
    # mkdir
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    
    n_rows = len(rt_cols)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2.5 * n_rows), sharex=False)

    axes[0, 0].set_title("Ordinary Reading", fontsize=13, fontweight='bold')
    axes[0, 1].set_title("Information Seeking", fontsize=13, fontweight='bold')

    x_coords = [0, 1]

    for i, rt_col in enumerate(rt_cols):
        # Extract data
        gathering_data = _extract_means_with_CI(Gathering0_results_file, rt_col)
        hunting_data   = _extract_means_with_CI(Hunting0_results_file,   rt_col)

        # Format means for each
        gather_means_str, gather_fontsize = _format_mean_values(gathering_data)
        hunt_means_str,   hunt_fontsize   = _format_mean_values(hunting_data)

        # Find min & max
        all_means    = np.concatenate([gathering_data['Mean'], hunting_data['Mean']])
        all_err_up   = np.concatenate([gathering_data['ErrHigh'], hunting_data['ErrHigh']])
        all_err_down = np.concatenate([gathering_data['ErrLow'],  hunting_data['ErrLow']])

        overall_max = np.max(all_means + all_err_up)
        overall_min = np.min(all_means - all_err_down)
        margin = 0.1 * (overall_max - overall_min)
        y_min = overall_min - margin
        y_max = overall_max + margin

        # ====== LEFT => Ordinary Reading ======
        ax_left = axes[i, 0]
        ax_left.errorbar(
            x_coords,
            gathering_data['Mean'],
            yerr=[gathering_data['ErrLow'], gathering_data['ErrHigh']],
            fmt='o', capsize=5, elinewidth=1, capthick=1, color='black'
        )
        for j, level in enumerate(gathering_data['Level']):
            ax_left.plot(x_coords[j], gathering_data['Mean'][j], 
                         marker='o', markersize=6, color=LEVEL_COLORS[level])
            offset = 0.02 * (y_max - y_min)
            ax_left.text(
                x_coords[j],
                gathering_data['Mean'][j] + gathering_data['ErrHigh'][j] + offset,
                gather_means_str[j],
                ha='center', va='bottom',
                color=LEVEL_TEXT_COLORS[level],
                fontsize=gather_fontsize
            )
        ax_left.set_ylim([y_min, y_max])
        ax_left.set_ylabel(rt_col, fontsize=11)

        # Move bracket ~12% below top so text isn't clipped
        bracket_y = y_max - 0.12 * (y_max - y_min)
        add_significance_bracket(
            ax_left,
            x1=0, x2=1,
            y=bracket_y,
            text=gathering_data['Significance'].iloc[0],
            bracket_height=0.03 * (y_max - y_min)
        )

        # ====== RIGHT => Information Seeking ======
        ax_right = axes[i, 1]
        ax_right.errorbar(
            x_coords,
            hunting_data['Mean'],
            yerr=[hunting_data['ErrLow'], hunting_data['ErrHigh']],
            fmt='o', capsize=5, elinewidth=1, capthick=1, color='black'
        )
        for j, level in enumerate(hunting_data['Level']):
            ax_right.plot(x_coords[j], hunting_data['Mean'][j], 
                          marker='o', markersize=6, color=LEVEL_COLORS[level])
            offset = 0.02 * (y_max - y_min)
            ax_right.text(
                x_coords[j],
                hunting_data['Mean'][j] + hunting_data['ErrHigh'][j] + offset,
                hunt_means_str[j],
                ha='center', va='bottom',
                color=LEVEL_TEXT_COLORS[level],
                fontsize=hunt_fontsize
            )
        ax_right.set_ylim([y_min, y_max])
        ax_right.set_ylabel(rt_col, fontsize=11)

        bracket_y = y_max - 0.12 * (y_max - y_min)
        add_significance_bracket(
            ax_right,
            x1=0, x2=1,
            y=bracket_y,
            text=hunting_data['Significance'].iloc[0],
            bracket_height=0.03 * (y_max - y_min)
        )

    # Optional x ticks
    for i in range(n_rows):
        axes[i, 0].set_xticks(x_coords)
        axes[i, 1].set_xticks(x_coords)
        axes[i, 0].set_xticklabels(['Adv','Ele'], fontsize=10)
        axes[i, 1].set_xticklabels(['Adv','Ele'], fontsize=10)

    for col in range(n_cols):
        axes[-1, col].set_xlabel("Level", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_pdf_path)
    # save also to overleaf
    saving_dir = OVERLEAF_PATH / "effect_on_RT"
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    plt.close()
    logger.info(f"Dot+error plot saved to {output_pdf_path}")

def _get_percantage_significant(df_sub, pred_col, text_or_subject):
    # get the number of significant effects
    n_total = len(df_sub)
    n_negative = len(df_sub[df_sub['Coef.'] < 0])
    n_positive = len(df_sub[df_sub['Coef.'] > 0])
    n_sig = len(df_sub[df_sub['Pr(>|z|)_symbol'] != 'ns'])
    n_low_negative = len(df_sub[(df_sub['Coef.'] < 0) & (df_sub['Pr(>|z|)_symbol'] == '*')])
    n_med_negative = len(df_sub[(df_sub['Coef.'] < 0) & (df_sub['Pr(>|z|)_symbol'] == '**')])
    n_high_negative = len(df_sub[(df_sub['Coef.'] < 0) & (df_sub['Pr(>|z|)_symbol'] == '***')])
    n_sig_negative = len(df_sub[(df_sub['Coef.'] < 0) & (df_sub['Pr(>|z|)_symbol'] != 'ns')])
    per_sif_negative = n_sig_negative / n_total * 100 if n_negative > 0 else 0
    n_low_positive = len(df_sub[(df_sub['Coef.'] > 0) & (df_sub['Pr(>|z|)_symbol'] == '*')])
    n_med_positive = len(df_sub[(df_sub['Coef.'] > 0) & (df_sub['Pr(>|z|)_symbol'] == '**')])
    n_high_positive = len(df_sub[(df_sub['Coef.'] > 0) & (df_sub['Pr(>|z|)_symbol'] == '***')])
    n_sig_positive = len(df_sub[(df_sub['Coef.'] > 0) & (df_sub['Pr(>|z|)_symbol'] != 'ns')])
    per_sif_positive = n_sig_positive / n_total * 100 if n_positive > 0 else 0
    
    # store in df with short names
    df = pd.DataFrame({
        'n_total': n_total,
        'n_negative': n_negative,
        'n_positive': n_positive,
        'n_sig': n_sig,
        'n_low_neg': n_low_negative,
        'n_med_neg': n_med_negative,
        'n_high_neg': n_high_negative,
        'n_sig_neg': n_sig_negative,
        'per_sig_neg': per_sif_negative,
        'n_low_pos': n_low_positive,
        'n_med_pos': n_med_positive,
        'n_high_pos': n_high_positive,
        'n_sig_pos': n_sig_positive,
        'per_sig_pos': per_sif_positive
    }, index=[0])
    return df

def _get_n_significant_subjects(df_sub, pred_col):
    # asset 180 rows in df_sub - one per subject
    assert len(df_sub) == 180, f"Expected 180 rows, got {len(df_sub)} for {pred_col}"
    sig_stats_df = _get_percantage_significant(df_sub, pred_col, "subjects")
    return sig_stats_df

def _get_n_significant_items(df_sub, pred_col):
    # asset 162 rows in df_sub - one per paragraph
    assert len(df_sub) == 162, f"Expected 162 rows, got {len(df_sub)} for {pred_col}"
    sig_stats_df = _get_percantage_significant(df_sub, pred_col, "items")
    return sig_stats_df

def _single_plot_per_subject_text(ax, data, pred_col, per):
    """
    Plot the per-subject Ele effect in the given 'ax'.
    df_sub has columns like: Coef., l_conf, u_conf, Pr(>|z|)_symbol, etc.
    """
    # Keep only rows with effect_level = Ele
    data = data[(data["effect_level"] == "Ele") & (data['Name'] == 'is_Ele')].copy()
    # Filter to this pred_col
    df_sub = data[data['pred_col'] == pred_col].copy()
    
    if df_sub.empty:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, f"No data for {pred_col}", ha='center', va='center')
        return
    
    if per == "subject":
        sig_stats_df = _get_n_significant_subjects(df_sub, pred_col)
    elif per == "text":
        sig_stats_df = _get_n_significant_items(df_sub, pred_col)
    sig_stats_df['pred_col'] = pred_col
    sig_stats_df['per'] = per

    # Sort by ascending Coef.
    df_sub = df_sub.sort_values(by="Coef.").copy()
    x_coords = np.arange(len(df_sub))
    df_sub["x_coord"] = x_coords

    # Plot each significance group
    for sig_symbol, grp in df_sub.groupby("Pr(>|z|)_symbol"):
        c = SIGNIFICANCE_COLORS.get(sig_symbol, SIGNIFICANCE_COLORS['ns'])
        ms = SIGNIGICANCE_MARKERSIZE.get(sig_symbol, SIGNIGICANCE_MARKERSIZE['ns'])
        xx = grp["x_coord"].values
        yy = grp["Coef."].values
        y_down = yy - grp["l_conf"].values
        y_up   = grp["u_conf"].values - yy
        
        if pred_col == "is_correct" or "comprehension" in pred_col:
            # convert to percentage
            yy = yy * 100
            y_down = y_down * 100
            y_up = y_up * 100

        ax.errorbar(
            xx, yy,
            yerr=[y_down, y_up],
            fmt='o',
            capsize=3,
            elinewidth=1,
            capthick=2.0,
            color=c,
            markersize=ms
        )
    # Horizontal lines
    ax.axhline(0, color='black', linestyle='-', linewidth=2)  # y=0
    median_val = df_sub["Coef."].median()
    mean_val = df_sub["Coef."].mean()
    if pred_col == "is_correct" or "comprehension" in pred_col:
        # convert to percentage
        median_val = median_val * 100
        mean_val = mean_val * 100
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    def _format_val(val):
        dec_places = 2
        val_str = f"{val:.2f}"
        if round(val, 2) == 0:
            dec_places += 1
            round_val = round(val, dec_places)
            while round_val == 0:
                dec_places += 1
                round_val = round(val, dec_places)
            val_str = f"{val:.{dec_places}f}"
            return val_str
            
        return f"{val:.2f}"
    
    mean_str = _format_val(mean_val)
    median_str = _format_val(median_val)
    ax.axhline(median_val, color=EFFECT_COLOR, linestyle='--', linewidth=2)  # dashed
    y_min, y_max = ax.get_ylim()
    y_offset = 0.03 * (y_max - y_min)
    x_pos = len(x_coords) - 30  # place text near right side
    ax.text(
        x_pos,
        y_min + 2.5*y_offset,
        f"Median = {mean_str}",
        color=EFFECT_COLOR,
        va='center',
        ha='left',
        fontsize=14,
    )
    # add text of mean value at the bottom right corner
    ax.text(
        x_pos,
        y_min + y_offset,
        f"Mean = {median_str}",
        color='grey',
        va='center',
        ha='left',
        fontsize=14,
    )

    # X ticks => 1..N
    ax.set_xticks(x_coords)
    id_numbers = range(1, len(df_sub)+1)
    # ax.set_xticklabels(id_numbers, rotation=90, fontsize=9)
    # add x tick label every 10th subject
    x_ticks = [id_numbers[i] if ((i+1) % 10)== 0 else '' for i in range(len(id_numbers))]
    ax.set_xticklabels(x_ticks, fontsize=12)

    pred_col_label = PRED_COLS_SHORT_LABELS.get(pred_col, pred_col)
    ax.set_ylabel(f"{pred_col_label}\nSimplification Effect", fontsize=17)
    ax.set_xlabel(SUBJECT_TEXT_LABELS[per], fontsize=17)
    return sig_stats_df

def plot_Ele_effect_on_col_per_subject_text(
    per: str,
    L1_or_L2: str, 
    pred_cols: List[str],
    output_file_name: str,
    reading_modes = ["Gathering0", "Hunting0"],
):
    """
    Creates a grid of subplots (len(rt_cols) rows, 2 columns).
      - Rows: each RT metric
      - Columns: Gathering (left) vs. Hunting (right)
    
    Each subplot shows a per-subject coefficient for "level: Ele":
      - X-axis: subject_id, sorted by ascending 'Coef.' (the effect size)
      - Y-axis: the Ele effect (Coef.)
      - Error bars use [l_conf, u_conf].
    
    Adds:
      - a horizontal line at y=0 (solid gray)
      - a horizontal line at the mean effect across subjects (dashed or dotted)
      - coloring by significance group
      - a single legend at the bottom describing the significance colors.
    """
    # Load the two per-subject result files
    if 'is_correct' in pred_cols or 'QA_RT' in pred_cols or 'words_per_sec_based_P_RT' in pred_cols:
        file_name = f"Ele_effect_on_comprehension_per_{per}.csv"
    else:
        file_name = f"Ele_effect_on_RT_per_{per}.csv"

    n_rows = len(pred_cols)
    n_cols = len(reading_modes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11 * n_cols, 6 * n_rows), sharex=False)

    # Titles
    for j, reading_mode in enumerate(reading_modes):
        if n_cols > 1:
            axes[0, j].set_title(REGIME_LABELS[reading_mode], fontsize=16, fontweight='bold')

    sig_stats_dfs = []
    # Loop over each pred_col (row) & reading mode (column)
    for j, reading_mode in enumerate(reading_modes):
        # path
        path = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/{reading_mode}/{file_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        # Load the data
        data = pd.read_csv(path)
        
        for i, pred_col in enumerate(pred_cols):
            ax = axes[i, j] if n_cols > 1 else axes[i]
            sig_stats_df = _single_plot_per_subject_text(ax, data, pred_col, per)
            sig_stats_dfs.append(sig_stats_df)
    
    sig_stats_df = pd.concat(sig_stats_dfs)

    add_significance_legend(fig, increase_fontsize=True)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    
    # Build output path
    if per == 'subject':
        dir_name = "individual_effect"
    elif per == 'text':
        dir_name = "text_effect"
    saving_dir = Path(os.path.abspath(
        os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", dir_name)
    ))
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    sig_stats_df.to_csv(saving_dir / f"significant_stats_{output_file_name.split('.')[0]}.csv")
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / dir_name
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = overleaf_dir / output_file_name
    plt.savefig(output_pdf_path)
    plt.close()
    logger.info(f"Per-{per} Ele effects plot saved to {output_pdf_path}")
    
    # call save_latex_figure
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type=(file_name.split('.')[0]),
        reading_regimes=reading_modes,
        pred_cols=pred_cols,
        reader_type=L1_or_L2,
        per=per
    )

def plot_Ele_effect_per_subject_text_grid(
    L1_or_L2: str,
    output_file_name: str,
    reading_regime: str,
):
    grid_labels_columns = ['Reading Fluency', 'Reading Comprehension']
    grid_labels_rows = ['Per-Participant Effect', 'Per-Paragraph Effect']
    per_rows = ['subject', 'text']
    
    n_cols = len(grid_labels_columns)
    n_rows = len(grid_labels_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11 * n_cols, 6 * n_rows), sharex=False)
    
    # Titles
    for j, col_label in enumerate(grid_labels_columns):
        if n_cols > 1:
            axes[0, j].set_title(col_label, fontsize=18, fontweight='bold')
    
    sig_stats_dfs = []    
    for j, grid_col in enumerate(grid_labels_columns):
        if grid_col == 'Reading Fluency':
            pred_col = 'nonzero_TF'
            on = "RT"
        elif grid_col == 'Reading Comprehension':
            pred_col = 'is_correct'
            on = "comprehension"
        
        for i, per in enumerate(per_rows):
            # path
            file_name = f"Ele_effect_on_{on}_per_{per}.csv"
            path = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/{reading_regime}/{file_name}"
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            # Load the data
            data = pd.read_csv(path)
            
            ax = axes[i, j] if n_cols > 1 else axes[i]
            sig_stats_df = _single_plot_per_subject_text(ax, data, pred_col, per)
            sig_stats_dfs.append(sig_stats_df)
    
    sig_stats_df = pd.concat(sig_stats_dfs)
    
    fig = add_significance_legend(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    # Build saving path
    saving_dir = Path(
        os.path.abspath(os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", "effect_main_grids"))
    )
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    sig_stats_df.to_csv(saving_dir / f"significant_stats_{output_file_name.split('.')[0]}.csv")
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / "effect_per_text_subject_grid"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    overleaf_path = overleaf_dir / output_file_name
    plt.savefig(overleaf_path)
    
    plt.close() 
    logger.info(f"Ele effects main grid saved to {output_pdf_path}")
    
    # call save_latex_figure
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type="Ele_effects_grid_per_text_subject",
        reading_regimes=reading_regime,
        pred_cols=['nonzero_TF', 'is_correct'],
        reader_type=L1_or_L2
    )

def plot_Ele_effect_per_subject_text_grid_SM(
    L1_or_L2: str,
    reading_regime: str,
    on: str,
    output_file_name: str,
    comp_cols: List[str]=None,
    rt_cols: List[str]=None,
):
    grid_labels_columns = ['Per-Participant Effect', 'Per-Paragraph Effect']
    per_columns = ['subject', 'text']
    
    n_cols = len(grid_labels_columns)
    if comp_cols is not None:
        pred_cols = comp_cols
        on = "comprehension"
    elif rt_cols is not None:
        pred_cols = rt_cols
        on = "RT"
    else:
        raise ValueError("Either comp_cols or rt_cols must be provided")
    
    n_rows = len(pred_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11 * n_cols, 6 * n_rows), sharex=False)
    
    # Titles
    for j, col_label in enumerate(grid_labels_columns):
        if n_cols > 1:
            ax = axes[0, j] if n_rows > 1 else axes[j]
            ax.set_title(col_label, fontsize=18, fontweight='bold')
    
    sig_stats_dfs = []        
    for j, per in enumerate(per_columns):
        # path
        file_name = f"Ele_effect_on_{on}_per_{per}.csv"
        path = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/{reading_regime}/{file_name}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Load the data
        data = pd.read_csv(path)
        
        for i, pred_col in enumerate(pred_cols):
            if pred_col == 'words_per_sec_based_P_RT':
                speed_file_name =f"Ele_effect_on_comprehension_per_{per}.csv"
                speed_path = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/{reading_regime}/{speed_file_name}"
                data = pd.read_csv(speed_path)
            else:
                data = pd.read_csv(path)
            ax = axes[i, j] if n_rows > 1 else axes[j]
            sig_stats_df = _single_plot_per_subject_text(ax, data, pred_col, per)
            sig_stats_dfs.append(sig_stats_df)
    
    sig_stats_df = pd.concat(sig_stats_dfs)
    
    
    fig = add_significance_legend(fig)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    
    # Build saving path
    saving_dir = Path(
        os.path.abspath(os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", "effect_main_grids"))
    )
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    sig_stats_df.to_csv(saving_dir / f"significant_stats_{output_file_name.split('.')[0]}.csv")
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / "effect_per_text_subject_grid"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    overleaf_path = overleaf_dir / output_file_name
    plt.savefig(overleaf_path)
    
    plt.close() 
    logger.info(f"Ele effects main grid saved to {output_pdf_path}")
    
    # call save_latex_figure
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type="Ele_effects_grid_on_col_per_text_subject_SM",
        reading_regimes=reading_regime,
        pred_cols=pred_cols,
        effect_on=on,
        reader_type=L1_or_L2
    )

def _add_formula(formula_text):
    if formula_text is not None:
        # Wrap the formula to, say, width=90
        wrapped_formula = textwrap.fill(str(formula_text), width=90)
        plt.figtext(
            0.5, 0.04,  # slightly above the legend
            f"Formula:\n{wrapped_formula}",
            ha='center', va='bottom', fontsize=9
        )

def _load_on_col_by_col_files(base_path, effect_on, by_col, per, per_type, by_col_type, reading_regime, fit_only=False):
    effects_vals_file = f'per_{per}_ele_effects_on_{effect_on}_with_{by_col}.csv'
    results_file = f'per_{per}_Ele_effect_on_{effect_on}_by_{by_col}.csv'
    
    effects_vals_path = base_path / reading_regime / f"{per_type}_effect_{by_col_type}" / effects_vals_file
    fit_res_path = base_path / reading_regime / f"{per_type}_effect_{by_col_type}" / results_file

    if not fit_res_path.exists():
        raise FileNotFoundError(f"Results file not found: {fit_res_path}")
    fit_df = pd.read_csv(fit_res_path)
    fit_df = add_p_val_symbols(fit_df, p_val_col="Pearson_pval")
    if fit_only:
        return None, fit_df
    
    if not effects_vals_path.exists():
        raise FileNotFoundError(f"Effects file not found: {effects_vals_path}")
    effect_vals_df = pd.read_csv(effects_vals_path)
    return effect_vals_df, fit_df

def _single_reg_by_col_plot(ax, sub_df, fit_df, pred_col, by_col, y_axis_label):
    """
    Plots individual effects for the given condition (Gathering/Hunting).
    """
    if sub_df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if pred_col == "is_correct" or "comprehension" in pred_col:
        # convert Ele_effect to percentage
        sub_df = sub_df.reset_index(drop=True)
        sub_df["Ele_effect"] = sub_df["Ele_effect"] * 100

    # (A) Plot each subject as a dot, colored by significance
    # start with ns data
    ns_df = sub_df[sub_df["Pr(>|z|)_symbol"] == 'ns']
    c = SIGNIFICANCE_COLORS.get('ns', SIGNIFICANCE_COLORS['ns'])
    xx = ns_df[by_col].values
    yy = ns_df["Ele_effect"].values
    ax.plot(xx, yy, 'o', color=c, markersize=4)
    # plot sig data
    not_ns_df = sub_df[sub_df["Pr(>|z|)_symbol"] != 'ns']
    not_ns_df = not_ns_df.sort_values(by="Pr(>|z|)_symbol")
    
    # round by_col, Ele_effect to 2 decimal places
    if not_ns_df["Ele_effect"].min() < 1:
        not_ns_df["Ele_effect"] = not_ns_df["Ele_effect"].round(4)
    else:
        not_ns_df["Ele_effect"] = not_ns_df["Ele_effect"].round(2)
    if not_ns_df[by_col].min() < 1:
        not_ns_df[by_col] = not_ns_df[by_col].round(4)
    else:
        not_ns_df[by_col] = not_ns_df[by_col].round(2)
    
    
    # 1) Group by the x,y pair => count how many rows share the same (by_col, Ele_effect)
    #    so each unique x,y is aggregated
    agg_df = (
            not_ns_df
            .groupby([by_col, "Ele_effect"], as_index=False)
            .size()  # name the freq 'size'
        )
    print(f'n rows in agg_df {len(agg_df)} | by_col = {by_col}')
    
    for sig_symbol, grp in not_ns_df.groupby("Pr(>|z|)_symbol"):
        c = SIGNIFICANCE_COLORS.get(sig_symbol, SIGNIFICANCE_COLORS['ns'])

        # 2) For each unique x,y, we have a "size" column telling how many rows overlap
        #    We'll plot one marker with marker size = freq
        for row in grp.itertuples(index=False):
            xval = getattr(row, by_col)
            yval = row.Ele_effect
            if agg_df['size'].max() == 1:
                freq = 1
            else:
                freq = agg_df[(agg_df[by_col]==xval)&(agg_df["Ele_effect"]==yval)]['size'].item()
            # choose a marker size based on freq
            # for example, ms = freq or ms = 5 + freq*1.5, etc.
            ms = 2 + freq * 2
            
            ax.plot(xval, yval, 'o', color=c, markersize=ms)

    # (B) Add regression line + CI using seaborn
    sns.regplot(
        x=by_col, y="Ele_effect",
        data=sub_df,
        ax=ax,
        scatter=False,
        ci=95,
        line_kws={"color": EFFECT_COLOR, "linestyle": "--", "linewidth": 2}
    )

    # (C) Add horizontal line at y=0
    ax.axhline(0, color='black', linestyle='-', linewidth=1)

    # (D) Show slope significance from fit_df
    slope_row = fit_df[(fit_df["pred_col"] == pred_col) & (fit_df["Name"] == by_col)]
    _add_slope_star(slope_row, ax)
    
    # (E) Set labels
    by_col_label = BY_COL_LABELS.get(by_col, by_col)
    ax.set_xlabel(by_col_label, fontsize=13)
    ax.set_ylabel(y_axis_label, fontsize=13)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _single_by_col_bar_plot(ax, sub_df, fit_df, pred_col, by_col, y_axis_label):
    if sub_df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    if pred_col == "is_correct" or "comprehension" in pred_col:
        # convert Ele_effect to percentage
        sub_df = sub_df.reset_index(drop=True)
        sub_df["Ele_effect"] = sub_df["Ele_effect"] * 100
    
    # replace in sub_df: 1-> Yes and 0-> No
    sub_df = sub_df.reset_index(drop=True)
    sub_df[by_col] = sub_df[by_col].replace({1: "Yes", 0: "No"})
    
    x_vals = sub_df[by_col].unique()
    means = sub_df.groupby(by_col)["Ele_effect"].mean()
    if means.iloc[0] < 1:
        means_str = [f"{val:.3f}" for val in means]
    else:
        means_str = [f"{val:.2f}" for val in means]
    err_ups = []
    err_downs = []
    for val, val_df in sub_df.groupby(by_col):
        low_ci, up_ci, _ = get_mean_ci(val_df, "Ele_effect")
        val_mean = means.loc[val]
        err_up = up_ci - val_mean
        err_down = val_mean - low_ci
        err_ups.append(err_up)
        err_downs.append(err_down)
    yerr = np.vstack([err_downs, err_ups])
    # colors - BASE_COLOR for Yes and BASE_COLOR_2 for No
    colors = [BASE_COLOR, BASE_COLOR_2]
    
    # 3) Plot bars, color by x_val
    ax.bar(
        x_vals,
        means,
        yerr=yerr,
        capsize=5,
        width=0.3,
        color=colors,
    )
    
    # 4) Place numeric labels above bars
    overall_min = (means - yerr[0]).min()
    overall_max = (means + yerr[1]).max()
    # Force y=0 into the axis range
    overall_min = min(overall_min, 0)
    overall_max = max(overall_max, 0)
    # Add some padding so bars / text aren’t on the very edge
    up_padding = 0.05 * (overall_max - overall_min)  # 5% padding
    # down_padding = 0.1 * (overall_max - overall_min)  # 5% padding
    y_min = overall_min
    y_max = overall_max + up_padding

    # add text
    for j, _ in enumerate(x_vals):
        offset = 0.02 * (y_max - y_min)
        mean_val = means.iloc[j]
        # where to place text
        if mean_val < 0:
            text_y = y_max
        else:
            text_y = mean_val + err_ups[j] + offset
        ax.text(
            j,
            text_y,
            means_str[j],
            color='black',
            ha='center', va='bottom',
            fontsize=10
        )
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    y_min, y_max = ax.get_ylim()
    if y_max < 0.01:
        new_y_max = 0.01 + 0.4 * abs(y_min)
        ax.set_ylim(y_min, new_y_max)
    else:
        new_y_max = y_max + 0.25 * (y_max - y_min)
        ax.set_ylim(y_min, new_y_max)
    
    # expand a bit x axis
    ax.set_xlim(-0.5, 1.5)
    
    # (C) Add horizontal line at y=0
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    # (E) Set labels
    by_col_label = BY_COL_LABELS.get(by_col, by_col)
    ax.set_xlabel(by_col_label, fontsize=13)
    ax.set_ylabel(y_axis_label, fontsize=13)
    
    # (D) Show slope significance from fit_df
    slope_row = fit_df[(fit_df["pred_col"] == pred_col) & (fit_df["Name"] == by_col)]
    _add_slope_star(slope_row, ax)
    
    # tight layout
    plt.tight_layout()

def plot_Ele_effect_per_subject_text_on_col_by_col(
    per: str,
    L1_or_L2: str, 
    pred_cols: List[str], 
    by_col: str, 
    effect_on: str, 
    SM_suffix: str,
    reading_modes = ["Gathering0", "Hunting0"],
    ):
    """
    Creates a grid: len(pred_cols) rows, 2 columns:
      - Left  => Gathering
      - Right => Hunting

    Each subplot:
      X-axis = speed_col (e.g., speed_col)
      Y-axis = Ele_effect
      Dots colored by significance
      A regression line + 95% CI band from sns.regplot (with scatter=False).
    """
    by_col_type = _get_by_col_type(by_col)
    per_type = _get_per_type(per)
    base_path = src_path / f'Eye_metrics/{L1_or_L2}/mixed_effects'

    n_rows = len(pred_cols)
    n_cols = len(reading_modes)
    # if by_col in ['is_student', 'is_secondary', 'is_undergrad', 'is_postgrad', 'edu_level_num']:
    #     width = REG_PLOT_WIDTH / 1.3
    #     height = REG_PLOT_HEIGHT / 1.5
    # else:
    #     width = REG_PLOT_WIDTH
    #     height = REG_PLOT_HEIGHT
    width = REG_PLOT_WIDTH
    height = REG_PLOT_HEIGHT
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows), sharex=False)

    # Titles
    for j, reading_mode in enumerate(reading_modes):
        if n_cols > 1:
            axes[0, j].set_title(REGIME_LABELS[reading_mode], fontsize=13, fontweight='bold')
    
    # Loop over each pred_col (row) & reading mode (column)
    for i, pred_col in enumerate(pred_cols):
        for j, reading_mode in enumerate(reading_modes):
            effect_vals_df, fit_df = _load_on_col_by_col_files(
                base_path, 
                effect_on, 
                by_col, 
                per,
                per_type, 
                by_col_type, 
                reading_mode)
            
            ax = axes[i, j] if n_cols > 1 else axes[i]
            pred_col_label = PRED_COLS_SHORT_LABELS[pred_col]
            y_axis_label = f"{pred_col_label}\nSimplification Effect"
            if by_col in ['is_student', 'is_secondary', 'is_undergrad', 'is_postgrad', 'edu_level_num']:
                # catgeorical by_col
                _single_by_col_bar_plot(
                    ax, 
                    sub_df=effect_vals_df[effect_vals_df["pred_col"] == pred_col], 
                    fit_df=fit_df[fit_df["pred_col"] == pred_col], 
                    pred_col=pred_col,
                    by_col=by_col,
                    y_axis_label=y_axis_label
                )
                add_significance_legend(fig, with_colors=False, handlelength=0.5)
                plt.tight_layout(rect=[0, 0.03, 1, 1])
            else:
                # continuous by_col
                _single_reg_by_col_plot(
                    ax, 
                    sub_df=effect_vals_df[effect_vals_df["pred_col"] == pred_col], 
                    fit_df=fit_df[fit_df["pred_col"] == pred_col], 
                    pred_col=pred_col,
                    by_col=by_col,
                    y_axis_label=y_axis_label
                )
                add_significance_legend(fig)
                plt.tight_layout(rect=[0, 0.02, 1, 1])

    # 3) Build output path
    saving_dir = base_path / f"{per_type}_effect_{by_col_type}"
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_file_name = f"per_{per}_Ele_effect_on_{effect_on}_by_{by_col}{SM_suffix}.pdf"
    output_pdf_path = saving_dir / output_file_name
    # Save the plot
    plt.savefig(output_pdf_path)
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / f"{per_type}_effect_{by_col_type}"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = overleaf_dir / output_file_name
    plt.savefig(output_pdf_path)
    plt.close()
    logger.info(f"plot_{per_type}_effect_{by_col_type} saved to {output_pdf_path}")

    # call save_latex_figure
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type="reg_plot_on_col_by_col",
        reading_regimes=reading_modes,
        pred_cols=pred_cols,
        effect_on=effect_on,
        by_col=by_col,
        per=per,
        reader_type=L1_or_L2
    )

def plot_reg_lines_grid(
    L1_or_L2: str,
    rt_col: str,
    reading_regime: str,
    per: str,
    by_cols: str,
    SM_suffix: str,
):
    per_type = _get_per_type(per)
    base_path = src_path / f'Eye_metrics/{L1_or_L2}/mixed_effects'
    
    if rt_col == "nonzero_TF":
        grid_labels_columns = ['Reading Fluency - Total Fixation Duration\n', 'Reading Comprehension - QA Accuracy\n']
        y_axis_label_left = "Simplification Effect (ms)"
    elif rt_col == "SkipTotal":
        grid_labels_columns = ['Reading Fluency - Skip Probability\n', 'Reading Comprehension - QA Accuracy\n']
        y_axis_label_left = "Simplification Effect"
    elif rt_col == "RegCountTotal":
        grid_labels_columns = ['Reading Fluency - Regression Rate\n', 'Reading Comprehension - QA Accuracy\n']
        y_axis_label_left = "Simplification Effect"
        
    n_cols = len(grid_labels_columns)
    n_rows = len(by_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(REG_PLOT_WIDTH * n_cols, REG_PLOT_HEIGHT * n_rows), 
                             sharex=False)
    
    # Titles
    for j, col_label in enumerate(grid_labels_columns):
        if n_cols > 1:
            axes[0, j].set_title(col_label, fontsize=14, fontweight='bold')
            
    for j, grid_col in enumerate(grid_labels_columns):
        if j==0:
            effect_on = "RT"
            pred_col = rt_col
        elif j==1:
            effect_on = "comprehension"
            pred_col = 'is_correct'
            
        for i, by_col in enumerate(by_cols):
            by_col_type = _get_by_col_type(by_col)
            
            effect_vals_df, fit_df = _load_on_col_by_col_files(
                base_path, 
                effect_on, 
                by_col, 
                per,
                per_type, 
                by_col_type, 
                reading_regime)
            
            ax = axes[i, j] if n_cols > 1 else axes[i]
            pred_col_label = PRED_COLS_SHORT_LABELS[pred_col]
            y_axis_label = f"{pred_col_label}\nSimplification Effect"
            if j==0:
                y_axis_label = y_axis_label_left
            if j==1:
                y_axis_label = "Simplification Effect (%)"
            
            if by_col in ['is_student', 'is_secondary', 'is_undergrad', 'is_postgrad', 'edu_level_num']:
                # catgeorical by_col
                _single_by_col_bar_plot(
                    ax, 
                    sub_df=effect_vals_df[effect_vals_df["pred_col"] == pred_col], 
                    fit_df=fit_df[fit_df["pred_col"] == pred_col], 
                    pred_col=pred_col,
                    by_col=by_col,
                    y_axis_label=y_axis_label
                )
            else:
                _single_reg_by_col_plot(
                    ax, 
                    sub_df=effect_vals_df[effect_vals_df["pred_col"] == pred_col], 
                    fit_df=fit_df[fit_df["pred_col"] == pred_col], 
                    pred_col=pred_col,
                    by_col=by_col,
                    y_axis_label=y_axis_label
                )
    
    fig = add_significance_legend(fig)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # 3) Build output path
    saving_dir = base_path / f"{per_type}_effect"
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_file_name = f"reg_lines_grid_per_{per}_{rt_col}_Ele_effect{SM_suffix}.pdf"
    output_pdf_path = saving_dir / output_file_name
    # Save the plot
    plt.savefig(output_pdf_path)
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / f"{per_type}_effect"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = overleaf_dir / output_file_name
    plt.savefig(output_pdf_path)
    plt.close()
    logger.info(f"grid_per_{per}_{rt_col}_Ele_effect{SM_suffix} saved to {output_pdf_path}")
    
    # call save_latex_figure
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type="reg_lines_grid",
        reading_regimes=reading_regime,
        pred_cols=['nonzero_TF', 'is_correct'],
        per=per,
        reader_type=L1_or_L2
    )

def _shorten_coef_name(name: str) -> str:
    """
    If ' & ' is in the name, split by ' & ', 
    replace known words using SHORTEN_DICT, 
    then rejoin with ' & '.
    Otherwise, also check if the single name matches something.
    """
    parts = name.split(" & ")
    new_parts = []
    for p in parts:
        # If p is exactly one of the keys in SHORTEN_DICT, replace it
        if p in SIMPLIFICATION_TYPES_SHORT_LABELS:
            new_parts.append(SIMPLIFICATION_TYPES_SHORT_LABELS[p])
        else:
            new_parts.append(p)
    # rejoin
    new_name = " & ".join(new_parts)
    return new_name

def plot_simplification_type_effect_on_RT(
    L1_or_L2: str, 
    output_file_name: str, 
    formula_version: str, 
    rt_cols: List[str], 
    predict_type: str,
    reading_modes = ["Gathering0", "Hunting0"],
    ):
    """
    Grid Plot. RT_cols x 2 (Gathering0, Hunting0).
    
    Each subplot:
      y-axis: bar with coefficient of simplification-type effect, with error bars from [l_conf, u_conf].
        - If 'Std. Error' is NaN/null, then place "No data" text above that bar.
      x-axis: simplification types (the CSV rows). 
        - If name is very long or has " & ", shorten with `_shorten_coef_name`.
        - Color each bar by 'Pr(>|z|)_symbol' using SIGNIFICANCE_COLORS.
        - Add a small text annotation of significance on or near each bar as well (optional).
      Add a horizontal line at y=0.
      At the bottom of the figure:
        - A single legend explaining significance colors.
        - The formula text from the row where Name=='(Intercept)'.
    """
    # 1) CSV paths
    Hunting0_results_file = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/Hunting0/edit_effect/Edit_effect_on_RT_formula={formula_version}.csv"
    Gathering0_results_file = f"src/Eye_metrics/{L1_or_L2}/mixed_effects/Gathering0/edit_effect/Edit_effect_on_RT_formula={formula_version}.csv"

    # 2) Build output path
    saving_dir = Path(os.path.abspath(
        os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", "edit_effect_on_RT")
    ))
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name

    # 3) Load each CSV into DataFrames
    if not os.path.exists(Hunting0_results_file):
        raise FileNotFoundError(f"File not found: {Hunting0_results_file}")
    if not os.path.exists(Gathering0_results_file):
        raise FileNotFoundError(f"File not found: {Gathering0_results_file}")

    hunt_df = pd.read_csv(Hunting0_results_file)
    gather_df = pd.read_csv(Gathering0_results_file)

    # 4) Create subplots: rows=len(rt_cols), cols=2
    n_rows = len(rt_cols)
    n_cols = len(reading_modes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=False)
 
    if formula_version == "4":
        height_sub_plot = 4
        width_plot = 10
    elif formula_version == "7":
        height_sub_plot = 5
        width_plot = 12
    else:
        height_sub_plot = 4
        width_plot = 10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_plot, height_sub_plot*n_rows), sharex=False)

    # Titles
    for j, reading_mode in enumerate(reading_modes):
        if n_cols > 1:
            axes[0, j].set_title(REGIME_LABELS[reading_mode], fontsize=13, fontweight='bold')


    # We'll store the formula from the (Intercept) row (we only need it once).
    formula_text = None

    # For each RT col (row in the grid)
    for i, rt_col in enumerate(rt_cols):
        for j, reading_mode in enumerate(reading_modes):
            if reading_mode == "Gathering0":
                regime_df = gather_df
                regime_label = "Ordinary Reading"
            elif reading_mode == "Hunting0":
                regime_df = hunt_df
                regime_label = "Information Seeking"
            else:
                raise ValueError(f"Unknown reading_mode: {reading_mode}")
            
            ax = axes[i, j] if n_cols > 1 else axes[i]

            # Filter data
            sub_df = regime_df[regime_df["pred_col"] == rt_col].copy()
            if sub_df.empty:
                ax.text(0.5, 0.5, f"No data for {rt_col}", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Possibly retrieve formula if not set
            if formula_text is None:
                intercept_row = sub_df[sub_df["Name"] == "(Intercept)"]
                if not intercept_row.empty and "formula" in intercept_row.columns:
                    formula_text = intercept_row["formula"].iloc[0]

            # Exclude the intercept row if we only want to show actual simplification types
            plot_df = sub_df[sub_df["Name"] != "(Intercept)"].copy()
            if plot_df.empty:
                ax.text(0.5, 0.5, "No rows except intercept", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            x_vals = np.arange(len(plot_df))
            means = plot_df["Coef."].values
            l_conf = plot_df["l_conf"].values
            u_conf = plot_df["u_conf"].values
            err_down = means - l_conf
            err_up   = u_conf - means
            yerr = np.vstack([err_down, err_up])

            sig_syms = plot_df["Pr(>|z|)_symbol"].values
            # color each bar
            bar_colors = [SIGNIFICANCE_COLORS.get(sym, SIGNIFICANCE_COLORS["ns"])
                          for sym in sig_syms]

            # Plot bars
            ax.bar(
                x_vals, means,
                yerr=yerr,
                capsize=5,
                color=bar_colors,
                edgecolor='black'
            )

            # If "Std. Error" is NaN => "No data" (rotated 90°)
            for row_idx, row in plot_df.iterrows():
                local_x = plot_df.index.tolist().index(row_idx)
                if pd.isna(row["Std. Error"]):
                    ax.text(
                        local_x,
                        means[local_x],
                        "No data",
                        ha='center',
                        va='bottom',
                        color='red',
                        fontsize=8,
                        rotation=90
                    )

            # X tick labels, rotated 90, smaller font
            # Also shorten names if they contain " & "
            new_names = []
            for name in plot_df["Name"]:
                if " & " in name:
                    new_name = _shorten_coef_name(name)
                else:
                    new_name = name
                new_names.append(new_name)

            ax.set_xticks(x_vals)
            ax.set_xticklabels(new_names, rotation=90, ha='center', fontsize=8)

            # Horizontal line at y=0
            ax.axhline(0, color='grey', linewidth=1, linestyle='-')

            # Label
            if 'diff' in predict_type:
                ax.set_ylabel(f"diff {rt_col}\nSimplification Effect", fontsize=9)
            else:  
                ax.set_ylabel(f"{rt_col}\nSimplification Effect", fontsize=9)
            if i == 0:
                ax.set_title(regime_label, fontsize=11, fontweight='bold')

    add_significance_legend(fig)

    # Add formula text, two-line or more if it's too long
    if formula_text is not None:
        # Wrap the formula to, say, width=90
        wrapped_formula = textwrap.fill(str(formula_text), width=90)
        plt.figtext(
            0.5, 0.04,  # slightly above the legend
            f"Formula:\n{wrapped_formula}",
            ha='center', va='bottom', fontsize=9
        )

    # Adjust layout, leaving space at bottom for the legend + formula
    plt.tight_layout(rect=[0, 0.15, 1, 1])  # push the plot upward
    plt.savefig(output_pdf_path)
    # save also to overleaf
    saving_dir = OVERLEAF_PATH / "edit_effect_on_RT"
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    plt.close()
    logger.info(f"Plot saved to {output_pdf_path}")

def _add_level_legend_to_subplot(ax, loc='upper right'):
    """
    Example of how to add an 'Adv' vs. 'Ele' legend in a single subplot 'ax'.
    """
    # Create the patch handles for level legend
    adv_patch = mpatches.Patch(
        facecolor='white',
        edgecolor='black',
        hatch='',      # no hatch for 'Adv'
        label=LEVEL_LABELS['Adv']
    )
    ele_patch = mpatches.Patch(
        facecolor='white',
        edgecolor='black',
        hatch='///',   # diagonal hatch for 'Ele'
        label=LEVEL_LABELS['Ele']
    )
    handles = [adv_patch, ele_patch]

    # Add the legend to this specific axes
    ax.legend(
        handles=handles,
        loc=loc,   # or wherever you prefer
        fontsize=8,
        frameon=False
    )
    
def _get_formula(coef_df, rt_col):
    intercept_row = coef_df[coef_df["Name"] == "(Intercept)"]
    if not intercept_row.empty and "formula" in intercept_row.columns:
        formula_text = intercept_row["formula"].iloc[0]
        # replace words in formula_text
        replace_dict = {
            'subject_id': 'participant_id',
            'text_id': 'paragraph_id',
            # 'freq': 'frequency',
            # 'len': 'length',
            # 'surp': 'surprisal',
            rt_col: 'RT'
        }
        # replace using dict
        for prev_str, new_str in replace_dict.items():
            formula_text = formula_text.replace(prev_str, new_str)
        return formula_text
        
def _process_rt_response_df(original_data, rt_col, surp_col):
    # filter
    data = original_data[(original_data["surp_col"]==surp_col) & (original_data["pred_col"]==rt_col)].reset_index(drop=True)

    # replace level str "['Ele']" with 'Ele', "['Adv']" with 'Adv'
    data['level'] = data['level'].str.replace("['Ele']", 'Ele').str.replace("['Adv']", 'Adv')
    
    # sig_df - get rows of interactions "is_Ele &"
    sig_df = data[data["Name"].str.contains("is_Ele &")].copy()

    # coef_df - get rows where 'level' is in ['Ele','Adv']
    coef_df = data[data["level"].isin(["Ele","Adv"])]
    # before removing intercept
    formula_text = _get_formula(coef_df, rt_col)
    # dont need coef for (Intercept)
    coef_df = coef_df[coef_df["Name"] != "(Intercept)"].copy()
    # filter out coef names with 'prev' in them
    coef_df = coef_df[~coef_df["Name"].str.contains("prev")]
    
    if coef_df.empty or sig_df.empty:
        raise RuntimeError(("No data after filtering"))
    return coef_df.reset_index(drop=True), sig_df.reset_index(drop=True), formula_text

def _add_sign_for_coefs(
    ax,
    sig_df, 
    adv_tops, 
    ele_tops, 
    coef_names,
    offset_Adv,
    offset_Ele,
    bracket_y=None,
    bracket_height=0.02
    ):
    # Now add significance bracket for each "freq","len","surp" or any?
    # The requirement says: "only for freq, len, surp"
    # We look up a row with Name==f"is_Ele & {coef_name}" to find the symbol
    # Then bracket from adv x => ele x
    # The bracket_y is max(adv_tops, ele_tops) + some offset
    # We'll do it only if both adv & ele exist
    target_coefs = ["freq", "len", "surp"]
    for cname in target_coefs:
        if cname in adv_tops and cname in ele_tops:
            # find significance row
            name_isEle = f"is_Ele & {cname}"
            # or perhaps your CSV stores that row differently
            row_diff = sig_df[sig_df["Name"] == name_isEle]
            if row_diff.empty:
                # no difference row => skip
                continue
            diff_sym = row_diff["Pr(>|z|)_symbol"].iloc[0]
            if diff_sym == "" or pd.isna(diff_sym):
                continue

            # bracket from adv_x to ele_x
            ix = coef_names.index(cname)
            x_adv = ix + offset_Adv
            x_ele = ix + offset_Ele
            _, y_max = ax.get_ylim()
            if not bracket_y:
                bracket_y = y_max * 0.9
            # bracket_y = max(adv_tops[cname], ele_tops[cname]) * 1.1
            add_significance_bracket(
                ax=ax, x1=x_adv, x2=x_ele,
                y=bracket_y,
                text=diff_sym,
                bracket_height=bracket_height,
            )
    return ax

def _single_rt_response_plot(ax, coef_df, sig_df, rt_col):
    if coef_df.empty:
        ax.text(0.5, 0.5, f"No data for {rt_col}", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # We'll store bar x positions: 
    offset_Adv = -0.1
    offset_Ele = +0.1
    width = 0.2

    # We'll keep track of bar top positions so we can place significance brackets
    adv_tops = {}
    ele_tops = {}

    coef_names = coef_df["Name"].unique().tolist()
    for ix, cname in enumerate(coef_names):
        # For each cname, we have 2 rows: level=Adv, level=Ele
        c_df = coef_df[(coef_df["Name"] == cname)]
        # For each level in c_df
        for lvl in ["Adv","Ele"]:
            row = c_df[c_df["level"] == lvl]
            if row.empty:
                # No row for that level => skip
                continue

            mean_val = row["Coef."].iloc[0]
            l_conf   = row["l_conf"].iloc[0]
            u_conf   = row["u_conf"].iloc[0]
            err_down = mean_val - l_conf
            err_up   = u_conf - mean_val
            yerr     = np.array([[err_down],[err_up]])  # shape=2,1

            colorbar = BASE_COLOR

            # bar position
            if lvl=="Adv":
                bar_x = ix + offset_Adv
                hatch_style = LEVEL_HATCH["Adv"]
            else:
                bar_x = ix + offset_Ele
                hatch_style = LEVEL_HATCH["Ele"]

            # Plot bar
            ax.bar(
                bar_x, mean_val,
                yerr=yerr,
                capsize=2,
                color=colorbar,
                edgecolor='black',
                hatch=hatch_style,
                width=width
            )

            # If "Std. Error" is NaN => "No data"
            if pd.isna(row["Std. Error"].iloc[0]):
                ax.text(
                    bar_x, mean_val, 
                    "No data",
                    ha='center', va='bottom',
                    color='red', fontsize=8
                )

            # Store top
            top_val = mean_val + max(0, err_up)
            if lvl=="Adv":
                adv_tops[cname] = top_val
            else:
                ele_tops[cname] = top_val
    
    # remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    y_min, y_max = ax.get_ylim()
    if y_max < 1:
        new_y_max = max(0.03, y_max) * 1.3
        ax.set_ylim(y_min, new_y_max)
        bracket_y = new_y_max * 0.9 + y_min * 0.1
        bracket_height = 0.0005
    else:
        new_y_max = y_max + 0.1 * (y_max - y_min)
        ax.set_ylim(y_min, new_y_max)
        bracket_y = new_y_max * 0.9
        bracket_height = 0.02
    
    # Add significance brackets
    ax = _add_sign_for_coefs(
        ax, sig_df, adv_tops, ele_tops, coef_names, offset_Adv, offset_Ele, 
        bracket_y, bracket_height
    )

    # X tick = the center between the two bars => just ix
    ax.set_xticks(np.arange(len(coef_names)))
    coef_names_labels = [_get_label_linguistic_col(cname) for cname in coef_names]
    ax.set_xticklabels(coef_names_labels, rotation=90, fontsize=12)

    # horizontal line at y=0
    ax.axhline(0, color='grey', linestyle='-', linewidth=1)

    ax.set_ylabel("Coefficient", fontsize=11)
    
    
def plot_linguistic_effect_on_RT(
    L1_or_L2: str, 
    output_file_name: str, 
    formula_version: str, 
    rt_cols: list, 
    normed_RT: bool, 
    surp_col: str,
    reading_modes = ["Gathering0", "Hunting0"],
    ):
    """
    Grid Plot. (len(rt_cols) rows) x 2 columns (Gathering0, Hunting0).

    In each subplot:
      - y-axis: bars for each linguistic coefficient name (freq, len, surp, etc.), 
                with error bars from [l_conf, u_conf].
      - We have 2 bars per coefficient: one for 'Adv', one for 'Ele'.
        * Distinguish them by different hatch patterns (LEVEL_HATCH).
        * Color them by significance symbol (SIGNIFICANCE_COLORS).
      - If 'Std. Error' is NaN => "No data" above that bar.
      - We draw a bracket above each pair (Adv, Ele) for the same coefficient name if there's
        a row with Name=f"is_Ele & {coef_name}" that indicates their difference significance.
      - Grey horizontal line at y=0.
    At bottom:
      - Legend for significance colors.
      - The formula text from row Name=='(Intercept)'.
    """

    n_cols = len(rt_cols)
    n_rows = len(reading_modes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows), sharex=False)

    # titles
    for j, rt_col in enumerate(rt_cols):
        if n_cols > 1:
            title_label = f"{PRED_COLS_FULL_LABELS[rt_col]}"
            title_label = title_label.replace("(ms)", "")
            wrapped_title = wrap_label(title_label, 20)
            ax = axes[0, j] if n_rows > 1 else axes[j]
            ax.set_title(wrapped_title, fontsize=14, fontweight='bold')

    # Loop over each RT column (row in the grid)
    formula_text = None
    for i, reading_mode in enumerate(reading_modes):
        path = (
            f"src/Eye_metrics/{L1_or_L2}/mixed_effects/{reading_mode}/RT_response/"
            f"RT_response_formula={formula_version}_norm={normed_RT}.csv"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        data = pd.read_csv(path)
        
        for j, rt_col in enumerate(rt_cols):
            coef_df, sig_df, formula_text = _process_rt_response_df(data, rt_col, surp_col)
            ax = axes[i, j] if n_rows > 1 else axes[j]
            # plot
            _single_rt_response_plot(ax, coef_df, sig_df, rt_col)

    _add_level_legend_to_subplot(fig, loc="lower right")
    # _add_formula(formula_text)
    # plt.tight_layout(rect=[0, 0.09, 1, 1])  # space for legend & formula
    
    if len(rt_cols) == 2:
        handlelength = 0.3
        bbox_to_anchor=(0.4, 0.0)
        plt.tight_layout(rect=[0, 0.06, 0.85, 1])  # space for legend
    else:
        handlelength = 0.5
        bbox_to_anchor=(0.5, 0.0)
        plt.tight_layout(rect=[0, 0.06, 1, 1])  # space for legend
        
    add_significance_legend(fig, with_colors=False, handlelength=handlelength, bbox_to_anchor=bbox_to_anchor, fontsize_legend_text=10)
    
    # Build output path
    saving_dir = Path(os.path.abspath(
        os.path.join("src", "Eye_metrics", L1_or_L2, "mixed_effects", "RT_response")
    ))
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / "RT_response"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = overleaf_dir / output_file_name
    plt.savefig(output_pdf_path)
    plt.close()
    logger.info(f"Plot saved to {output_pdf_path}")
    
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type="RT_response",
        reading_regimes=reading_modes,
        rt_cols=rt_cols,
        reader_type=L1_or_L2,
    )
    
def _single_R_bar_plot(
    ax, results_dict, 
    pred_col, pred_col_label, 
    by_cols, by_cols_labels,
    output_file_name):
    
    r_vals = []
    p_vals = []
    
    for by_col in by_cols:
        res_df = results_dict[by_col]
        # filter by pred_col
        res_df = res_df[res_df["pred_col"] == pred_col]
        # get r_val and sig_symbol
        assert res_df['Pearson_r'].nunique() == 1
        r_val = res_df["Pearson_r"].iloc[0]
        p_val = res_df["Pearson_pval"].iloc[0]
        r_vals.append(r_val)
        p_vals.append(p_val)
        
    p_vals_df = pd.DataFrame(p_vals, columns=["p_val"])
    sig_symbols = add_p_val_symbols(p_vals_df, "p_val")
    sig_symbols = sig_symbols["p_val_symbol"]
    
    # bars - Pearson_r values
    # color of bars - significance of Pr(>|t|)_symbol
    # wrapped_by_col_labels = [wrap_label(by_col_label, line_width=10) for by_col_label in by_cols_labels]
    ax.bar(
        by_cols_labels, r_vals,
        color=[SIGNIFICANCE_COLORS.get(sym, SIGNIFICANCE_COLORS["ns"]) for sym in sig_symbols],
        width=0.3,
    )
    # add r val on top of bars
    for i, r_val in enumerate(r_vals):
        ax.text(i, r_val, f"{r_val:.2f}", ha='center', va='bottom', fontsize=10)
    
    # labels
    # title for ax with pred_col_label
    title = pred_col_label.split("(")[0].strip()
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    ax.set_ylabel(r"$Pearson$ $r$", fontsize=12)
    # change fontsize of x labels
    ax.tick_params(axis='x', labelsize=10)
    
    # y lim
    # Force y=0 into the axis range
    overall_min = min(r_vals)
    overall_max = max(r_vals)
    overall_min = min(overall_min, 0)
    overall_max = max(overall_max, 0)
    padding = 0.3 * (overall_max - overall_min)
    overall_max = overall_max + padding
    new_max = max(overall_max, 0.5)
    ax.set_ylim([overall_min, new_max])
    
    # x lim
    padding = 0.5
    ax.set_xlim([-padding, len(by_cols) - 1 + padding])
    
    if 'SM' in output_file_name:
        # add vertical dashed line after the bar of by_col=='multivar_all' 
        # add vertical dashed line after the bar of by_col=='mean_pythia70m_surprisal_adv' 
        ax.axvline(x=2.5, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=5.5, color='black', linestyle='--', linewidth=1)

def plot_r_bar_plot_multivar_text(
    L1_or_L2: str,
    output_file_name: str,
    reading_regime: str,
    ):
    grid_labels_columns = ['Reading Fluency Correlations', 'Reading Comprehension Correlations']
    reading_comp_cols = ['is_correct', 'QA_RT', 'words_per_sec_based_P_RT']
    RT_cols = ['nonzero_TF', 'SkipTotal', 'RegCountTotal']
    by_y_cols = ['multivar_diff', 'multivar_adv', 'multivar_all']
    if 'SM' in output_file_name:
        by_y_cols += ['wordFreq_frequency_adv', 'word_length_adv', 'mean_pythia70m_surprisal_adv',
                     'diff_wordFreq_frequency', 'diff_word_length', 'diff_mean_pythia70m_surprisal']
        width = MEANS_PLOT_WIDTH*3
        height = MEANS_PLOT_HIGHT*1.5
    else:
        width = MEANS_PLOT_WIDTH+0.7
        height = (MEANS_PLOT_HIGHT+0.1)
    by_y_cols_labels = [BY_COL_LABELS[by_col] for by_col in by_y_cols]
    base_path = src_path / f'Eye_metrics/{L1_or_L2}/mixed_effects'
    per = "text"
    per_type = "text"
    
    n_rows = max(len(RT_cols), len(reading_comp_cols))
    n_cols = len(grid_labels_columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols,  height * n_rows), sharex=False)

    # # Titles
    # for j, col_label in enumerate(grid_labels_columns):
    #     if n_cols > 1:
    #         axes[0, j].set_title(col_label, fontsize=11, fontweight='bold')

    results_on_RT = {}
    results_on_comp = {}
    for by_col in by_y_cols:
        for effect_on in ['RT', 'comprehension']:
            _, fit_df = _load_on_col_by_col_files(
                base_path=base_path, 
                effect_on=effect_on, 
                by_col=by_col, 
                per=per, 
                per_type=per_type, 
                by_col_type=_get_by_col_type(by_col), 
                reading_regime=reading_regime,
                fit_only=True
            )
            if effect_on == 'RT':
                results_on_RT[by_col] = fit_df
            else:
                results_on_comp[by_col] = fit_df
            
    for j, grid_col in enumerate(grid_labels_columns):
        if grid_col == 'Reading Fluency Correlations':
            pred_cols = RT_cols
            results_dict = results_on_RT
        elif grid_col == 'Reading Comprehension Correlations':
            pred_cols = reading_comp_cols
            results_dict = results_on_comp
        
        for i, pred_col in enumerate(pred_cols):
            ax = axes[i, j] if n_cols > 1 else axes[i]
            _single_R_bar_plot(
                ax, results_dict, 
                pred_col, PRED_COLS_SHORT_LABELS[pred_col], 
                by_y_cols, by_y_cols_labels, output_file_name)
    
    # # blank subplots
    # ax = axes[2, 1]
    # ax.axis('off')
    
    fig = add_significance_legend(fig, with_colors=True)
    plt.tight_layout(rect=[0, 0.03, 0.98, 1])
    
    # Save plot
    by_col_type = "multivar"
    output_file_name = f"per_{per}_R_bars_grid_{by_col_type}{SM_suffix}.pdf"
    saving_dir = base_path / f"{per_type}_effect_{by_col_type}"
    saving_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = saving_dir / output_file_name
    plt.savefig(output_pdf_path)
    # save also to overleaf
    overleaf_dir = OVERLEAF_PATH / f"{per_type}_effect_{by_col_type}"
    overleaf_dir.mkdir(parents=True, exist_ok=True)
    output_pdf_path = overleaf_dir / output_file_name
    plt.savefig(output_pdf_path)
    plt.close()
    logger.info(f"{output_file_name} saved to {output_pdf_path}")

    # call save_latex_figure
    _save_latex_figure(
        overleaf_dir=overleaf_dir,
        pdf_file_name=output_file_name,
        plot_type="per_text_R_bars_grid",
        reading_regimes=[reading_regime],
        reader_type=L1_or_L2
    )
    
if __name__ == "__main__":
    # src Path
    src_path = Path.cwd() / Path("src")
    
    dont_edit_latex = True
    
    # Example usage
    L1_or_L2 = "L1"
    
    # RT columns
    main_RT_cols = ['nonzero_TF', 'SkipTotal', 'RegCountTotal']
    minimal_RT_cols = ['nonzero_TF']
    # Supplemetary Material RT columns
    SM_RT_cols = [
        'nonzero_GD', 'FirstPassGD', 
        'nonzero_FF', 'FirstPassFF', 
        'HigherPassFixation','NF', 
        'SkipFirstPass', 
        'RegCountFirstPass']
    
    for RT_cols in [main_RT_cols, SM_RT_cols]:
        SM_suffix = "_SM" if RT_cols == SM_RT_cols else ""
        
        # ----------------------------------------
        # Grid: main effects bar plots
        # Fig 1
        
        # if SM_suffix != '_SM':
        #     for reading_mode in ["Gathering0", "Hunting0"]:
        #         plot_Ele_effects_bars_grid(
        #             L1_or_L2,
        #             RT_cols,
        #             output_file_name=f"Ele_effects_grid_{reading_mode}{SM_suffix}.pdf",
        #             reading_regime=reading_mode,
        #     )

        # -----------------------------------
        # Grid: Response to linguistic features
        # Fig 2
        
        # for normed_RT in [True, False]:
        #     if SM_suffix == '_SM':
        #         rt_response_cols = ['FirstPassFF', 'FirstPassGD', 'SkipTotal', 'RegCountTotal']
        #     else:
        #         rt_response_cols = ['nonzero_FF', 'nonzero_GD', 'nonzero_TF']
        #     # for surp_col in ['gpt2', 'pythia70m']:
        #     for surp_col in ['pythia70m']:
        #         for formula_version in ["6"]:
        #             try:
        #                     plot_linguistic_effect_on_RT(
        #                     L1_or_L2, 
        #                     output_file_name=f"Linguistic_Effects_on_RT_{surp_col}_formula_{formula_version}_norm_{normed_RT}{SM_suffix}.pdf", 
        #                     formula_version=formula_version, 
        #                     rt_cols=rt_response_cols, 
        #                     normed_RT=normed_RT,
        #                     surp_col=surp_col,
        #                     reading_modes=['Gathering0'],
        #                 )
        #             except FileNotFoundError as e:
        #                 logger.error(e)
        
        # -------------------------------------
        # Grid: effect per subject/text
        # Fig 3
        
        # if SM_suffix != '_SM':
        #     plot_Ele_effect_per_subject_text_grid(
        #         L1_or_L2=L1_or_L2,
        #         output_file_name=f"Ele_effects_grid_per_text_subject{SM_suffix}.pdf",
        #         reading_regime="Gathering0",
        #     )
        # else:
        #     plot_Ele_effect_per_subject_text_grid_SM(
        #         L1_or_L2=L1_or_L2,
        #         reading_regime="Gathering0",
        #         on="RT",
        #         rt_cols=['words_per_sec_based_P_RT', 'FirstPassGD', 'nonzero_GD'],
        #         output_file_name = "Ele_effects_grid_on_RT_per_text_subject_SM_1.pdf"
        #         )
        #     plot_Ele_effect_per_subject_text_grid_SM(
        #         L1_or_L2=L1_or_L2,
        #         reading_regime="Gathering0",
        #         on="RT",
        #         rt_cols=['HigherPassFixation', 'SkipTotal', 'RegCountTotal'],
        #         output_file_name = "Ele_effects_grid_on_RT_per_text_subject_SM_2.pdf"
        #         )
        
        # -------------------------------------
        # Grid: effect per subject by col
        # Fig 4
        
        # per = "subject"
        # if SM_suffix != '_SM':
        #     main_grid_cols = ['words_per_sec_based_P_RT_adv', 'comprehension_score_adv', 'n_total_reading']
        #     grid_cols = main_grid_cols
        # else:
        #     grid_cols = ['age', 'years_education', 'is_student']
        # for curr_col in main_RT_cols:
        #     plot_reg_lines_grid(
        #         L1_or_L2, 
        #         rt_col=curr_col,
        #         reading_regime="Gathering0",
        #         per=per,
        #         by_cols=grid_cols,
        #         SM_suffix=SM_suffix,
        #         )     
        
        # -------------------------------------
        # # Grid: effect per text by col
        # # Fig 5
        
        # plot_r_bar_plot_multivar_text(
        #     L1_or_L2=L1_or_L2,
        #     output_file_name=f"R_bar_plot_multivar_text{SM_suffix}.pdf",
        #     reading_regime="Gathering0",
        #     )
        
        
        

                

        
