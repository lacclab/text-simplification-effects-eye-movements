# Code for the paper: The Effect of Text Simplification on Reading Fluency and Reading Comprehension in L1 English Speakers

[![Ruff](https://github.com/lacclab/text-simplification-effects-eye-movements/actions/workflows/ruff.yml/badge.svg?branch=main)](https://github.com/lacclab/text-simplification-effects-eye-movements/actions/workflows/ruff.yml)

## Quick Start

Set up environment: run `conda env create -f env.yml`

## Download Data
Follow the instructions in https://github.com/lacclab/OneStop-Eye-Movements.

### Main Files

Plots - `src/Eye_metrics/plot_mixed_effects_results.py` (uncomment the relevant Fig lines at the __main__ section)
Models Fit:
1. Fig 1 Main effects: `src/Eye_metrics/fit_mixed_effects_julia.py`
2. Fig 2 Response to RT: `src/Eye_metrics/fit_response_to_linguistic.py`
3. Fig 3 Effects per subject / item: `src/Eye_metrics/fit_mixed_effects_julia_per_subject_text.py`
4. Fig 4 Effect per subject by different characteristics: `src/Eye_metrics/fit_julia_effects_by_col.py`
5. Fig 5 Effect per text by textual properties: `src/Eye_metrics/fit_julia_effects_by_col.py`
6. Julia functions: `src/Julia_models.py`

## Citation

If you use this repository, please consider citing the following work:

```bibtex
@article{gruteke-klein-etal-2025-simplification-effect,
  title={The effect of text simplification on reading fluency and reading comprehension in l1 english speakers},
  author={Gruteke Klein, Keren and
      Shubi, Omer and
      Frenkel, Shachar and
      Berzak, Yevgeni},
  url="https://osf.io/dhk8c"
}
```
