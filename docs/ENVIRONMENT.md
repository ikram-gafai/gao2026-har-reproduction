# Environment

## Software

The direct dependencies are pinned in `pyproject.toml`. The full resolved set is in `requirements-lock.txt`. The evidence-note build uses a separate environment, `paper/requirements-paper.txt`.

| Component | Version |
|---|---|
| Python | 3.11.3 (CPython, MSC v.1934, 64-bit) |
| xgboost | 3.2.0 (the newest version on PyPI for this platform on 2026-11-25) |
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 (installed as a scikit-learn dependency) |
| pytest | 9.2.0 |

The remaining resolved packages are cloudpickle 3.1.3, colorama 0.4.6, iniconfig 2.3.1, joblib 1.6.0, narwhals 2.29.0, packaging 26.4, pluggy 1.7.0, Pygments 2.21.0, python-dateutil 4.0.0, threadpoolctl 3.8.0 and tzdata 2026.6.

**XGBoost version decision:**
- xgboost 3.2.0 is the **primary** implementation.
- The paper used **R xgboost 1.7.7.1** under R 4.3.3, so this is a cross-language approximation.
- Python xgboost 1.7.6 is reserved for an *optional, separately labelled* version-parity sensitivity run, which would need its own environment. It is never a replacement chosen on test results.

**XGBoost behaviour checked on synthetic data** (`tests/test_xgb_config_synthetic.py`):
- `tree_method="exact"` resolves to the `grow_colmaker,prune` updaters.
- `base_score=0.5` is saved as a uniform 6-class intercept of 0.5. Without it, this version estimates the intercept.
- Training uses 1 thread.
- The same seed gives bit-identical probabilities; a different seed gives different ones.

## Hardware (development machine)

- Windows 11 Home 10.0.26200, x64
- AMD Ryzen 7 8845HS, 16 logical CPUs
- 15.2 GB RAM
- CPU only: models are trained with `nthread=1` and no GPU is used.

The installed xgboost wheel is built with CUDA support, but `device` stays `cpu`.

Observed runtimes in the pre-registered run of 25 Nov 2026 (single training thread, `fit_seconds` in `results/runs/*/run_info.json`):

- full-561 fits: 313–386 s;
- compact-100 fits: 32–52 s;
- sum of the 20 fits: 2,511 s.
