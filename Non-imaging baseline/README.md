# Non-Imaging Baseline: Feature Importance

This branch extends the baseline non-imaging pCR prediction pipeline by adding a module to analyze feature importance in the trained logistic regression model.

It complements `baseline_pcr_simple.py` by producing interpretable summaries of which features most strongly contribute to predicting pCR using demographic and metadata features only.

---

## Goal
Identify and rank the most predictive non-imaging features using:

- **Logistic regression coefficients**
- **Permutation importance** (validation-set performance impact)

---

## Script

### `feature_importance.py`

**Purpose:**  
Compute and visualize feature importance for a trained baseline model.

**Inputs:**
| Argument | Description |
|-----------|--------------|
| `--model` | Path to trained model file (`model.pkl`) |
| `--json-dir` | Directory containing patient JSON metadata |
| `--split-csv` | CSV defining data splits (train/val) |
| `--output` | Output directory for generated files |
| `--top-k` | *(Optional)* Number of features shown in the plot (default: 20) |
| `--n-repeats` | *(Optional)* Number of permutation repeats (default: 20) |

---

## Usage Example

```bash
python feature_importance.py \
  --model outdir/model.pkl \
  --json-dir /net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files \
  --split-csv splits_v1.csv \
  --output feature_outdir \
  --top-k 20
