## Data Preprocessing and Assumptions

This baseline model uses a minimal metadata feature set extracted from patient JSON files to predict (pCR).  
Each patient corresponds to one JSON file and one row in `splits_v1.csv`.

### **Input Features**

| Feature | Type | Description |
|----------|------|--------------|
| `age` | numeric | Patient age at time of imaging (from `clinical_data.age`). |
| `tumor_subtype` | categorical | Tumor molecular subtype (from `primary_lesion.tumor_subtype`). |
| `bbox_volume` | numeric | Approximate lesion volume computed as:<br>`(x_max - x_min) * (y_max - y_min) * (z_max - z_min)` from `primary_lesion.breast_coordinates`. |
| `pCR` | binary (0/1) | Pathologic complete response label (from `primary_lesion.pcr`). |

---

### **Missing Data Handling**

| Field | Strategy | Rationale |
|--------|-----------|-----------|
| **pCR** | Keep as `None` if missing. Excluded from model training and AUC computation, but retained for inference. | Some patients lack confirmed outcome labels. |
| **age** | Imputed using **median** age. | Median is robust against skewed age distribution. |
| **tumor_subtype** | Missing or blank values are filled with `"unknown"`. | Ensures consistent categorical encoding in one-hot features. |
| **bbox_volume** | Imputed using **median** volume. | Missing coordinates may occur; median preserves scale. |
| **split** | Defined in `splits_v1.csv` as `train`, `val`, or `test`. | Split file ensures consistent partitioning. |
| **other fields** | Ignored. | Not part of the minimal baseline feature set. |

---

### **Data Integrity Rules**

- Each JSON file must include a valid `patient_id` matching one row in `splits_v1.csv`.  
- Patients not present in the split file are **skipped**.  
- Patients with non-numeric or malformed values for `age` or `bbox_volume` are converted to `NaN` and imputed.  
- All numeric features are standardized (`StandardScaler`), and categorical features are one-hot encoded (`OneHotEncoder(handle_unknown="ignore")`).

---

### **Label Usage**

- Model training and evaluation (ROC/AUC) use **only labeled patients** (`pCR` ∈ {0,1}).  
- All patients (including unlabeled) receive predictions in `predictions.csv`.  
- The final metrics file (`metrics.json`) records both labeled and unlabeled counts.

