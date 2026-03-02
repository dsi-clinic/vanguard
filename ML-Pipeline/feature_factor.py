import pandas as pd

def get_clinical_features(config):
    """Loads and cleans the high-value features from Excel."""
    path = config['data_paths']['clinical_excel']
    df = pd.read_excel(path)
    
    # Columns from EDA
    cols = [
        'patient_id', 'age', 'menopause', 'tumor_subtype', 
        'hr', 'er', 'pr', 'her2', 'nottingham_grade', 'bmi_group'
    ]
    df = df[cols].copy()
    df = pd.get_dummies(df, columns=['tumor_subtype', 'menopause', 'bmi_group'])
    
    return df
