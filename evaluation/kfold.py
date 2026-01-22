"""K-fold cross-validation split generation and aggregation."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold


def create_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray | None = None,
    n_splits: int = 5,
    stratify: bool = True,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[dict[str, np.ndarray | int]]:
    """
    Create k-fold splits and return them as a list of dictionaries.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    patient_ids : np.ndarray, optional
        Patient IDs for tracking
    n_splits : int, default=5
        Number of folds
    stratify : bool, default=True
        Whether to use stratified k-fold (maintains class distribution)
    shuffle : bool, default=True
        Whether to shuffle data before splitting
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    list[dict[str, np.ndarray | int]]
        List of dictionaries, one per fold, each containing:
        - "fold_idx": fold number (0-indexed)
        - "train_indices": indices for training
        - "val_indices": indices for validation
        - "train_patient_ids": patient IDs for training (if available)
        - "val_patient_ids": patient IDs for validation (if available)
    """
    # Choose splitter
    if stratify:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
    else:
        splitter = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )
    
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        split_dict = {
            "fold_idx": fold_idx,
            "train_indices": train_idx,
            "val_indices": val_idx,
        }
        
        # Add patient IDs if available
        if patient_ids is not None:
            split_dict["train_patient_ids"] = patient_ids[train_idx]
            split_dict["val_patient_ids"] = patient_ids[val_idx]
        else:
            split_dict["train_patient_ids"] = None
            split_dict["val_patient_ids"] = None
        
        splits.append(split_dict)
    
    return splits
