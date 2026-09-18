"""
Dataset Split Generator for PS3 - Structural Health Monitoring (SHM)

Creates reproducible, stratified train/validation/test splits based on damage quartiles:
- Train: 48 files (75%)
- Val: 16 files (25%) - exactly 4 files per damage quartile
- Test: 16 unlabelled files (held-out competition set)
- Includes 4-fold and 5-fold CV indices for K-fold ensembling.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42

def create_data_splits(
    train_labels_csv: str = "problem_statements/PS3/02_Datasets/SHM/Train_Labels.csv",
    train_dir: str = "problem_statements/PS3/02_Datasets/SHM/Train",
    test_dir: str = "problem_statements/PS3/02_Datasets/SHM/Test",
    output_csv: str = "ps3/SHM/data_splits.csv"
):
    train_df = pd.read_csv(train_labels_csv)
    train_df["file_path"] = train_df["filename"].apply(lambda f: os.path.join(train_dir, f).replace("\\", "/"))
    train_df["damage_quartile"] = pd.qcut(train_df["damage"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    # 4-Fold Stratified Split
    skf_4 = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    train_df["cv_fold_4"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf_4.split(train_df, train_df["damage_quartile"])):
        train_df.loc[val_idx, "cv_fold_4"] = fold_idx

    # 5-Fold Stratified Split
    skf_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_df["cv_fold_5"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf_5.split(train_df, train_df["damage_quartile"])):
        train_df.loc[val_idx, "cv_fold_5"] = fold_idx

    # Fixed Holdout: Fold 0 is validation (16 files), Folds 1-3 are train (48 files)
    train_df["split"] = np.where(train_df["cv_fold_4"] == 0, "val", "train")

    # Test set metadata
    test_files = sorted(os.listdir(test_dir))
    test_df = pd.DataFrame({
        "filename": test_files,
        "split": "test",
        "damage": np.nan,
        "damage_quartile": np.nan,
        "cv_fold_4": -1,
        "cv_fold_5": -1,
        "file_path": [os.path.join(test_dir, f).replace("\\", "/") for f in test_files]
    })

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"Data splits successfully saved to: {output_csv}")
    return combined_df

if __name__ == "__main__":
    create_data_splits()
