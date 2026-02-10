#!/usr/bin/env python3
"""Quick test to verify evaluation framework integration works."""

import numpy as np
import pandas as pd
from pathlib import Path

from eval_adapter import (
    create_evaluator_from_radiomics_data,
    create_train_test_results,
    save_evaluation_results,
)

def main():
    """Test evaluation framework integration with dummy data."""
    print("Testing evaluation framework integration...")

    # Create dummy data matching radiomics format
    n_samples = 100
    n_features = 50

    # Simulate training data
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    patient_ids_train = [f"PATIENT_{i:03d}" for i in range(n_samples)]

    # Simulate test data
    n_test = 30
    X_test = np.random.randn(n_test, n_features)
    y_test = np.random.randint(0, 2, n_test)
    patient_ids_test = [f"TEST_{i:03d}" for i in range(n_test)]

    # Simulate predictions (random for this test)
    y_pred = np.random.randint(0, 2, n_test)
    y_prob = np.random.rand(n_test)

    print(f"  Training samples: {n_samples}")
    print(f"  Test samples: {n_test}")
    print(f"  Features: {n_features}")

    # Step 1: Create evaluator
    print("\nStep 1: Creating evaluator...")
    evaluator = create_evaluator_from_radiomics_data(
        X_train=X_train,
        y_train=y_train,
        patient_ids_train=np.array(patient_ids_train),
        model_name="test_radiomics_rf",
    )
    print("  ✓ Evaluator created")

    # Step 2: Create results
    print("\nStep 2: Creating results object...")
    results = create_train_test_results(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        patient_ids=np.array(patient_ids_test),
        model_name="test_radiomics_rf",
    )
    print(f"  ✓ Results created")
    print(f"    AUC: {results.metrics.get('auc', 'N/A'):.3f}")

    # Step 3: Save results
    print("\nStep 3: Saving results...")
    output_dir = Path("radiomics_baseline/test_integration_output")
    save_evaluation_results(evaluator, results, output_dir)
    print("  ✓ Results saved")

    # Verify output files
    print("\nGenerated files:")
    model_dir = output_dir / "test_radiomics_rf" / "test"
    for file in sorted(model_dir.glob("*")):
        print(f"  - {file.relative_to(output_dir)}")

    print("\n✅ Integration test successful!")
    print(f"\nCheck outputs in: {output_dir}")


if __name__ == "__main__":
    main()
