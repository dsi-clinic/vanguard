#!/usr/bin/env python3
"""
Batch segmentation script for processing all .nii.gz files in /images directory
and extracting STEP-2 breast mask segmentations (.npy files).

This script:
1. Finds all .nii.gz files in the images directory
2. Processes each file through the breast segmentation pipeline (STEP-1 → STEP-2)
3. Collects all STEP-2 .npy files (breast masks) for further processing
4. Provides options for parallel processing and progress tracking
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List, Tuple, Optional

# Add the segmentation project to Python path
sys.path.append('/home/ruochun/vanguard/3D-Breast-FGT-and-Blood-Vessel-Segmentation')

import numpy as np
import SimpleITK as sitk
from preprocessing import normalize_image, zscore_image


def find_nii_files(images_dir: str) -> List[Tuple[str, str]]:
    """
    Find all .nii.gz files in the images directory.
    
    Args:
        images_dir: Path to the images directory
        
    Returns:
        List of tuples (patient_id, file_path)
    """
    nii_files = []
    
    # Get all patient directories
    patient_dirs = [d for d in os.listdir(images_dir) 
                   if os.path.isdir(os.path.join(images_dir, d))]
    
    for patient_id in patient_dirs:
        patient_path = os.path.join(images_dir, patient_id)
        
        # Find all .nii.gz files in this patient directory
        pattern = os.path.join(patient_path, "*.nii.gz")
        files = glob.glob(pattern)
        
        for file_path in files:
            nii_files.append((patient_id, file_path))
    
    return nii_files


def preprocess_image(input_path: str, output_path: str) -> bool:
    """
    Preprocess a single .nii.gz file (STEP-1).
    
    Args:
        input_path: Path to input .nii.gz file
        output_path: Path to save preprocessed .npy file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the image
        original_array = sitk.GetArrayFromImage(sitk.ReadImage(input_path))
        
        # Preprocess: rotate axes and normalize
        preprocessed_array = zscore_image(normalize_image(
            np.swapaxes(np.swapaxes(original_array, 0, 2), 0, 1)[::-1]
        ))
        
        # Save as .npy
        np.save(output_path, preprocessed_array)
        return True
        
    except Exception as e:
        print(f"Error preprocessing {input_path}: {e}")
        return False


def run_breast_segmentation(step1_dir: str, step2_dir: str, model_path: str) -> bool:
    """
    Run breast segmentation on preprocessed data (STEP-2).
    
    Args:
        step1_dir: Directory containing STEP-1 preprocessed files
        step2_dir: Directory to save STEP-2 breast masks
        model_path: Path to the breast segmentation model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Change to the segmentation project directory
        original_cwd = os.getcwd()
        os.chdir('/home/ruochun/vanguard/3D-Breast-FGT-and-Blood-Vessel-Segmentation')
        
        # Run the prediction script
        cmd = [
            'python', 'predict.py',
            '--target-tissue', 'breast',
            '--image', step1_dir,
            '--save-masks-dir', step2_dir,
            '--model-save-path', model_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Restore original working directory
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            return True
        else:
            print(f"Segmentation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error running segmentation: {e}")
        return False


def process_single_file(args: Tuple[str, str, str, str, str]) -> Tuple[str, bool, str]:
    """
    Process a single .nii.gz file through the complete pipeline.
    
    Args:
        args: Tuple containing (patient_id, file_path, temp_dir, output_dir, model_path)
        
    Returns:
        Tuple of (patient_id, success, output_path)
    """
    patient_id, file_path, temp_dir, output_dir, model_path = args
    
    try:
        # Create temporary directories for this file
        step1_dir = os.path.join(temp_dir, f"{patient_id}_step1")
        step2_dir = os.path.join(temp_dir, f"{patient_id}_step2")
        
        os.makedirs(step1_dir, exist_ok=True)
        os.makedirs(step2_dir, exist_ok=True)
        
        # Extract filename without extension
        filename = os.path.basename(file_path)
        base_name = filename.replace('.nii.gz', '')
        
        # STEP-1: Preprocess
        step1_file = os.path.join(step1_dir, f"{base_name}.npy")
        if not preprocess_image(file_path, step1_file):
            return patient_id, False, ""
        
        # STEP-2: Breast segmentation
        if not run_breast_segmentation(step1_dir, step2_dir, model_path):
            return patient_id, False, ""
        
        # Move the result to output directory
        step2_file = os.path.join(step2_dir, f"{base_name}.npy")
        output_file = os.path.join(output_dir, f"{patient_id}_{base_name}_breast_mask.npy")
        
        if os.path.exists(step2_file):
            shutil.move(step2_file, output_file)
            return patient_id, True, output_file
        else:
            return patient_id, False, ""
            
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return patient_id, False, ""


def collect_all_step2_files(output_dir: str) -> List[str]:
    """
    Collect all STEP-2 .npy files from the output directory.
    
    Args:
        output_dir: Directory containing the processed files
        
    Returns:
        List of paths to all .npy files
    """
    npy_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files


def main():
    parser = argparse.ArgumentParser(
        description='Batch process all .nii.gz files and extract breast mask segmentations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--images-dir', 
        default='/net/projects2/vanguard/MAMA-MIA-syn60868042/images',
        help='Directory containing patient subdirectories with .nii.gz files'
    )
    
    parser.add_argument(
        '--output-dir',
        default='/home/ruochun/vanguard/breast_masks',
        help='Directory to save all STEP-2 .npy files'
    )
    
    parser.add_argument(
        '--temp-dir',
        default='/tmp/batch_segmentation',
        help='Temporary directory for intermediate processing'
    )
    
    parser.add_argument(
        '--model-path',
        default='/home/ruochun/vanguard/3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/breast_model.pth',
        help='Path to the breast segmentation model'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers'
    )
    
    parser.add_argument(
        '--patient-limit',
        type=int,
        default=None,
        help='Limit processing to first N patients (for testing)'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up temporary files after processing'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    
    print(f"Finding .nii.gz files in {args.images_dir}...")
    nii_files = find_nii_files(args.images_dir)
    
    if args.patient_limit:
        nii_files = nii_files[:args.patient_limit]
        print(f"Limited to first {args.patient_limit} files for testing")
    
    print(f"Found {len(nii_files)} .nii.gz files to process")
    
    # Prepare arguments for parallel processing
    process_args = [
        (patient_id, file_path, args.temp_dir, args.output_dir, args.model_path)
        for patient_id, file_path in nii_files
    ]
    
    # Process files in parallel
    successful_files = []
    failed_files = []
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_patient = {
            executor.submit(process_single_file, arg): arg[0] 
            for arg in process_args
        }
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_patient), 1):
            patient_id, success, output_path = future.result()
            
            if success:
                successful_files.append(output_path)
                print(f"[{i}/{len(nii_files)}] ✓ {patient_id}: {output_path}")
            else:
                failed_files.append(patient_id)
                print(f"[{i}/{len(nii_files)}] ✗ {patient_id}: Failed")
    
    end_time = time.time()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {len(nii_files)}")
    print(f"Successful: {len(successful_files)}")
    print(f"Failed: {len(failed_files)}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Average time per file: {(end_time - start_time) / len(nii_files):.2f} seconds")
    
    if failed_files:
        print(f"\nFailed files:")
        for patient_id in failed_files:
            print(f"  - {patient_id}")
    
    # Collect all STEP-2 files
    all_npy_files = collect_all_step2_files(args.output_dir)
    print(f"\nAll STEP-2 .npy files saved to: {args.output_dir}")
    print(f"Total .npy files: {len(all_npy_files)}")
    
    # Cleanup
    if args.cleanup:
        print(f"\nCleaning up temporary directory: {args.temp_dir}")
        shutil.rmtree(args.temp_dir, ignore_errors=True)
    
    return successful_files, failed_files


if __name__ == '__main__':
    main()

