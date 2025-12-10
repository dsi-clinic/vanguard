# Centerline Extraction

This directory contains scripts for extracting graph-based centerlines from vessel segmentation masks.

## Scripts

### `extract_centerlines.py`

The main centerline extraction script that implements a skeletonization-based pipeline:

1. **Binarizes** the input segmentation at a specified threshold
2. **Skeletonizes** the binary mask to extract a 3D skeleton
3. **Connects fragmented islands** using k-nearest neighbor search (optional)
4. **Builds a graph structure** from the skeleton (nodes = branch points, edges = vessel segments)
5. **Extracts centerlines** as polylines from the graph structure
6. **Outputs** centerlines as VTK PolyData files (`.vtp` or `.vtk`)

**Key features:**
- Graph-based extraction using 26-connectivity (based on Matlab Skel2Graph3D)
- Island connection to heal fragmented skeletons before graph building
- Off-screen PyVista visualizations for debugging (headless/remote-safe)
- Supports multiple input formats: `.nii.gz`, `.nrrd`, `.npy` (4D arrays with channel selection)

**Usage:**
```bash
python extract_centerlines.py <input_segmentation> <output_centerline> [options]
```

**Options:**
- `--binarize-threshold FLOAT`: Threshold for binarization (default: 0.5)
- `--max-connection-distance-mm FLOAT`: Max distance to connect skeleton islands in mm (default: 15.0)
- `--no-island-connection`: Disable island connection step (faster, but may produce fragmented centerlines)
- `--no-visualizations`: Disable MP4 visualization generation
- `--extract-label INT`: Label ID to extract from multi-label inputs (default: 1)
- `--npy-channel INT`: Channel index for 4D NumPy arrays (default: 1)

**Example:**
```bash
python extract_centerlines.py vessel_segmentation.npy output_centerlines.vtp --no-visualizations
```

**Dependencies:**
- VTK >= 9.2
- scikit-image (for skeletonization)
- scipy (for k-nearest neighbor search)
- PyVista (for visualizations, optional)
- nibabel, nrrd, einops (for file I/O)

---

### `run_centerline_extraction.py`

A convenience wrapper that combines centerline extraction and JSON conversion in a single command. Instead of running two separate commands, it:

1. Extracts centerlines from a segmentation file (using `extract_centerlines.py`)
2. Converts the centerlines to JSON format (using `centerline_to_json.py`)
3. Cleans up intermediate files automatically

**Usage:**
```bash
python run_centerline_extraction.py <input_segmentation> <output_json> [options]
```

**Options:**
- `--binarize-threshold FLOAT`: Threshold for binarization (default: 0.5)
- `--max-connection-distance-mm FLOAT`: Max distance to connect skeleton islands in mm (default: 15.0)
- `--no-island-connection`: Disable island connection step
- `--spacing X Y Z`: Voxel spacing in mm (default: 1.0 1.0 1.0)

**Example:**
```bash
python run_centerline_extraction.py vessel_segmentation.npy output_centerlines.json --spacing 1.0 1.0 1.0
```

**Note:** For batch processing multiple files, use `../batch_processing/batch_extract_centerlines.py` or the SLURM submit scripts in `../slurm_submit_scripts/`.

