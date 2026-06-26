# Documentation Gaps and Cluster-Specific Issues

## Documentation Gaps

### 1. **Missing Data Derivation Steps**
- `pcr_labels.csv` is required but not documented as needing to be derived
- No instructions on extracting PCR response from `clinical_and_imaging_info.xlsx`
- Unclear which fields are prerequisites vs. generated during pipeline

### 2. **Incomplete Path Documentation**
- Configs reference `/net/projects2/vanguard/` paths without noting they are DSI-cluster-specific
- No alternative path mapping for other clusters (randi, etc.)
- No documentation of which paths exist on which clusters or what to do if unavailable

### 3. **Missing Submodule Setup Instructions**
- `vanguard-blood-vessel-segmentation` is referenced in code but not registered as a git submodule
- No `.gitmodules` entry; users must manually acquire model weights (`breast_model.pth`, `dv_model.pth`)
- No clear instructions: "get models from DSI `/net` before moving to another cluster"

### 4. **Undocumented Pre-requisites**
- GPU partition names vary by cluster (no documentation of how to discover or override)
- Scratch directory availability and quota assumptions not documented
- Cluster resource constraints (e.g., 48 total GPUs, node fragmentation) not mentioned

### 5. **No Cross-Cluster Setup Guide**
- No README section on "Running on a Different Cluster"
- No checklist: mount points, model availability, partition names, path overrides
- No example configs for non-DSI environments

---

## Cluster-Specific Issues Discovered

### 1. **Filesystem Differences**
| Issue | DSI Cluster | Randi Cluster |
|-------|-----------|---------------|
| `/net` mount | ✅ Available | ❌ Not mounted |
| Centerlines path | `/net/projects2/vanguard/centerlines_tc4d/studies/` | Must use `/scratch/t-9sbose/` |
| Model weights | Assumed in repo or `/net` | Must be manually copied from DSI |

### 2. **GPU Resource Constraints (Randi)**
- Total GPUs: 48 (across 6 nodes, 8 per node)
- Full 38-GPU array job = 80% cluster utilization + long queue times
- **Solution:** Submit smaller batch jobs (≤8 GPUs per job = fits 1 node, faster turnaround)

### 3. **Model Weight Acquisition**
- `vanguard-blood-vessel-segmentation/trained_models/` doesn't exist on randi
- Models must be copied from DSI before switching clusters
- **Current blocker:** No automated mechanism; manual transfer required

### 4. **Configuration Management**
- Existing configs are hardcoded for DSI paths
- New clusters require either:
  - Environment variable overrides (`IMAGES_DIR`, `OUTPUT_DIR`, `PARTITION`)
  - New YAML config files with cluster-specific paths (e.g., `configs/randi_mamamia.yaml`)

### 5. **Partition Discovery**
- GPU partition name (`gpu`, `gpu-v100`, etc.) varies by cluster
- Must run `sinfo` on login node to discover available partitions
- No default documented for randi; had to determine empirically

---

## Recommended Documentation Additions

1. **New README Section:** "Running on a Different Cluster"
   - Checklist: mount points, model file paths, partition names
   - Example env vars and config overrides for common clusters

2. **Submodule Setup Guide**
   - Document how to obtain `vanguard-blood-vessel-segmentation` and models
   - Add `.gitmodules` entry or provide fallback instructions

3. **Data Preparation Notebook**
   - Document derivation of `pcr_labels.csv` from clinical Excel
   - List all prerequisite files and where they should be

4. **Cluster Resource Profile**
   - Document GPU counts, node layout, typical queue times per cluster
   - Recommend job sizing based on available resources

5. **Config Template**
   - Provide example YAML with all path variables clearly marked
   - Add comments explaining DSI vs. randi path differences
