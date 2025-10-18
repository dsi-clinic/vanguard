###############################################
Circle of Willis Centerline Data Release
###############################################


1. Overview

This public dataset is based on the TopCoW 2024 training data, which consists of 250 (=125 pairs of) CTA and MRA images displaying brain vasculature, including the Circle of Willis (CoW). For access to the raw imaging data and the multiclass mask, we refer to the TopCoW Zenodo dataset repository: https://zenodo.org/records/15692630. 

2. Data Usage License

Our data is released under the CC BY-NC (Attribution-NonCommercial) license. 
By downloading the data you agree with the license terms.

3. Contents of Data

For each patient and imaging modality (n=250), the dataset includes
- CoW graph: Labeled CoW centerline graph (.vtp)
- CoW mesh: Labeled CoW surface mesh (.vtp)
- CoW variant: Variant description encoding presence/absence of segments and fetal-type PCA (.json)
- CoW nodes: Node descriptions for start, end, bifurcation and segment boundary points (.json)
- CoW features: Morphometric feature descriptions for segments and bifurcations (.json)

The files follow the same naming pattern as introduced by TopCoW: `topcow_{modality}_{pat_id}.<suffix>`
  * `modality`: `mr` for MRA, `ct` for CTA
  * `pat_id`: patient ID, `001`, `002`, ...

The data folder has the following sub-folders:

* `cow_graphs`: Labeled CoW centerline graphs as vtkPolyData objects (.vtp). The centerlines were extracted from the TopCoW multiclass masks. 
  * Point attributes: point ID, (x,y,z)-coordinates, degree 
  * Edge attributes: cell ID, class label, Voreen average radius, MIS radius, CE radius
  * Labels of the different CoW vessel segments as introduced by TopCoW:
    * 1: BA, 2: R-PCA, 3: L-PCA, 4: R-ICA, 5: R-MCA, 6: L-ICA, 7: L-MCA, 8: R-Pcom, 9: L-Pcom, 10: Acom, 11: R-ACA, 12: L-ACA, 15: 3rd-A2
* `cow_meshes`: CoW surface meshes as vtkPolyData objects (.vtp). The surface meshes are based on the TopCoW multiclass masks.
* `cow_nodes`: Node descriptions for start, end, bifurcation and segment boundary points of the centerline graph.
  * json files indicating the ID, degree, label and coordinates of the nodes
    * All the nodes in order of their respective label:
      * BA (label 1): BA start, BA bifurcation, R-PCA boundary, L-PCA boundary
      * {R,L}-PCA (2,3): BA boundary, Pcom bifurcation, Pcom boundary, PCA end
      * {R,L}-ICA (4,6): ICA start, Pcom bifurcation, Pcom boundary, ICA bifurcation, ACA boundary, MCA boundary
      * {R,L}-MCA (5,7): ICA boundary, MCA end
      * {R,L}-Pcom (8,9): ICA boundary, PCA boundary
      * Acom (10): R-ACA boundary, L-ACA boundary, 3rd-A2 bifurcation, 3rd-A2 boundary
      * {R,L}-ACA (11,12): ICA boundary, Acom bifurcation, Acom boundary, ACA end
      * 3rd-A2 (15): Acom boundary, 3rd-A2 end
* `cow_variants`: Variant information of the CoW graphs.
  * json files indicating the presence of segments, fetal-type PCA and arterial fenestrations (0: absent, 1: present).
    * 4 `anterior` segments:
      * L-A1     
      * Acom
      * 3rd-A2
      * R-A1
    * 4 `posterior` segments:
      * L-Pcom
      * L-P1
      * R-P1
      * R-Pcom
    * 2 `fetal` PCA segments:
      * L-PCA
      * R-PCA 
    * 5 `fenestration` segments:
      * L-A1
      * Acom
      * R-A1
      * L-P1
      * R-P1
* `cow_features`: Geometric features of the CoW centerline graphs.
  * json files containing segment features for all 13 CoW segments (including subsegments A1/A2, P1/P2, C6/C7 for ACA, PCA and ICA respectively) and bifurcation features for BA, ICA, Acom and Pcom bifurcations.
    * 5 sets of segment features:
      * radius (mean, sd, median, min, q1, q3, max)
      * length 
      * tortuosity
      * volume 
      * curvature (mean, sd, median, min, q1, q3, max)
    * 3 sets of bifurcation features:
      * angles 
      * radius ratios (only for BA & ICA)
      * bifurcation exponent (only for BA & ICA)

4. Citation

If you use the CoW centerline data in your work, please cite our paper:

Musio et al., “Circle of Willis Centerline Graphs: A Dataset and Baseline Algorithm” (2025)

5. Contact

Fabio Musio (fabio.musio@zhaw.ch) and Prof. Dr. Sven Hirsch (sven.hirsch@zhaw.ch) from Zurich University of Applied Sciences (ZHAW)


Updated October-15-2025
