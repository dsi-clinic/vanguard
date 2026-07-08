# Validation: old vs. fast segmentation pipeline

- Images dir: `/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images`
- Sample files: indices 0-6 (7 files)
- Old output dir: `/ess/scratch/scratch1/t-9sbose/validate_speed_and_accuracy/old`
- Fast output dir: `/ess/scratch/scratch1/t-9sbose/validate_speed_and_accuracy/fast`
- Threshold for Dice: 0.5

## Per-file results

| case                               |    old_s |   fast_s | shape_ok |    dice |  max_abs |  mean_abs | note |
|---|---|---|---|---|---|---|---|
| DUKE_001/DUKE_001_0000             |   287.07 |    17.35 |     True |  1.0000 |   0.0039 |  0.000001 |  |
| DUKE_001/DUKE_001_0001             |   267.55 |    16.89 |     True |  0.9998 |   0.0095 |  0.000001 |  |
| DUKE_001/DUKE_001_0002             |   267.81 |    16.92 |     True |  0.9998 |   0.0120 |  0.000001 |  |
| DUKE_001/DUKE_001_0003             |   269.26 |    17.04 |     True |  0.9999 |   0.0059 |  0.000001 |  |
| DUKE_001/DUKE_001_0004             |   268.46 |    17.12 |     True |  0.9998 |   0.0056 |  0.000001 |  |
| DUKE_002/DUKE_002_0000             |   310.14 |    18.50 |     True |  0.9999 |   0.0039 |  0.000000 |  |
| DUKE_002/DUKE_002_0001             |   309.02 |    18.73 |     True |  0.9998 |   0.0032 |  0.000000 |  |

## Summary

- Files compared: 7
- Old runtime (s): 282.76 +/- 19.58
- Fast runtime (s): 17.51 +/- 0.77
- Mean speedup: 16.15x
- Shape matches: 7/7
- Dice (thr=0.5): min=0.9998 mean=0.9999 max=1.0000
- Max |prob diff|: max=0.0120 mean=0.0063
- Mean |prob diff|: mean=0.000001

## Exact commands used

```
python /ess/home/home1/t-9sbose/vanguard/faster-segmentation-test/validate_speed_and_accuracy.py --images-dir /ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images --file-start 0 --file-end 6 --output-root /ess/scratch/scratch1/t-9sbose/validate_speed_and_accuracy
```

Old output dir: `/ess/scratch/scratch1/t-9sbose/validate_speed_and_accuracy/old`

Fast output dir: `/ess/scratch/scratch1/t-9sbose/validate_speed_and_accuracy/fast`
