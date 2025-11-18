import sys
import os

PROJECT_ROOT = "/home/jmcarias/vanguard"
sys.path.append(os.path.join(PROJECT_ROOT, "notebooks"))
sys.path.append(os.path.join(PROJECT_ROOT, "notebooks/skeleton3d_utils"))
from skeleton3d_utils.pipeline import process_vessel_image

input_file = sys.argv[1]
output_folder = sys.argv[2]
threshold = float(sys.argv[3])

process_vessel_image(input_file, threshold, output_folder)