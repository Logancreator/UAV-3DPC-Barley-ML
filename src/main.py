import os
import time
import pandas as pd
import numpy as np
from utils import register, cal_indices
import warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)

def create_paths(base_path, date):
    """Creates and returns a dictionary of file paths."""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")
    
    base_folder = os.path.join(base_path, "qgis")
    ortho_folder = os.path.join(base_path, "extracted_ortho", date)
    return {
        "rgb_3dpc_shp": os.path.join(base_folder, "3dpc", f"{date}-rgb3dpc2.shp"),
        "rgb_3dpc": os.path.join(base_path, "rgb_point_clouds", f"{date}_rgb_3dpc.txt"),
        "rgb_3dpc_output": os.path.join(base_path, "extracted_rgb_point_clouds", date, "register"),

        "msi_3dpc_shp": os.path.join(base_folder, "3dpc", f"{date}-msi3dpc2.shp"),
        "msi_3dpc": os.path.join(base_path, "msi_point_clouds", f"{date}_msi_3dpc.txt"),
        "msi_3dpc_output": os.path.join(base_path, "extracted_msi_point_clouds", date, "register"),

        "rgb_shp": os.path.join(base_folder, "3dpc", f"{date}-rgb3dpc2.shp"),
        "rgb_ortho": os.path.join(base_path, "rgb_ortho", f"{date}_rgb_ortho.tif"),
        "rgb_ortho_output": os.path.join(ortho_folder, "rgb"),

        "msi_shp": os.path.join(base_folder, "3dpc", f"{date}-msi3dpc2.shp"),
        "msi_ortho": os.path.join(base_path, "msi_ortho", f"{date}_msi_ortho.tif"),
        "msi_ortho_output": os.path.join(ortho_folder, "msi")
    }

def process_3dpc_register(paths, date, base_path):
    """Extracts point clouds"""
    print("Processing 3DPC data...")
    register.extract_and_save_point_clouds(paths["rgb_3dpc_shp"], paths["rgb_3dpc"], paths["rgb_3dpc_output"], point_cloud_type="auto")
    register.extract_and_save_point_clouds(paths["msi_3dpc_shp"], paths["msi_3dpc"], paths["msi_3dpc_output"], point_cloud_type="auto")


def process_3dpc_calculate(paths, date, base_path):
    """Calculates 3DPC indices."""
    print("Processing 3DPC data...")
    cal_indices.calculate_rgb_3d_indices_folder(paths["rgb_3dpc_output"][:-9], date, base_path=base_path, sub_folder="register")
    cal_indices.calculate_msi_3d_indices_folder(paths["msi_3dpc_output"][:-9], date, base_path=base_path, sub_folder="register")


def process_ortho(paths, date, ortho_type, base_path):
    """Calculates and saves ortho indices."""
    print(f"Processing {ortho_type.upper()} ortho data...")
    
    zonal_stats_func = cal_indices.zonal_stats_rgb_VIs if ortho_type == "rgb" else cal_indices.zonal_stats_msi_VIs
    shp_path = paths[f"{ortho_type}_shp"]
    ortho_path = paths[f"{ortho_type}_ortho"]

    if ortho_type == "rgb":
        dummy_indices = cal_indices.get_rgb_VIs(np.array([0]), np.array([0]), np.array([0]))
        index_names = list(dummy_indices.keys())
        # Pass calculate_glcm=False for RGB vegetation indices
        zonal_stats_dict = {name: zonal_stats_func(ortho_path, shp_path, name, calculate_glcm=False) for name in index_names}
    else:  # msi
        dummy_indices = cal_indices.get_msi_VIs(np.array([0]), np.array([0]), np.array([0]), 
                                   np.array([0]), np.array([0]), np.array([0]))
        index_names = list(dummy_indices.keys())
        zonal_stats_dict = {name: zonal_stats_func(ortho_path, shp_path, name) for name in index_names}

    zonal_stats_df = cal_indices.combine_stats_to_dataframe(zonal_stats_dict, add_object_id=True)

    file_path = os.path.join(base_path, f"{ortho_type}_zonal_stats_VIs_{date}.csv")
    zonal_stats_df.to_csv(file_path, index=False)
    print(f"Ortho indices saved to: {file_path}")

def process_ortho_glcm(paths, date, ortho_type, base_path, calculate_glcm=True):
    """Calculates and saves ortho GLCM features using RGB-to-grayscale conversion."""
    if calculate_glcm:
        print(f"Processing {ortho_type.upper()} ortho GLCM data...")
        if ortho_type != "rgb":
            raise ValueError("GLCM calculation is only supported for RGB ortho data.")
        
        # This function now correctly loops through geometries and calculates robust GLCM stats
        stats_list = cal_indices.zonal_stats_rgb_VIs(
            tiff=paths[f"{ortho_type}_ortho"], 
            shpfile=paths[f"{ortho_type}_shp"], 
            index_name='grayscale', # This is just a placeholder name
            calculate_glcm=True
        )
        
        if not stats_list:
            print("No GLCM stats were generated.")
            return

        zonal_stats_df = pd.DataFrame(stats_list)
        
        # Add the Object_ID column at the beginning
        zonal_stats_df.insert(0, 'Object_ID', range(1, len(zonal_stats_df) + 1))
        
        print(zonal_stats_df)
        file_path = os.path.join(base_path, f"{ortho_type}_glcm_stats_{date}.csv")
        zonal_stats_df.to_csv(file_path, index=False)
        print(f"GLCM stats saved to: {file_path}")
        
if __name__ == '__main__':
    # Main script
    laptop_path= r"E:\git\phenomics\barley\project\uav\data"
    sonic_path = r"/scratch/24212502/data"

    if os.path.exists(laptop_path):
        base_path = laptop_path
    elif os.path.exists(sonic_path):
        base_path = sonic_path
    else:
        raise FileNotFoundError(f"Neither {laptop_path} nor {sonic_path} exists.")
    print(f"Using base_path: {base_path}")

    dates = ["02-06-24", "30-06-24", "21-07-24", "11-08-24"]
    total_start_time = time.time()

    for date in dates:
        print(f"\n{'-' * 40}")
        print(f"Processing data of date: {date}")
        print(f"{'-' * 40}")
        paths = create_paths(base_path, date)

        try:
            # process_3dpc_register(paths, date, base_path)
            process_3dpc_calculate(paths, date, base_path)
            process_ortho(paths, date, "rgb", base_path=base_path)
            process_ortho_glcm(paths, date, "rgb", base_path=base_path, calculate_glcm=True)
            process_ortho(paths, date, "msi", base_path=base_path)
        except Exception as e:
            print(f"Error processing date {date}: {str(e)}")
            continue

        print(f"{'-' * 40}")

    total_end_time = time.time()
    print(f"\nTotal processing time: {total_end_time - total_start_time:.2f} seconds")
