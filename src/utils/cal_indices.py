import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
import scipy.stats as stats
from pyproj import CRS
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
class ReadPointCloud:
    """
    A class for reading and handling point cloud data, supporting different
    types (RGB, multispectral, generic) and providing methods to access
    specific channels.
    """

    def __init__(self, points=None, data=None, channel_info=None, point_cloud_type='auto'):
        """
        Initializes a ReadPointCloud object.

        Args:
            points (numpy.ndarray, optional):  The XYZ coordinates of the points.
                Defaults to None.
            data (numpy.ndarray, optional):  The data associated with each point
                (e.g., color, intensity). Defaults to None.
            channel_info (list of dict, optional):  Information about each data
                channel (name, original name, data type). Defaults to None.
            point_cloud_type (str, optional): The type of point cloud ('rgb',
                'multispectral', 'generic', 'auto'). Defaults to 'auto'.
        """
        self.points = points
        self.data = data
        self.channel_info = channel_info
        self.point_cloud_type = point_cloud_type
        # Create a dictionary mapping channel names to indices for faster access
        self.name_to_idx = {info['name']: idx for idx, info in enumerate(channel_info)} if channel_info else {}
        # Infer RGB indices based on original channel names
        self.rgb_indices = self._infer_rgb_indices() if channel_info else None

    def _infer_rgb_indices(self):
        """
        Infers RGB channel indices based on original channel names (case-insensitive).

        Returns:
            list or None:  A list of RGB channel indices, or None if fewer than
                3 RGB channels are found.
        """
        rgb_indices = [idx for idx, info in enumerate(self.channel_info)
                       if info['original'].lower() in ['blue', 'green', 'red', 'b', 'g', 'r']]
        return rgb_indices if len(rgb_indices) >= 3 else None

    def get_channel(self, name):
        """
        Gets data for a specific channel by name.

        Args:
            name (str): The name of the channel.

        Returns:
            numpy.ndarray: The data for the specified channel.

        Raises:
            KeyError: If the channel name is not found.
        """
        if name not in self.name_to_idx:
            raise KeyError(f"Channel '{name}' not found. Available channels: {list(self.name_to_idx.keys())}")
        return self.data[:, self.name_to_idx[name]]

    def get_rgb(self, normalize=True, use_default=True):
        """
        Gets RGB channels for visualization.

        Args:
            normalize (bool, optional): Whether to normalize RGB values to the
                range [0, 1]. Defaults to True.
            use_default (bool, optional):  If standard RGB channels are not found,
                whether to use the first three channels as a fallback.
                Defaults to True.

        Returns:
            numpy.ndarray or None:  The RGB data, or None if RGB channels
                cannot be determined.
        """
        if self.point_cloud_type == 'generic':
            print("Warning: `get_rgb` called on a generic point cloud.")

        rgb_indices = self.rgb_indices
        if rgb_indices is None:
            if use_default and self.channel_info is not None and len(self.channel_info) >= 3:  # Added check for channel_info
                print("Warning: Standard BGR not found, using the first three channels by default.")
                rgb_indices = [0, 1, 2]
            else:
                print("Warning: Cannot find valid RGB channels, and `use_default` is False.")
                return None

        rgb = self.data[:, rgb_indices]
        if normalize:
            if np.issubdtype(rgb.dtype, np.integer):
                rgb = rgb.astype(np.float32) / 255.0  # Normalize integer data
            else:
                rgb = np.clip(rgb, 0, 1)  # Clip float data to [0, 1]
        return rgb

    def get_blue(self):
        """
        Gets blue channel data.

        Returns:
            numpy.ndarray or None: The blue channel data, or None if not found.
        """
        return self._get_color_channel('blue')

    def get_green(self):
        """
        Gets green channel data.

        Returns:
            numpy.ndarray or None: The green channel data, or None if not found.
        """
        return self._get_color_channel('green')

    def get_red(self):
        """
        Gets red channel data.

        Returns:
            numpy.ndarray or None: The red channel data, or None if not found.
        """
        return self._get_color_channel('red')

    def get_nir(self):
        """
        Gets near-infrared channel data.

        Returns:
            numpy.ndarray or None: The NIR channel data, or None if not found.
        """
        return self._get_optional_channel('nir')

    def get_thermal(self):
        """
        Gets thermal channel data.

        Returns:
            numpy.ndarray or None: The thermal channel data, or None if not found.
        """
        return self._get_optional_channel('thermal')

    def _get_color_channel(self, channel_name):
        """
        Gets a specified color channel data, handling errors (KeyError).

        Args:
            channel_name (str): The name of the color channel ('blue', 'green', 'red').

        Returns:
            numpy.ndarray or None: The channel data, or None if the channel is not found.
        """
        try:
            return self.get_channel(channel_name)
        except KeyError:
            print(f"{channel_name.capitalize()} channel not found." +
                  (" This might be a grayscale RGB point cloud." if self.point_cloud_type == 'rgb' else ""))
            return None

    def _get_optional_channel(self, channel_name):
        """
        Gets an optional channel data (NIR, Thermal), handling errors (KeyError).

        Args:
            channel_name (str): The name of the optional channel ('nir', 'thermal').

        Returns:
            numpy.ndarray or None: The channel data, or None if the channel is not found.
        """
        try:
            return self.get_channel(channel_name)
        except KeyError:
            print(f"{channel_name.upper()} channel not found." +
                  (" This is likely an RGB point cloud." if self.point_cloud_type == 'rgb' else ""))
            return None

    def has_channel(self, channel_name):
        """
        Checks if the point cloud object contains a specific channel.

        Args:
            channel_name (str): The name of the channel.

        Returns:
            bool: True if the channel exists, False otherwise.
        """
        return channel_name in self.name_to_idx

    def get_point_cloud_type(self):
        """
        Returns the point cloud type.

        Returns:
            str: The point cloud type.
        """
        return self.point_cloud_type

    @classmethod
    def from_txt(cls, filepath, point_cloud_type='auto'):
        """
        Reads point cloud data from a .txt file.

        Args:
            filepath (str): The path to the .txt file.
            point_cloud_type (str, optional): The type of point cloud ('rgb',
                'multispectral', 'generic', 'auto'). Defaults to 'auto'.

        Returns:
            ReadPointCloud: A ReadPointCloud object containing the data.

        Raises:
            ValueError: If there's an error loading the data or if the
                number of columns is inconsistent with the specified point
                cloud type.
        """
        try:
            data = np.loadtxt(filepath)
        except Exception as e:
            raise ValueError(f"Error loading data from {filepath}: {e}")

        num_cols = data.shape[1]

        # Determine point cloud type if 'auto'
        if point_cloud_type == 'auto':
            if num_cols == 9:
                point_cloud_type = 'rgb'
            elif num_cols == 12:
                point_cloud_type = 'multispectral'
            else:
                point_cloud_type = 'generic'  # Default to generic for other cases
        
        # Validate number of columns based on point cloud type
        if point_cloud_type == 'rgb' and (num_cols != 9 and num_cols != 6):
            raise ValueError(f"Expected 9 columns (XYZ, BGR, Normals) for RGB point cloud, but found {num_cols}.")

        if point_cloud_type == 'multispectral' and (num_cols != 12 and num_cols != 9):
            raise ValueError(f"Expected 12 columns (XYZ, BGR, RedEdge, NIR, Thermal, Normals) for multispectral point cloud, but found {num_cols}.")

        points = data[:, :3]  # First 3 columns are always XYZ

        # Handle different point cloud types
        if point_cloud_type == 'rgb':
            data_values = data[:, 3:6]  # Columns 4-6 are BGR
            channel_info = [
                {'name': 'blue', 'original': 'blue', 'dtype': 'float32'},
                {'name': 'green', 'original': 'green', 'dtype': 'float32'},
                {'name': 'red', 'original': 'red', 'dtype': 'float32'},
            ]

        elif point_cloud_type == 'multispectral':
            data_values = data[:, 3:9]  # Columns 4-9 are BGR, RedEdge, NIR, Thermal
            channel_info = [
                {'name': 'blue', 'original': 'blue', 'dtype': 'float32'},
                {'name': 'green', 'original': 'green', 'dtype': 'float32'},
                {'name': 'red', 'original': 'red', 'dtype': 'float32'},
                {'name': 'red_edge', 'original': 'red_edge', 'dtype': 'float32'},
                {'name': 'nir', 'original': 'nir', 'dtype': 'float32'},
                {'name': 'thermal', 'original': 'thermal', 'dtype': 'float32'},
            ]
        else:  # generic
            data_values = data[:, 3:] if data.shape[1] > 3 else None # All columns after the first 3
            # Create channel info for generic type
            channel_info = [{'name': f'feature{i}', 'original': f'feature{i}', 'dtype': 'float32'} for i in range(data_values.shape[1])] if data_values is not None else None

        return cls(points=points, data=data_values, channel_info=channel_info, point_cloud_type=point_cloud_type)
    
def get_rgb_VIs(r, g, b, nodata=0):
    """Calculates RGB-based Vegetation Indices."""
    rgb_max_agisoft = 256
    r = np.nan_to_num(r, nan=nodata) / rgb_max_agisoft
    g = np.nan_to_num(g, nan=nodata) / rgb_max_agisoft
    b = np.nan_to_num(b, nan=nodata) / rgb_max_agisoft

    indices = {
            'R': r,
            'G': g,
            'B': b,
            'ExG': 2 * g - r - b,  # 无分母，不需额外处理
            'GCC': np.where(r + g + b != 0, g / (r + g + b), nodata),
            'GLI': np.where(2 * g + r + b != 0, (2 * g - r - b) / (2 * g + r + b), nodata),
            'IKAW': np.where(r + b != 0, (r - b) / (r + b), nodata),
            'RGBVI': np.where(g**2 + b * r != 0, (g**2 - b * r) / (g**2 + b * r), nodata),
            'TGI': g - 0.39 * r - 0.61 * b,  # 无分母，不需额外处理
            'VARI': np.where(g + r - b != 0, (g - r) / (g + r - b), nodata),
            'VEG': np.where(r**0.667 * b**0.334 != 0, g / (r**0.667 * b**0.334), nodata),
            # 'VNDVI': np.where((r**0.1294) * (b**-0.3389) * (g**0.603118) != 0,
            #                 0.5268 * (g / ((r**0.1294) * (b**-0.3389) * (g**0.603118))), nodata),
            'NGRDI': np.where(g + r != 0, (g - r) / (g + r), nodata),
            'GRRI': np.where(r != 0, g / r, nodata),
            'MGRVI': np.where(g * g + r * r != 0, (g * g - r * r) / (g * g + r * r), nodata),
            'VARI': np.where(g + r - b != 0, (g - r) / (g + r - b), nodata)
        }
    return indices


def get_msi_VIs(r, g, b, redE, nir, thermal, nodata=-9999):
    """Calculates Multi-Spectral Indices (MSI)."""
    msi_DN_agisoft = 32768
    r = np.nan_to_num(r, nan=nodata) / msi_DN_agisoft
    g = np.nan_to_num(g, nan=nodata) / msi_DN_agisoft
    b = np.nan_to_num(b, nan=nodata) / msi_DN_agisoft
    redE = np.nan_to_num(redE, nan=nodata) / msi_DN_agisoft
    nir = np.nan_to_num(nir, nan=nodata) / msi_DN_agisoft
    thermal = np.nan_to_num(thermal, nan=nodata) / msi_DN_agisoft

    indices = {
        'R': r,
        'G': g,
        'B': b,
        'redE': redE,
        'nir': nir,
        'thermal': thermal,
        'NDVI': np.where(nir + r != 0, (nir - r) / (nir + r), nodata),
        'EVI': np.where(nir + 6 * r - 7.5 * b + 1 != 0, 2.5 * (nir - r) / (nir + 6 * r - 7.5 * b + 1), nodata),
        'NDWI': np.where(g + nir != 0, (g - nir) / (g + nir), nodata),
        'SAVI2': np.where(nir + r + 0.5 != 0, (nir - r) / (nir + r + 0.5) * (1 + 0.5), nodata),
        'PSRI': np.where(redE != 0, (r - g) / redE, nodata),
        'NDRE': np.where(nir + redE != 0, (nir - redE) / (nir + redE), nodata),
        'CIredE': np.where(redE != 0, (nir / redE) - 1, nodata),
        'NGRDI': np.where(g + r != 0, (g - r) / (g + r), nodata),
        'CIgreen': np.where(g != 0, (nir - 1) / g, nodata),
        'DVI': nir - r,
        'EVI2': np.where(nir + 2.4 * r + 1 != 0, (2.5 * (nir - r)) / (nir + 2.4 * r + 1), nodata),
        'GNDVI': np.where(nir + g != 0, (nir - g) / (nir + g), nodata),
        'MCARI': ((redE - g) - 0.2 * (redE - r)) * np.where(r != 0, redE / r, nodata),
        'MSAVI': (2 * nir + 1 - np.sqrt(np.where((2 * nir + 1) ** 2 - 8 * (nir - r) >= 0, (2 * nir + 1) ** 2 - 8 * (nir - r), 0))) / 2,
        'MTVI2': np.where(((2 * nir + 1) ** 2 - (6 * nir - 5 * r)) ** 0.5 - 0.5 != 0,
                          (1.5 * (1.2 * (nir - g) - 2.5 * (r - g))) / (((2 * nir + 1) ** 2 - (6 * nir - 5 * r)) ** 0.5 - 0.5), nodata),
        'OSAVI': np.where(nir + r + 0.16 != 0, (1.16 * (nir - r)) / (nir + r + 0.16), nodata),
        'REDVI': nir - redE,  # Assuming this is DVI-REG
        'RESAVI': np.where(nir + redE + 0.5 != 0, (1.5 * (nir - redE)) / (nir + redE + 0.5), nodata),
        'TVI': np.sqrt(np.where(nir + r != 0, (nir - r) / (nir + r) + 0.5, 0)),
        'Vlopt1': np.where(np.log(nir) != 0, (100 * (np.log(nir) - np.log(redE))) / np.log(nir), nodata),
        'WDRVI': np.where(0.1 * nir + r != 0, (0.1 * nir - r) / (0.1 * nir + r), nodata),
        # New indices from the table
        'BNDVI': np.where(nir + b != 0, (nir - b) / (nir + b), nodata),
        'CI-RED': np.where(r != 0, (nir / r) - 1, nodata),
        'CVI': np.where(g != 0, (nir / g) * (r / g), nodata),
        'DVI-GREEN': nir - g,
        'GARI': np.where(nir + (g - 1.7 * (b - r)) != 0, (nir - (g - 1.7 * (b - r))) / (nir + (g - 1.7 * (b - r))), nodata),
        'GOSAVI': np.where(nir + g + 0.16 != 0, (nir - g) / (nir + g + 0.16), nodata),
        'GRVI': np.where(g + r != 0, (g - r) / (g + r), nodata),
        'LCI': np.where(nir - r != 0, (nir - redE) / (nir + r), nodata),
        'MCARI1': 1.2 * (2.5 * (nir - r) - 1.3 * (nir - g)),
        'MCARI2': np.where(((2 * nir + 1) ** 2 - 6 * (nir - 5 * np.sqrt(r)) - 0.5) != 0,
                           (3.75 * (nir - r) - 1.95 * (nir - g)) / np.sqrt((2 * nir + 1) ** 2 - 6 * (nir - 5 * np.sqrt(r)) - 0.5), nodata),
        'MNLI': np.where(nir ** 2 + r + 0.5 != 0, (1.5 * nir ** 2 - 1.5 * g) / (nir ** 2 + r + 0.5), nodata),
        'MSR': np.where(r != 0, ((nir / r) - 1) / np.sqrt((nir / r) + 1), nodata),
        'MSR-REG': np.where(redE != 0, ((nir / redE) - 1) / np.sqrt((nir / redE) + 1), nodata),
        'NDREI': np.where(redE + g != 0, (redE - g) / (redE + g), nodata),
        'NAVI': np.where(nir != 0, 1 - (r / nir), nodata),
        'OSAVI-REG': np.where(nir + redE + 0.16 != 0, 1.6 * (nir - redE) / (nir + redE + 0.16), nodata),
        'RDVI': np.where(nir + r != 0, (nir - r) / np.sqrt(nir + r), nodata),
        'RDVI-REG': np.where(nir + redE != 0, (nir - redE) / np.sqrt(nir + redE), nodata),
        'RGBVI': np.where(g ** 2 + b * r != 0, (g ** 2 - b * r) / (g ** 2 + b * r), nodata),
        'RTVI-CORE': 100 * (nir - redE) - 10 * (nir - g),
        'RVI': np.where(r != 0, nir / r, nodata),
        'SAVI': np.where(nir + r + 0.5 != 0, 1.5 * (nir - r) / (nir + r + 0.5), nodata),
        'SAVI-GREEN': np.where(nir + g + 0.5 != 0, 1.5 * (nir - g) / (nir + g + 0.5), nodata),
        'S-CCCI': np.where(((nir + r) * (nir + redE)) != 0, ((nir - redE) / (nir + redE)) / ((nir - r) / (nir + r)), nodata),
        'SIPI': np.where(nir - r != 0, (nir - b) / (nir - r), nodata),
        'SR-REG': np.where(redE != 0, nir / redE, nodata),
        'TCARI': 3 * ((redE - r) - 0.2 * (redE - g) * np.where(r != 0, redE / r, nodata)),
        'OSAVI': np.where((nir + r + 0.16) != 0, (3 * ((redE - r) - 0.2 * (redE - g) * np.where(r != 0, redE / r, nodata))) / ((1.16 * (nir - r)) / (nir + r + 0.16)), nodata)
    }
    return indices


def get_canopy_indices(point_cloud):
    """Calculates canopy height metrics."""
    heights = point_cloud[:, 2]
    return {
        'PH_max': np.max(heights),
        'PH_min': np.min(heights),
        'PH_mean': np.mean(heights)
    }

# def calculate_glcm_features(array, distances=[1], angles=[63 * np.pi / 180], properties=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']):
#     """Calculates GLCM features for a given array."""
#     glcm = graycomatrix(array, distances=distances, angles=angles, symmetric=True, normed=True)
#     features = {}
#     for prop in properties:
#         features[prop] = graycoprops(glcm, prop).flatten()  # Flatten to handle multiple distances/angles
#     # Calculate mean for each property across all distances/angles:
#     mean_features = {f"{prop}_mean": np.mean(val) if val.size > 0 else np.nan for prop, val in features.items()} 
#     return mean_features

def calculate_rgb_3d_indices(plant_point_cloud_path, nodata=0):
    """Calculates RGB and 3D indices from a point cloud using ReadPointCloud class."""
    try:
        # Load point cloud using ReadPointCloud class
        plant_pc = ReadPointCloud.from_txt(plant_point_cloud_path, point_cloud_type='rgb')

        if plant_pc.points is None or plant_pc.data is None:
            raise ValueError("The point cloud has no points or data.")

        plant_points = plant_pc.points
        # Get RGB data using the ReadPointCloud class
        colors = plant_pc.get_rgb(normalize=False, use_default=False)

        if colors is None:
            raise ValueError("Could not retrieve valid RGB data.")

        b, g, r = colors[:, 0], colors[:, 1], colors[:, 2]

        # Filter out points where all RGB values are zero
        valid_mask = (r != 0) | (g != 0) | (b != 0)
        b, g, r = b[valid_mask], g[valid_mask], r[valid_mask]
        plant_points = plant_points[valid_mask]

        if len(plant_points) == 0 or len(r) == 0:
            print("No valid points remain after filtering.")
            return None, None

        canopy_metrics = get_canopy_indices(plant_points)
        rgb_indices = get_rgb_VIs(r, g, b, nodata=nodata)
        # 计算统计值并清理异常值
        rgb_statistics = {}
        for index_name, values in rgb_indices.items():
            values_clean = np.where(np.isinf(values) | np.isnan(values), nodata, values)
            for stat in ['mean', 'max', 'min']:
                if stat == 'mean':
                    rgb_statistics[f'{index_name}_{stat}'] = np.nanmean(values_clean)
                else:
                    rgb_statistics[f'{index_name}_{stat}'] = getattr(np, stat)(values_clean)

        return canopy_metrics, rgb_statistics

    except ValueError as e:
        print(f"Skipping {plant_point_cloud_path}: {str(e)}")
        return None, None


def calculate_msi_3d_indices(plant_point_cloud_path, nodata=0):
    """Calculates MSI and 3D indices from a point cloud using ReadPointCloud class."""
    try:
        # Load point cloud using ReadPointCloud class
        plant_pc = ReadPointCloud.from_txt(plant_point_cloud_path, point_cloud_type='multispectral')

        if plant_pc.points is None or plant_pc.data is None:
            raise ValueError("The point cloud has no points or data.")

        plant_points = plant_pc.points

        # Get MSI data using the ReadPointCloud class
        msi_data = plant_pc.data

        if msi_data is None or msi_data.shape[1] < 5:
            raise ValueError("Could not retrieve valid MSI data with at least 5 channels.")

        b, g, r, redEdge, nir, thermal = msi_data[:, 0], msi_data[:, 1], msi_data[:, 2], msi_data[:, 3], msi_data[:, 4], msi_data[:, 5]


        # Filter out points where all MSI values are zero
        valid_mask = (r != 0) | (g != 0) | (b != 0) | (redEdge != 0) | (nir != 0)
        b, g, r, redEdge, nir, thermal = b[valid_mask], g[valid_mask], r[valid_mask], redEdge[valid_mask], nir[valid_mask], thermal[valid_mask]
        plant_points = plant_points[valid_mask]
        if len(plant_points) == 0 or len(r) == 0:
            print("No valid points remain after filtering.")
            return None, None

        canopy_metrics = get_canopy_indices(plant_points)
        msi_indices = get_msi_VIs(r, g, b, redEdge, nir, thermal, nodata=nodata)
        # 计算统计值并清理异常值
        msi_statistics = {}
        for index_name, values in msi_indices.items():
            values_clean = np.where(np.isinf(values) | np.isnan(values), nodata, values)
            for stat in ['mean', 'max', 'min']:
                if stat == 'mean':
                    msi_statistics[f'{index_name}_{stat}'] = np.nanmean(values_clean)
                else:
                    msi_statistics[f'{index_name}_{stat}'] = getattr(np, stat)(values_clean)

        return canopy_metrics, msi_statistics

    except ValueError as e:
        print(f"Skipping {plant_point_cloud_path}: {str(e)}")
        return None, None


def process_point_cloud_folder(folder_path, date, index_type, sub_folder, base_path):
    """Processes all .ply files in a folder, calculates indices, and saves results."""
    folder_path = os.path.join(folder_path, sub_folder)
    print(f"-> Processing folder: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return

    os.makedirs(base_path, exist_ok=True)

    #txt_files = sorted([f for f in os.listdir(folder_path) if f.startswith('extracted_fid') and f.endswith('.txt')], key=lambda x: int(re.search(r'(\d+)', x).group()))
    try:
        txt_files = sorted(
            [f for f in os.listdir(folder_path) if f.startswith('extracted_fid') and f.endswith('.txt')],
            key=lambda x: int(re.search(r'(\d+)', x).group()) if re.search(r'(\d+)', x) else 0
        )
    except Exception as e:
        print(f"Error sorting files: {e}")
        
    results = []
    for i, txt_file in enumerate(txt_files):
        print(f"Processing file {i + 1}/{len(txt_files)}: {txt_file}")
        file_path = os.path.join(folder_path, txt_file)

        calculate_function = calculate_rgb_3d_indices if index_type == 'rgb' else calculate_msi_3d_indices
        canopy_metrics, indices_statistics = calculate_function(file_path)

        if canopy_metrics is None or indices_statistics is None:
            print(f"Skipping {txt_file} due to missing data.")
            continue

        combined_dict = {**canopy_metrics, **indices_statistics, 'Object_ID': i + 1, 'file_name': txt_file}
        results.append(combined_dict)

    if results:
        df = pd.DataFrame.from_records(results)
        cols = ['Object_ID'] + [col for col in df.columns if col != 'Object_ID']
        df = df[cols]
        print(f"{index_type.upper()} 3DPC Processing completed. Results: \n")
        print(df)

        output_file_path = os.path.join(base_path, f"{index_type}_threeD_indices_{date}.csv")
        df.to_csv(output_file_path, index=False)
        print(f"Results saved to: {output_file_path}")
    else:
        print("No valid results to save.")

    print(f"{'-' * 40}")


def calculate_rgb_3d_indices_folder(folder_path, date, sub_folder, base_path):
    process_point_cloud_folder(folder_path, date, 'rgb', sub_folder, base_path)


def calculate_msi_3d_indices_folder(folder_path, date, sub_folder, base_path):
    process_point_cloud_folder(folder_path, date, 'msi', sub_folder, base_path)


def calculate_custom_stats(vector):
    """Calculates custom statistics from a 1D vector, handling NaN and empty arrays."""
    vector = np.ma.filled(vector, np.nan)  # Replace masked values with NaN
    vector = vector[~np.isnan(vector)]  # Remove NaN values

    if len(vector) == 0:
        return {stat: np.nan for stat in [
            'Mean', 'Std', 'Median', 'NMAD', 'BwMv', 'P2.5%', 'Q25%', 'Q75%', 'P97.5%',
            'IQR', 'IPR90%', 'IPR99%', 'Skewness', 'Kurtosis', 'CV', 'Min', 'Max', 'Range',
            'Geometric Mean', 'MAD', 'Excess Kurtosis'
        ]}

    mean = np.mean(vector)
    std = np.std(vector, ddof=1)
    median = np.median(vector)
    nmad = 1.4826 * np.median(np.abs(vector - median))

    p2_5, q25, q75, p97_5 = np.percentile(vector, [2.5, 25, 75, 97.5])
    iqr = q75 - q25
    ipr90 = np.percentile(vector, 95) - np.percentile(vector, 5)
    ipr99 = np.percentile(vector, [99.5, 0.5])
    ipr99 = ipr99[0] - ipr99[1]

    skewness = stats.skew(vector)
    kurtosis = stats.kurtosis(vector)

    def biweight_midvariance(arr):
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        u = (arr - med) / (9 * mad)
        w = (1 - u**2)**2
        w[np.abs(u) >= 1] = 0
        bwmv = len(arr) * np.sum(w * (arr - med)**2) / (np.sum(w) * (np.sum(w) - 1))
        return np.sqrt(bwmv)

    bwmv = biweight_midvariance(vector)

    coefficient_of_variation = std / mean if mean != 0 else 1e-10
    min_val, max_val = np.min(vector), np.max(vector)
    range_val = max_val - min_val
    geometric_mean = stats.gmean(vector)
    mad = np.median(np.abs(vector - median))
    excess_kurtosis = kurtosis - 3

    return {
        'Mean': mean, 'Std': std, 'Median': median, 'NMAD': nmad, 'BwMv': bwmv, 'P2.5%': p2_5,
        'Q25%': q25, 'Q75%': q75, 'P97.5%': p97_5, 'IQR': iqr, 'IPR90%': ipr90, 'IPR99%': ipr99,
        'Skewness': skewness, 'Kurtosis': kurtosis, 'CV': coefficient_of_variation, 'Min': min_val,
        'Max': max_val, 'Range': range_val, 'Geometric Mean': geometric_mean, 'MAD': mad,
        'Excess Kurtosis': excess_kurtosis
    }

# def zonal_stats_VIs(tiff, shpfile, index_name, index_type, band_count, calculate_glcm=False):
#     """Calculates zonal statistics for a given vegetation index from a raster."""
#     gdf = gpd.read_file(shpfile)

#     with rasterio.open(tiff) as src:
#         if gdf.crs != src.crs:
#             gdf = gdf.to_crs(src.crs)

#         bands = [src.read(i).astype("float32") for i in range(1, band_count + 1)]
#         nodata = src.nodata if src.nodata is not None else 0
        
#         if index_type == 'rgb':
#             if calculate_glcm:
#                 # Stack RGB bands into a 3D array (height, width, 3)
#                 rgb_array = np.stack(bands, axis=-1)  # Shape: (height, width, 3)
#                 # Convert RGB to grayscale
#                 gray_index_array = rgb2gray(rgb_array)
#                 # Normalize to [0, 255] and convert to uint8
#                 min_val, max_val = np.nanmin(gray_index_array), np.nanmax(gray_index_array)
#                 if min_val == max_val:  # Avoid division by zero
#                     gray_index_array = np.zeros_like(gray_index_array, dtype="uint8")
#                 else:
#                     gray_index_array = ((gray_index_array - min_val) / (max_val - min_val) * 255).astype("uint8")
#                 gray_index_array = np.nan_to_num(gray_index_array, nan=0)  # Replace NaN with 0
#                 index_array = gray_index_array  # For consistency in stats calculation
#             else:
#                 # Calculate vegetation index as before
#                 index_array = get_rgb_VIs(*bands, nodata=nodata)[index_name]
#         elif index_type == 'msi':
#             index_array = get_msi_VIs(*bands, nodata=nodata)[index_name]
#         else:
#             raise ValueError(f"Unsupported index_type: '{index_type}'. Expected 'rgb' or 'msi'.")

#         index_array = np.where(np.isinf(index_array) | np.isnan(index_array), nodata, index_array)

#         stats = ['mean', 'max', 'min']
#         if calculate_glcm and index_type == 'rgb':
#             glcm_features = calculate_glcm_features(gray_index_array)
#             index_stats = []  # Use list to store dictionary results when including GLCM
#             for idx, geom in enumerate(gdf.geometry):
#                 window = rasterio.features.geometry_window(src, [geom])
#                 subset = index_array[window.toslices()]
#                 masked_subset = np.ma.masked_where(subset == nodata, subset)
#                 feature_values = {stat: getattr(np, stat)(masked_subset) for stat in stats}
#                 # Add GLCM features for the window
#                 subset_gray = gray_index_array[window.toslices()]
#                 feature_values.update(calculate_glcm_features(subset_gray))
#                 index_stats.append(feature_values)
#             print("Finished GLCM calculation")
#         else:
#             index_stats = zonal_stats(
#                 vectors=gdf,
#                 raster=index_array,
#                 affine=src.transform,
#                 stats=stats,
#                 nodata=nodata
#             )
#     return index_stats

import geopandas as gpd
import rasterio
import numpy as np
from rasterio.mask import mask
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

def calculate_glcm_features(gray_array):
    """
    Calculates GLCM features for a given grayscale array.
    Averages the properties across all angles to create a rotationally-invariant measure.
    """
    # Ensure the input is an 8-bit integer array
    if gray_array.dtype != np.uint8:
         # Normalize to [0, 255] and convert to uint8 if it's not already
        min_val, max_val = np.nanmin(gray_array), np.nanmax(gray_array)
        if min_val == max_val:  # Avoid division by zero for uniform areas
            gray_array = np.zeros_like(gray_array, dtype=np.uint8)
        else:
            gray_array = ((gray_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Calculate GLCM for multiple angles
    glcm = graycomatrix(gray_array, 
                        distances=[1], 
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, 
                        symmetric=True, 
                        normed=True)

    # Calculate properties and average them over the angles
    # .mean() will average all values in the returned (1, 4) array
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'ASM': graycoprops(glcm, 'ASM').mean() # ASM (Angular Second Moment) is another name for energy
    }
    
    return features

def zonal_stats_VIs(tiff, shpfile, index_name, index_type, band_count, calculate_glcm=False):
    """Calculates zonal statistics for a given vegetation index from a raster."""
    gdf = gpd.read_file(shpfile)
    stats_list = []

    with rasterio.open(tiff) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        nodata = src.nodata if src.nodata is not None else 0

        # Iterate over each feature in the shapefile
        for geom in gdf.geometry:
            try:
                # Mask the raster using the single geometry
                out_image, out_transform = mask(src, [geom], crop=True, all_touched=True)

                if out_image.size == 0:
                    print("Warning: Geometry resulted in an empty masked image. Skipping.")
                    continue

                bands = [out_image[i].astype("float32") for i in range(band_count)]
                
                feature_stats = {}

                if index_type == 'rgb':
                    if calculate_glcm:
                        # Stack RGB bands
                        rgb_array = np.stack(bands, axis=-1)
                        # Convert to grayscale
                        gray_index_array = rgb2gray(rgb_array)

                        # Normalize to [0, 255] and convert to uint8
                        min_val, max_val = np.nanmin(gray_index_array), np.nanmax(gray_index_array)
                        if min_val == max_val:
                            gray_index_array = np.zeros_like(gray_index_array, dtype="uint8")
                        else:
                            gray_index_array = ((gray_index_array - min_val) / (max_val - min_val) * 255).astype("uint8")
                        
                        gray_index_array = np.nan_to_num(gray_index_array, nan=0)
                        
                        # Calculate basic stats on grayscale image
                        masked_subset = np.ma.masked_where(gray_index_array == nodata, gray_index_array)
                        for stat in ['mean', 'max', 'min']:
                             feature_stats[stat] = getattr(np, stat)(masked_subset) if masked_subset.count() > 0 else np.nan
                        
                        # Calculate GLCM features
                        if gray_index_array.size > 0:
                            feature_stats.update(calculate_glcm_features(gray_index_array))
                        
                    else:
                        # Original logic for vegetation indices
                        index_array = get_rgb_VIs(*bands, nodata=nodata)[index_name]
                        index_array = np.where(np.isinf(index_array) | np.isnan(index_array), nodata, index_array)
                        stats_result = zonal_stats([geom], index_array, affine=out_transform, stats=['mean', 'max', 'min'], nodata=nodata)
                        if stats_result:
                            feature_stats = stats_result[0]

                elif index_type == 'msi':
                    index_array = get_msi_VIs(*bands, nodata=nodata)[index_name]
                    index_array = np.where(np.isinf(index_array) | np.isnan(index_array), nodata, index_array)
                    stats_result = zonal_stats([geom], index_array, affine=out_transform, stats=['mean', 'max', 'min'], nodata=nodata)
                    if stats_result:
                        feature_stats = stats_result[0]

                stats_list.append(feature_stats)

            except ValueError as e:
                print(f"Skipping geometry due to error during masking: {e}")
                continue
    
    return stats_list

def zonal_stats_rgb_VIs(tiff, shpfile, index_name, calculate_glcm=True):
    """Calculates zonal statistics for RGB-based Vegetation Indices."""
    return zonal_stats_VIs(tiff, shpfile, index_name, "rgb", 3, calculate_glcm)


def zonal_stats_msi_VIs(tiff, shpfile, index_name):
    """Calculates zonal statistics for Multi-Spectral Indices (MSI)."""
    return zonal_stats_VIs(tiff, shpfile, index_name, "msi", 6)


def combine_stats_to_dataframe(stats_dict, add_object_id=False):
    dfs = []
    for indicator, stats_list in stats_dict.items():
        df = pd.DataFrame(stats_list)
        df.columns = [f"{indicator}_{col}" for col in df.columns]
        dfs.append(df)
    combined_df = pd.concat(dfs, axis=1)
    if add_object_id:
        combined_df.insert(0, 'Object_ID', range(1, len(combined_df) + 1))
    return combined_df
