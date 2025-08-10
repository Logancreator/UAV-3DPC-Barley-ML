# -*- coding: utf-8 -*-
import os
import time
import rasterio
import geopandas as gpd
import open3d as o3d
import numpy as np
from shapely.geometry import Point
from PIL import Image, ImageFile
from rasterio.transform import rowcol
from shapely.strtree import STRtree
from joblib import Parallel, delayed
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

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
        if self.channel_info is None:
            return None
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

def extract_and_save_point_clouds(shp_path, point_cloud_path, output_folder, point_cloud_type='auto', buffer_distance=0, block_size=1000000):
    """
    Extracts and saves point clouds for each polygon in a shapefile.

    This function reads a shapefile and a point cloud file, extracts points
    within each (optionally buffered) polygon from the shapefile, and saves
    the extracted points to separate text files.  It leverages parallel
    processing for improved performance.

    Args:
        shp_path (str): Path to the input shapefile (.shp).
        point_cloud_path (str): Path to the input point cloud file (.txt).
        output_folder (str): Path to the folder where extracted point clouds
            will be saved.
        point_cloud_type (str, optional):  Type of point cloud data.  This is
            passed to the `ReadPointCloud` class. Defaults to 'auto'.  See
            `ReadPointCloud` documentation for supported types.
        buffer_distance (float, optional): Distance to buffer each polygon
            before extracting points.  Units are the same as the shapefile's
            coordinate system. Defaults to 0 (no buffering).
        block_size (int, optional): Number of points to process in each
            parallel block.  Larger blocks may use more memory but could be
            faster. Defaults to 1,000,000.

    Returns:
        None.  Saves the extracted point clouds as .txt files in the specified
        output folder.  The files are named "extracted_fid_XXX.txt", where XXX
        is the feature ID (FID) of the corresponding polygon, padded to three
        digits.

    Raises:
        FileNotFoundError: If the shapefile or point cloud file does not exist.
        ValueError: If the block size is not a positive integer.
        Exception:  For any other errors during processing (e.g., file I/O,
            geometric operations, parallel processing issues).
    """

    total_start_time = time.time()

    print(f"-> Input Shapefile path: {shp_path}")
    print(f"-> Input Point Cloud path: {point_cloud_path}")
    print(f"-> Output folder: {output_folder}")
    print(f"-> Buffer distance: {buffer_distance}")
    print(f"-> Block size: {block_size}")
    print(f"-> Point cloud type: {point_cloud_type}")

    # --- 1. Read Shapefile and Point Cloud ---
    read_data_start_time = time.time()

    try:
        shp_df = gpd.read_file(shp_path, encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    except Exception as e:
        raise Exception(f"Error reading shapefile: {e}")

    try:
        # Use your ReadPointCloud class to read point cloud data
        point_cloud = ReadPointCloud.from_txt(point_cloud_path, point_cloud_type=point_cloud_type)
        points = point_cloud.points
        data = point_cloud.data  # Get data from all channels
    except FileNotFoundError:
        raise FileNotFoundError(f"Point cloud file not found: {point_cloud_path}")
    except Exception as e:
        raise Exception(f"Error reading point cloud: {e}")

    read_data_end_time = time.time()
    print(f"-> Time taken to read data: {read_data_end_time - read_data_start_time:.2f} seconds")

    # --- 2. Create Output Folder ---
    os.makedirs(output_folder, exist_ok=True)  # Creates the folder if it doesn't exist, does nothing if it does.

    # --- 3. Buffer Polygons ---
    buffer_start_time = time.time()
    try:
        buffered_polygons = {fid: polygon.buffer(buffer_distance) for fid, polygon in enumerate(shp_df.geometry, start=1)}
    except Exception as e:
        raise Exception(f"Error buffering polygons: {e}")
    buffer_end_time = time.time()
    print(f"-> Time taken to buffer polygons: {buffer_end_time - buffer_start_time:.2f} seconds")

    # --- 4. Parallel Processing Function ---
    def process_block(block_indices, buffered_polygons):
        """
        Processes a block of points to find those within buffered polygons.

        Args:
            block_indices (list): Indices of points belonging to the current block.
            buffered_polygons (dict): Dictionary of buffered polygons (FID: polygon).

        Returns:
            dict: Dictionary (FID: list of point indices) containing the indices of
                points within each buffered polygon for the current block.
        """
        block_points = points[block_indices]
        block_points_as_shapely = [Point(p) for p in block_points]
        block_tree = STRtree(block_points_as_shapely)

        block_results = {}
        for fid, buffered_polygon in buffered_polygons.items():
            # Efficiently find potential points within the bounding box of the polygon
            potential_indices = block_tree.query(buffered_polygon)

            # Filter to find points *actually* inside the polygon (not just the bounding box)
            indices_within = [block_indices[i] for i in potential_indices if buffered_polygon.contains(block_points_as_shapely[i])]
            block_results[fid] = indices_within

        return block_results

    # --- 5. Process Blocks and Extract Points ---
    parallel_start_time = time.time()
    num_points = len(points)

    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("Block size must be a positive integer.")

    num_blocks = (num_points + block_size - 1) // block_size  # Calculate the number of blocks
    try:
        with Parallel(n_jobs=-2, return_as="list") as parallel:
            block_results = parallel(delayed(process_block)(list(range(block_idx * block_size, min((block_idx + 1) * block_size, num_points))), buffered_polygons)
                                     for block_idx in range(num_blocks))
            if any(result is None for result in block_results):
                raise Exception("Some blocks failed to process")

    except Exception as e:
        raise Exception(f"Error during parallel processing: {e}")

    parallel_end_time = time.time()
    print(f"-> Time taken for parallel processing: {parallel_end_time - parallel_start_time:.2f} seconds")

    # --- 6. Merge and Save Results ---
    merge_and_save_start_time = time.time()
    try:
        for fid in buffered_polygons.keys():
            indices = []
            for block_result in block_results:
                if fid in block_result:
                    indices.extend(block_result[fid])

            if indices:
                extracted_points = points[indices]
                extracted_data = data[indices]  # Extract data from all channels

                # Save as a txt file
                formatted_fid = f"{fid:03d}"  # Format FID with leading zeros
                output_path = os.path.join(output_folder, f"extracted_fid_{formatted_fid}.txt")
                # Combine point coordinates and data from all channels, then save to txt file
                output_data = np.concatenate((extracted_points, extracted_data), axis=1)
                np.savetxt(output_path, output_data, fmt='%.6f')  # Adjust format as needed

    except Exception as e:
        raise Exception(f"Error merging and saving results: {e}")

    merge_and_save_end_time = time.time()
    print(f"-> Time taken to merge and save results: {merge_and_save_end_time - merge_and_save_start_time:.2f} seconds")

    total_end_time = time.time()
    print(f"-> Total time taken: {total_end_time - total_start_time:.2f} seconds")


def extract_and_save_ortho(shp_path, ortho_path, output_folder):
    """
    Extracts and saves orthorectified image regions corresponding to each
    polygon in a shapefile.

    This function reads a shapefile and a georeferenced orthorectified image
    (orthomosaic), extracts the portion of the image that falls within each
    polygon in the shapefile, and saves each extracted region as a separate
    GeoTIFF file.  It handles coordinate system differences and ensures that
    only the image data within the polygon is included in the output.

    Args:
        shp_path (str): Path to the input shapefile (.shp).
        ortho_path (str): Path to the input orthorectified image (e.g., .tif).
        output_folder (str): Path to the folder where extracted image regions
            will be saved.

    Returns:
        None. Saves the extracted image regions as .tif files in the specified
        output folder. Files are named "extracted_fid_X.tif", where X is the
        index of the polygon in the shapefile.

    Raises:
        FileNotFoundError: If the shapefile or orthophoto file does not exist.
        rasterio.RasterioIOError:  If the orthophoto file cannot be opened.
        Exception: For other errors during processing (e.g., CRS mismatch,
            geometric operations, file I/O).
    """

    print("-> Start Extracting the ortho...")
    start_time = time.time()

    # --- 1. Read Shapefile and Orthophoto ---
    try:
        shp_df = gpd.read_file(shp_path, encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")
    except Exception as e:
        raise Exception(f"Error reading shapefile: {e}")

    try:
        with rasterio.open(ortho_path) as src:
            ortho_array = src.read()
            transform = src.transform
            crs = src.crs
    except FileNotFoundError:
        raise FileNotFoundError(f"Orthophoto file not found: {ortho_path}")
    except rasterio.RasterioIOError as e:  # More specific exception for rasterio
        raise rasterio.RasterioIOError(f"Error opening orthophoto: {e}")
    except Exception as e:
        raise Exception(f"Error reading orthophoto: {e}")

    # --- 2. CRS Handling ---
    if shp_df.crs != crs:
        print("-> Shapefile and orthophoto CRS are different. Transforming Shapefile CRS...")
        try:
            shp_df = shp_df.to_crs(crs)
        except Exception as e:
            raise Exception(f"Error transforming shapefile CRS: {e}")
        print("Transformed Shapefile CRS:", shp_df.crs)

    # --- 3. Create Output Folder ---
    os.makedirs(output_folder, exist_ok=True)

    # --- 4. Iterate Through Polygons and Extract Image Regions ---
    for fid, polygon in shp_df.iterrows():  # Iterate by row index (FID)
        try:
            # --- 4.1. Calculate Window ---
            min_x, min_y, max_x, max_y = polygon.geometry.bounds

            # Convert geographic coordinates to raster pixel coordinates
            min_row, min_col = rowcol(transform, min_x, max_y)  # Note the order: max_y for min_row
            max_row, max_col = rowcol(transform, max_x, min_y)  # Note the order: min_y for max_row
            
            if min_row >= max_row or min_col >= max_col:
                raise ValueError("Invalid window coordinates")
                
            # Ensure row/col indices are within image bounds
            min_row = max(0, min_row)
            min_col = max(0, min_col)
            max_row = min(ortho_array.shape[1], max_row)  # Corrected: shape[1] is height (rows)
            max_col = min(ortho_array.shape[2], max_col)  # Corrected: shape[2] is width (cols)


            window = rasterio.windows.Window(min_col, min_row, max_col - min_col, max_row - min_row)
            extracted_region = ortho_array[:, min_row:max_row, min_col:max_col]

            # --- 4.2. Create Mask ---
            # Correctly create the mask array using the window size
            mask = np.zeros((max_row - min_row, max_col - min_col), dtype=bool)
            for y in range(max_row - min_row):
                for x in range(max_col - min_col):
                    px, py = transform * (min_col + x + 0.5, min_row + y + 0.5)  # Pixel center
                    if polygon.geometry.contains(Point(px, py)):
                        mask[y, x] = True

            # --- 4.3. Apply Mask and Extract Image ---
            extracted_image = np.zeros_like(extracted_region)
            for band in range(extracted_region.shape[0]):
                extracted_image[band][mask] = extracted_region[band][mask]

            # --- 4.4. Save Extracted Image ---
            if np.any(extracted_image):  # Check if there's any data in the extracted image
                output_path = os.path.join(output_folder, f"extracted_fid_{fid}.tif")
                with rasterio.open(
                        output_path,
                        'w',
                        driver='GTiff',
                        height=extracted_image.shape[1],
                        width=extracted_image.shape[2],
                        count=extracted_image.shape[0],
                        dtype=extracted_image.dtype,
                        transform=rasterio.windows.transform(window, transform),
                        crs=crs
                ) as dst:
                    dst.write(extracted_image)
                print(f"-> -> Saving ortho to: {output_path}")
            else:
                print(f"-> No ortho data is located in the polygon (FID {fid})")  # Informative message

        except Exception as e:
            print(f"Error processing polygon FID {fid}: {e}")  # Error handling per polygon
            continue  # Continue to the next polygon if an error occurs

    end_time = time.time()
    print(f"-> All images are extracted and saved in: {end_time - start_time:.2f} seconds!")