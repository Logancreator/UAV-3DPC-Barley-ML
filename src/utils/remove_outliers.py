import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

def dbscan_filter(pcd, eps=0.05, min_samples=10):
    """Filters a point cloud using DBSCAN clustering, selecting the largest cluster."""
    points = np.asarray(pcd.points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    # Find largest cluster, excluding noise
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True) # Exclude noise label (-1) when finding unique labels
    if not unique_labels.size:
        print("No clusters found (excluding noise).")
        return None

    largest_cluster_label = unique_labels[np.argmax(counts)]
    filtered_indices = np.where(labels == largest_cluster_label)[0]

    if not filtered_indices.size:  # Check for empty filtered indices
        print("Largest cluster contains no points.")
        return None

    # Create filtered point cloud
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[filtered_indices])

    # Copy colors and normals if they exist
    if pcd.has_colors():
        filtered_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[filtered_indices])
    if pcd.has_normals():
        filtered_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[filtered_indices])

    return filtered_pcd


def dbscan_filter_folder(folder_path, eps=0.05, min_samples=10):
    """Processes .ply files in a folder using DBSCAN filtering."""
    print(f"-> Processing folder: {folder_path}")

    ply_files = [f for f in os.listdir(folder_path)
                 if f.endswith('.ply') and f.startswith('cluster') and 'label_0' in f]

    for ply_file in ply_files:
        file_path = os.path.join(folder_path, ply_file)

        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                print(f"Skipping {file_path}: Point cloud is empty.")
                continue

            filtered_pcd = dbscan_filter(pcd, eps=eps, min_samples=min_samples)
            if filtered_pcd is None:
                print(f"Skipping {file_path}: No valid cluster found.")
                continue

            output_file_path = os.path.join(folder_path, f"filtered_{ply_file}")
            o3d.io.write_point_cloud(output_file_path, filtered_pcd)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"-> Processing completed for {folder_path}.")
    print(f"{'-' * 40}")