import os
import CSF
import time
import numpy as np
from sklearn.cluster import KMeans
import open3d as o3d
import matplotlib.pyplot as plt

def cluster_point_cloud(point_cloud, n_clusters=2, random_state=0):
    """Performs K-Means clustering on a point cloud based on its colors."""
    colors = np.asarray(point_cloud.colors)
    if colors.size == 0:
        raise ValueError("Point cloud has no color information.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto').fit(colors)  # Set n_init explicitly to avoid warning
    labels = kmeans.labels_

    # Sort labels by cluster size and map to new labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_labels = unique_labels[np.argsort(counts)[::-1]]
    label_mapping = dict(zip(sorted_labels, range(len(sorted_labels))))
    kmeans_labels = np.vectorize(label_mapping.get)(labels)  # Vectorized label mapping

    return kmeans_labels

def visualize_clustered_point_cloud(point_cloud, labels, input_file):
    """Visualizes and saves the clustered point cloud as an image."""
    points = np.asarray(point_cloud.points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Fixed colors for two clusters
    colors = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])  # Blue, Yellow

    # Assign colors based on labels
    point_colors = colors[labels]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([np.min(points[:, 0]), np.max(points[:, 0])])
    ax.set_ylim([np.min(points[:, 1]), np.max(points[:, 1])])
    ax.set_zlim([np.min(points[:, 2]), np.max(points[:, 2])])
    ax.view_init(elev=30, azim=45)

    dir_name, file_name = os.path.split(input_file)
    file_base = os.path.splitext(file_name)[0]  # Extract base filename

    output_image_filename = os.path.join(dir_name, f"cluster_visualization_{file_base}.png")
    plt.savefig(output_image_filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Clustered visualization image saved to {output_image_filename}")


def save_labeled_point_clouds(point_cloud, labels, input_file):
    """Saves points with label 0 and label 1 as separate point cloud files."""
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    dir_name, file_name = os.path.split(input_file)
    file_base, file_ext = os.path.splitext(file_name)

    for label in [0, 1]:
        mask = labels == label
        if np.any(mask):  # Check if there are any points with this label
            pcd_label = o3d.geometry.PointCloud()
            pcd_label.points = o3d.utility.Vector3dVector(points[mask])
            pcd_label.colors = o3d.utility.Vector3dVector(colors[mask])
            output_filename = os.path.join(dir_name, f"cluster_{file_base}_label_{label}{file_ext}")
            o3d.io.write_point_cloud(output_filename, pcd_label)
            print(f"Label {label} point cloud saved to {output_filename}")


def process_point_cloud_folder(folder_path):
    """Processes all .ply files in the given folder."""
    print(f"-> Processing folder: {folder_path}")

    ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply') and not f.startswith('cluster')]

    for ply_file in ply_files:
        file_path = os.path.join(folder_path, ply_file)
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                print(f"Skipping {file_path}: Point cloud is empty.")
                continue

            labels = cluster_point_cloud(pcd, n_clusters=2)
            visualize_clustered_point_cloud(pcd, labels, file_path)
            save_labeled_point_clouds(pcd, labels, file_path)

        except Exception as e:
            print(f"Skipping {file_path}: {e}")  # Simplified error message

    print(f"-> Saving completed for {folder_path}.")

def csf_ply(file_path, outfile_path_ground, outfile_path_non_ground, cloth_resolution=0.1, rigidness=3, time_step=0.65,
             class_threshold=0.03, iterations=500, visualize=False, prefix="CSF"):
    """Classifies ground and non-ground points in a PLY file using the Cloth Simulation Filter (CSF)."""
    start_time = time.time()
    print(f"[{prefix}] Starting CSF processing...")

    try:
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"[{prefix}] Successfully read PLY file: {file_path}")
    except FileNotFoundError:
        print(f"[{prefix}] Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"[{prefix}] Error reading PLY file: {e}")
        return None

    xyz = np.asarray(pcd.points)
    print(f"[{prefix}] Point cloud loaded: {xyz.shape[0]} points")

    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = time_step
    csf.params.class_threshold = class_threshold
    csf.params.iterations = iterations

    csf.setPointCloud(xyz)

    ground_indices = CSF.VecInt()
    non_ground_indices = CSF.VecInt()

    print(f"[{prefix}] Starting filtering process...")
    csf.do_filtering(ground_indices, non_ground_indices)
    print(f"[{prefix}] Filtering complete.")

    ground_points = xyz[ground_indices]
    non_ground_points = xyz[non_ground_indices]

    print(f"[{prefix}] Ground points extracted: {ground_points.shape[0]} points")
    print(f"[{prefix}] Non-ground points extracted: {non_ground_points.shape[0]} points")

    def save_pcd(points, file_path, color):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
        o3d.io.write_point_cloud(file_path, pcd)
        print(f"[{prefix}] Point cloud saved to: {file_path}")
        return pcd

    try:
        ground_pcd = save_pcd(ground_points, outfile_path_ground, [1, 0, 0])  # Red
        non_ground_pcd = save_pcd(non_ground_points, outfile_path_non_ground, [0, 1, 0])  # Green

    except Exception as e:
        print(f"[{prefix}] Error writing output PLY file: {e}")
        return None

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[{prefix}] Processing finished in {elapsed_time:.2f} seconds.")

    if visualize:
        o3d.visualization.draw_geometries([ground_pcd, non_ground_pcd])
    return ground_pcd, non_ground_pcd