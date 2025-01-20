#!/usr/bin/env python3
import json
import os
from datetime import datetime
import numpy as np
from PIL import Image
import meshio
import logging
from minio import Minio
from storage import (
    build_filename,
    check_file_exists_in_minio,
    upload_file_to_minio,
    BUCKETS,
)
from rich.logging import RichHandler
from PIL import Image
from scipy.fft import fft2, fftshift
from skimage.filters import sobel

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)


def collect_and_upload_metadata(
    noise_name: str,
    noise_params: dict,
    map_size: int,
    client: Minio,
    displacement_obj: str,
    stl_obj: str,
    msh_obj: str,
    displacement_path: str,
    stl_path: str,
    msh_path: str,
    generation_start_time: float,
    hmm_start_time: float,
    ftetwild_start_time: float,
    hmm_triangles: int,
    hmm_error: float,
    hmm_z_exaggeration: float,
    hmm_z_value: float,
    hmm_base: float,
    ftetwild_bin: str,
    rebuild_metadata: bool,
):
    """
    Collect and upload metadata about the generated displacement map, STL, and mesh (.msh)
    to a MinIO bucket as a JSON file.
    """
    try:
        # Define metadata filename and object name
        metadata_filename = build_filename(noise_name, noise_params, map_size, "json")
        metadata_obj = f"{noise_name}/{metadata_filename}"
        metadata_path = os.path.join(os.path.dirname(msh_path), metadata_filename)

        # Check if metadata exists unless rebuilding
        if not rebuild_metadata and check_file_exists_in_minio(
            client, BUCKETS.METADATA, metadata_obj
        ):
            logger.debug(
                f"Metadata {metadata_obj} already exists in MinIO. Skipping metadata generation."
            )
            return

        # Calculate generation times
        generation_end_time = datetime.now().timestamp()
        total_generation_time = generation_end_time - generation_start_time
        hmm_time = ftetwild_start_time - hmm_start_time
        ftetwild_time = generation_end_time - ftetwild_start_time

        # Load displacement map for roughness calculations
        disp_image = Image.open(displacement_path)
        displacement_array = np.array(disp_image)
        height, width = displacement_array.shape

        # Calculate surface roughness parameters
        roughness_params = calculate_surface_roughness_parameters(displacement_array)
        advanced_roughness = calculate_advanced_surface_roughness(displacement_array)

        # Calculate additional displacement map statistics
        displacement_stats = calculate_displacement_stats(displacement_array)

        # Extract mesh statistics (common for both STL and tetrahedral meshes)
        mesh_stats = extract_mesh_statistics(msh_path)

        # Compute STL-specific statistics
        stl_mesh = meshio.read(stl_path)
        stl_normals = compute_triangle_normals(stl_mesh)
        stl_unique_normals_count = count_unique_normals(stl_normals)
        stl_top_face_stats = calculate_top_face_stats(stl_mesh)

        # Get file sizes
        displacement_size = get_file_size(displacement_path)
        stl_size = get_file_size(stl_path)
        msh_size = get_file_size(msh_path)

        # Compile metadata with additional statistics
        metadata = {
            "noise_name": noise_name,
            "noise_params": noise_params,
            "map_size": map_size,
            "displacement_object": displacement_obj,
            "stl_object": stl_obj,
            "msh_object": msh_obj,
            "timestamp": datetime.now().isoformat(),
            "generation_time_seconds": {
                "total": total_generation_time,
                "hmm": hmm_time,
                "ftetwild": ftetwild_time,
            },
            "hmm_statistics": {
                "triangles": hmm_triangles,
                "base": hmm_base,
                "error": hmm_error,
                "z_value": hmm_z_value,
                "z_exaggeration": hmm_z_exaggeration,
            },
            "surface_roughness": {
                "Ra": roughness_params["Ra"],
                "Rq": roughness_params["Rq"],
                "Rz": roughness_params["Rz"],
                "Rv": roughness_params["Rv"],
                "Rp": roughness_params["Rp"],
                "Rsk": advanced_roughness["Rsk"],
                "Rku": advanced_roughness["Rku"],
                "Rtm": advanced_roughness["Rtm"],
                "Rda": advanced_roughness["Rda"],
                "Rdq": advanced_roughness["Rdq"],
            },
            "displacement_map": {
                "original_height": disp_image.height,
                "original_width": disp_image.width,
                "stats": displacement_stats,
            },
            "stl_statistics": {
                "vertices": mesh_stats["vertices"],
                "faces": mesh_stats["faces"],
                "cell_counts": mesh_stats["cell_counts"],
                "surface_area_top": mesh_stats["surface_area_top"],
                "volume": mesh_stats["volume"],
                "height": height,
                "width": width,
                "top_point": mesh_stats["top_point"],
                "bottom_point": mesh_stats["bottom_point"],
                "unique_normals_count": stl_unique_normals_count,
                **stl_top_face_stats,
            },
            "tet_statistics": {
                "vertices": mesh_stats["vertices"],
                "faces": mesh_stats["faces"],
                "cell_counts": mesh_stats["cell_counts"],
                "surface_area_top": mesh_stats["surface_area_top"],
                "volume": mesh_stats["volume"],
                "height": height,
                "width": width,
                "top_point": mesh_stats["top_point"],
                "bottom_point": mesh_stats["bottom_point"],
            },
            "file_sizes_bytes": {
                "displacement_png": displacement_size,
                "stl_file": stl_size,
                "msh_file": msh_size,
            },
            "ftetwild_bin": ftetwild_bin,
        }

        # Save metadata to JSON file
        with open(metadata_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)
        logger.debug(f"Generated metadata JSON: {metadata_path}")

        # Upload metadata JSON to MinIO
        logger.debug("[MinIO] Uploading Metadata JSON...")
        upload_file_to_minio(
            client=client,
            bucket_name="metadata",
            file_path=metadata_path,
            object_name=metadata_obj,
            overwrite=True,
        )

    except Exception as e:
        logger.error(
            f"Failed to collect or upload metadata for {noise_name}, size {map_size}: {e}"
        )
        raise


def calculate_surface_roughness_parameters(displacement_array: np.ndarray) -> dict:
    """
    Calculate basic surface roughness parameters (Ra, Rq, Rz, etc.) for the displacement map.
    """
    try:
        Z_mean = np.mean(displacement_array)
        Ra = np.mean(np.abs(displacement_array - Z_mean))
        Rq = np.sqrt(np.mean((displacement_array - Z_mean) ** 2))
        Rz = np.max(displacement_array) - np.min(displacement_array)
        Rv = np.abs(np.min(displacement_array))
        Rp = np.max(displacement_array)
        return {
            "Ra": float(Ra),
            "Rq": float(Rq),
            "Rz": float(Rz),
            "Rv": float(Rv),
            "Rp": float(Rp),
        }
    except Exception as e:
        logger.error(f"Failed to calculate surface roughness parameters: {e}")
        raise


def calculate_advanced_surface_roughness(displacement_array: np.ndarray) -> dict:
    """
    Calculate advanced roughness parameters, including skewness (Rsk),
    kurtosis (Rku), total peak-to-valley (Rtm), and simplified slope
    parameters (Rda, Rdq).
    """
    try:
        Z_mean = np.mean(displacement_array)
        Rq = np.sqrt(np.mean((displacement_array - Z_mean) ** 2))

        # Skewness
        Rsk = (np.mean((displacement_array - Z_mean) ** 3)) / (
            Rq**3 if Rq != 0 else 1e-9
        )

        # Kurtosis
        Rku = (np.mean((displacement_array - Z_mean) ** 4)) / (
            Rq**4 if Rq != 0 else 1e-9
        )

        # Rtm
        Rtm = np.max(displacement_array) - np.min(displacement_array)

        # Approximate slope (Rda, Rdq) via gradient
        gradients = np.gradient(displacement_array)
        # If displacement_array is 2D, gradients is a tuple of (gy, gx).
        # Combine them to get a slope magnitude at each pixel.
        if isinstance(gradients, (list, tuple)) and len(gradients) == 2:
            slope = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
        else:
            # If 1D, just use gradients as is
            slope = np.abs(gradients)

        Rda = np.mean(slope)
        Rdq = np.sqrt(np.mean(slope**2))

        return {
            "Rsk": float(Rsk),
            "Rku": float(Rku),
            "Rtm": float(Rtm),
            "Rda": float(Rda),
            "Rdq": float(Rdq),
        }
    except Exception as e:
        logger.error(f"Failed to calculate advanced surface roughness parameters: {e}")
        raise


def analyze_frequency(image: np.ndarray) -> dict:
    """
    Perform Fourier analysis on the image and compute frequency domain statistics.
    Returns a dictionary of frequency stats.
    """
    # 2D FFT
    frequency_data = fftshift(fft2(image))
    magnitude_spectrum = np.abs(frequency_data)
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)

    mean_frequency = np.mean(magnitude_spectrum)
    std_frequency = np.std(magnitude_spectrum)
    max_frequency = np.max(magnitude_spectrum)
    min_frequency = np.min(magnitude_spectrum)
    # location of the dominant frequency
    dominant_frequency = np.unravel_index(
        np.argmax(magnitude_spectrum), magnitude_spectrum.shape
    )

    frequency_stats = {
        "mean_frequency": float(mean_frequency),
        "std_frequency": float(std_frequency),
        "max_frequency": float(max_frequency),
        "min_frequency": float(min_frequency),
        "dominant_frequency": dominant_frequency,
        "log_magnitude_spectrum": log_magnitude_spectrum.tolist(),
    }
    return frequency_stats


def detect_edges(image: np.ndarray) -> tuple:
    """
    Detect edges in the image using the Sobel filter.
    Returns the edge image and a dictionary of edge statistics.
    """
    edges = sobel(image)
    edge_stats = {
        "edge_mean": float(np.mean(edges)),
        "edge_std": float(np.std(edges)),
        "edge_max": float(np.max(edges)),
        "edge_min": float(np.min(edges)),
        "edge_count": int(np.sum(edges > 0)),
    }
    return edges, edge_stats


def calculate_displacement_stats(displacement_array: np.ndarray) -> dict:
    """
    Calculate various statistics for a displacement map, including:
      - Mean, Std Dev, Min, Max, Histogram, Percentiles
      - Frequency domain statistics (via Fourier analysis)
      - Edge detection statistics (via Sobel filter)
    """
    try:
        # Basic stats
        hist, bin_edges = np.histogram(displacement_array, bins=256, range=(0, 255))
        mean_val = float(np.mean(displacement_array))
        std_val = float(np.std(displacement_array))
        min_val = float(np.min(displacement_array))
        max_val = float(np.max(displacement_array))

        # Frequency analysis
        frequency_stats = analyze_frequency(displacement_array)

        # Edge detection
        edges, edge_stats = detect_edges(displacement_array)

        # Combine into one dictionary
        stats = {
            "mean": mean_val,
            "std_dev": std_val,
            "min": min_val,
            "max": max_val,
            "percentiles": {
                "25th": float(np.percentile(displacement_array, 25)),
                "50th": float(np.percentile(displacement_array, 50)),  # median
                "75th": float(np.percentile(displacement_array, 75)),
            },
            "histogram": hist.tolist(),
            "histogram_bin_edges": bin_edges.tolist(),
            "frequency_stats": frequency_stats,
            "edge_stats": edge_stats,
        }
        return stats

    except Exception as e:
        print(f"Failed to calculate displacement stats: {e}")
        raise


def extract_mesh_statistics(msh_file_path: str) -> dict:
    """
    Extract overall mesh statistics from a given .msh or .stl file using meshio.
    This includes vertex count, face count, surface area, volume, and top/bottom points.
    """
    try:
        mesh = meshio.read(msh_file_path)
        vertices = mesh.points.shape[0]
        cells = mesh.cells_dict
        num_cells = {cell_type: len(cells[cell_type]) for cell_type in cells}
        surface_area_top = calculate_top_surface_area(mesh)
        volume = calculate_mesh_volume(mesh)

        # Get top and bottom points
        extremes = get_top_bottom_points(mesh)

        return {
            "vertices": vertices,
            "faces": len(cells.get("triangle", [])),
            "cell_counts": num_cells,
            "surface_area_top": surface_area_top,
            "volume": volume,
            "top_point": extremes["top_point"],
            "bottom_point": extremes["bottom_point"],
        }
    except Exception as e:
        logger.error(f"Failed to extract mesh statistics from {msh_file_path}: {e}")
        raise


def get_file_size(file_path: str) -> int:
    """
    Return the file size in bytes for the given file path.
    """
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        raise


def calculate_total_surface_area(mesh: meshio.Mesh) -> float:
    """
    Calculate the total surface area from triangles/quads in a mesh.
    """
    try:
        surface_area = 0.0
        for cell_type in ["triangle", "quad"]:
            if cell_type in mesh.cells_dict:
                for cell in mesh.cells_dict[cell_type]:
                    vertices = mesh.points[cell]
                    area = calculate_polygon_area(vertices)
                    surface_area += area
        return float(surface_area)
    except Exception as e:
        logger.error(f"Failed to calculate total surface area: {e}")
        raise


def calculate_polygon_area(vertices: np.ndarray) -> float:
    """
    Calculate the 2D polygon area (projection onto XY plane) given 2D coordinates in `vertices`.
    If the polygon is in 3D, only X and Y are considered for area.
    """
    try:
        if len(vertices) < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area
    except Exception as e:
        logger.error(f"Failed to calculate polygon area: {e}")
        raise


def calculate_top_surface_area(mesh: meshio.Mesh, threshold: float = 0.95) -> float:
    """
    Calculate the total area of triangles whose normal vector's Z component
    is above `threshold`, implying they are mostly "facing upward."
    """
    try:
        surface_area_top = 0.0
        if "triangle" not in mesh.cells_dict:
            return 0.0
        for cell in mesh.cells_dict["triangle"]:
            vertices = mesh.points[cell]
            normal = calculate_normal(vertices)
            if normal[2] >= threshold:
                area = calculate_polygon_area(vertices)
                surface_area_top += area
        return float(surface_area_top)
    except Exception as e:
        logger.error(f"Failed to calculate top surface area: {e}")
        raise


def calculate_normal(vertices: np.ndarray) -> np.ndarray:
    """
    Calculate a unit normal vector for a triangle defined by 3 points in 3D.
    """
    try:
        vec1 = vertices[1] - vertices[0]
        vec2 = vertices[2] - vertices[0]
        normal = np.cross(vec1, vec2)
        norm = np.linalg.norm(normal)
        if norm == 0:
            return np.array([0.0, 0.0, 0.0])
        return normal / norm
    except Exception as e:
        logger.error(f"Failed to calculate normal vector: {e}")
        raise


def calculate_mesh_volume(mesh: meshio.Mesh) -> float:
    """
    Calculate the total volume of all tetrahedrons in a mesh (if any).
    If there are no tetrahedrons, returns 0.0.
    """
    try:
        if "tetra" not in mesh.cells_dict:
            return 0.0
        volume = 0.0
        for cell in mesh.cells_dict["tetra"]:
            tetra_points = mesh.points[cell]
            v = calculate_tetrahedron_volume(tetra_points)
            volume += v
        return float(volume)
    except Exception as e:
        logger.error(f"Failed to calculate mesh volume: {e}")
        raise


def calculate_tetrahedron_volume(tetra_points: np.ndarray) -> float:
    """
    Calculate the volume of a single tetrahedron (4 points in 3D).
    """
    try:
        v1 = tetra_points[1] - tetra_points[0]
        v2 = tetra_points[2] - tetra_points[0]
        v3 = tetra_points[3] - tetra_points[0]
        volume = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        return volume
    except Exception as e:
        logger.error(f"Failed to calculate tetrahedron volume: {e}")
        raise


def get_top_bottom_points(mesh: meshio.Mesh) -> dict:
    """
    Find the top (max Z) and bottom (min Z) points in the mesh.
    """
    points = mesh.points
    top_index = np.argmax(points[:, 2])
    bottom_index = np.argmin(points[:, 2])
    return {
        "top_point": points[top_index].tolist(),
        "bottom_point": points[bottom_index].tolist(),
    }


def compute_triangle_normals(mesh: meshio.Mesh) -> np.ndarray:
    """
    Compute normal vectors for each triangle in the mesh.
    """
    if "triangle" not in mesh.cells_dict:
        return np.array([])
    normals = []
    for tri in mesh.cells_dict["triangle"]:
        vertices = mesh.points[tri]
        normal = calculate_normal(vertices)
        normals.append(normal)
    return np.array(normals)


def count_unique_normals(normals: np.ndarray) -> int:
    """
    Count the number of unique normal vectors (rounded to 6 decimal places).
    """
    if normals.size == 0:
        return 0
    rounded_normals = np.round(normals, decimals=6)
    unique_normals = np.unique(rounded_normals, axis=0)
    return int(len(unique_normals))


def calculate_top_face_stats(mesh: meshio.Mesh, threshold: float = 0.95) -> dict:
    """
    Calculate statistics for triangles whose normal vector's Z component is
    above `threshold`, indicating top-facing triangles.
    """
    top_faces_count = 0
    total_top_area = 0.0
    normals_list = []

    if "triangle" not in mesh.cells_dict:
        return {
            "top_faces_count": 0,
            "total_top_area": 0.0,
            "average_top_area": 0.0,
            "mean_top_normal": [0.0, 0.0, 0.0],
        }

    for cell in mesh.cells_dict["triangle"]:
        vertices = mesh.points[cell]
        normal = calculate_normal(vertices)
        if normal[2] >= threshold:
            area = calculate_polygon_area(vertices)
            total_top_area += area
            top_faces_count += 1
            normals_list.append(normal)

    average_top_area = total_top_area / top_faces_count if top_faces_count else 0.0
    mean_top_normal = (
        np.mean(normals_list, axis=0).tolist() if normals_list else [0.0, 0.0, 0.0]
    )

    return {
        "top_faces_count": top_faces_count,
        "total_top_area": float(total_top_area),
        "average_top_area": float(average_top_area),
        "mean_top_normal": mean_top_normal,
    }


def extract_stl_statistics(stl_file_path: str) -> dict:
    """
    Extract statistics specifically for an STL mesh,
    reusing existing routines (e.g. extract_mesh_statistics).
    """
    meshio.read(stl_file_path)  # Just to confirm it's readable
    stats = extract_mesh_statistics(stl_file_path)
    # Optionally add top-facing details
    stl_mesh = meshio.read(stl_file_path)
    top_face_stats = calculate_top_face_stats(stl_mesh)
    stats.update(top_face_stats)
    return stats
