import json
import os
from datetime import datetime
import numpy as np
from PIL import Image
import meshio
import logging
from minio import Minio
from storage import build_filename, check_file_exists_in_minio, upload_file_to_minio

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
    rebuild_metadata: bool
):
    try:
        # Define metadata filename and object name
        metadata_filename = build_filename(noise_name, noise_params, map_size, "json")
        metadata_obj = f"{noise_name}/{metadata_filename}"
        metadata_path = os.path.join(os.path.dirname(msh_path), metadata_filename)

        # Check if metadata exists unless rebuilding
        if not rebuild_metadata:
            if check_file_exists_in_minio(client, "metadata", metadata_obj):
                logger.info(f"Metadata {metadata_obj} already exists in MinIO. Skipping metadata generation.")
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

        # Calculate surface roughness parameters: Ra, Rq, Rz
        roughness_params = calculate_surface_roughness_parameters(displacement_array)

        # Extract mesh statistics: vertices, faces, cell_counts, surface_area_top, volume
        mesh_stats = extract_mesh_statistics(msh_path)

        # Calculate advanced surface roughness parameters
        advanced_roughness = calculate_advanced_surface_roughness(displacement_array)

        # Get file sizes in bytes
        displacement_size = get_file_size(displacement_path)
        stl_size = get_file_size(stl_path)
        msh_size = get_file_size(msh_path)

        # Compile metadata
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
                "ftetwild": ftetwild_time
            },
            "hmm_statistics": {
                "triangles": hmm_triangles,
                "base": hmm_base,
                "error": hmm_error,
                "z_value": hmm_z_value,
                "z_exaggeration": hmm_z_exaggeration
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
                "Rdq": advanced_roughness["Rdq"]
            },
            "displacement_map": {
                "original_height": disp_image.height,
                "original_width": disp_image.width
            },
            "mesh_statistics": {
                "vertices": mesh_stats["vertices"],
                "faces": mesh_stats["faces"],
                "cell_counts": mesh_stats["cell_counts"],
                "surface_area_top": mesh_stats["surface_area_top"],
                "volume": mesh_stats["volume"],
                "height": height,
                "width": width
            },
            "file_sizes_bytes": {
                "displacement_png": displacement_size,
                "stl_file": stl_size,
                "msh_file": msh_size
            },
            "ftetwild_bin": ftetwild_bin
        }

        # Save metadata to JSON file
        with open(metadata_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        logger.info(f"Generated metadata JSON: {metadata_path}")

        # Upload metadata JSON to MinIO
        logger.info("[MinIO] Uploading Metadata JSON...")
        upload_file_to_minio(
            client=client,
            bucket_name="metadata",
            file_path=metadata_path,
            object_name=metadata_obj,
            overwrite=True  # Always overwrite metadata
        )

    except Exception as e:
        logger.error(f"Failed to collect or upload metadata for {noise_name}, size {map_size}: {e}")
        raise

def calculate_surface_roughness_parameters(displacement_array: np.ndarray) -> dict:
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
            "Rp": float(Rp)
        }
    except Exception as e:
        logger.error(f"Failed to calculate surface roughness parameters: {e}")
        raise

def calculate_advanced_surface_roughness(displacement_array: np.ndarray) -> dict:
    try:
        Z_mean = np.mean(displacement_array)
        Rq = np.sqrt(np.mean((displacement_array - Z_mean) ** 2))

        # Skewness
        Rsk = (np.mean((displacement_array - Z_mean) ** 3)) / (Rq ** 3)

        # Kurtosis
        Rku = (np.mean((displacement_array - Z_mean) ** 4)) / (Rq ** 4)

        # Rtm
        Rtm = np.max(displacement_array) - np.min(displacement_array)

        # Rda and Rdq (simplified for profile slopes)
        gradients = np.gradient(displacement_array)
        slope = np.sqrt(gradients[0] ** 2 + gradients[1] ** 2)
        Rda = np.mean(slope)
        Rdq = np.sqrt(np.mean(slope ** 2))

        return {
            "Rsk": float(Rsk),
            "Rku": float(Rku),
            "Rtm": float(Rtm),
            "Rda": float(Rda),
            "Rdq": float(Rdq)
        }
    except Exception as e:
        logger.error(f"Failed to calculate advanced surface roughness parameters: {e}")
        raise

def extract_mesh_statistics(msh_file_path: str) -> dict:
    try:
        mesh = meshio.read(msh_file_path)
        vertices = mesh.points.shape[0]
        cells = mesh.cells_dict
        num_cells = {cell_type: len(cells[cell_type]) for cell_type in cells}
        total_surface_area = calculate_total_surface_area(mesh)
        surface_area_top = calculate_top_surface_area(mesh)
        volume = calculate_mesh_volume(mesh)
        return {
            "vertices": vertices,
            "faces": len(cells.get("triangle", [])),
            "cell_counts": num_cells,
            "surface_area_top": total_surface_area,
            "volume": volume
        }
    except Exception as e:
        logger.error(f"Failed to extract mesh statistics from {msh_file_path}: {e}")
        raise

def get_file_size(file_path: str) -> int:
    try:
        return os.path.getsize(file_path)
    except Exception as e:
        logger.error(f"Failed to get file size for {file_path}: {e}")
        raise

def calculate_total_surface_area(mesh: meshio.Mesh) -> float:
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

def calculate_top_surface_area(mesh: meshio.Mesh) -> float:
    try:
        surface_area_top = 0.0
        threshold = 0.95
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
    try:
        v1 = tetra_points[1] - tetra_points[0]
        v2 = tetra_points[2] - tetra_points[0]
        v3 = tetra_points[3] - tetra_points[0]
        volume = np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        return volume
    except Exception as e:
        logger.error(f"Failed to calculate tetrahedron volume: {e}")
        raise
