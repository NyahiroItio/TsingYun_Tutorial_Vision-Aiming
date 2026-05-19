import json
from pathlib import Path

import cv2
import numpy as np

TASK_ROOT = Path(__file__).resolve().parents[1]

# TODO(student): fill in your camera parameter file from calibrate.py.
# run calibrate.py first and point this path at the generated camera_params.json
# the JSON must contain camera_matrix and dist_coeffs
CAMERA_PARAMS_PATH = TASK_ROOT / "output" / "camera_params.json"

# TODO(student): fill in your own ArUco video path.
# use a video where the marker is clear and not too motion-blurred
# keep the marker dictionary and physical length below consistent with this video
ARUCO_VIDEO_PATH = TASK_ROOT / "data" / "aruco" / "aruco.mp4"

# TODO(student): fill in your own ArUco settings.
# choose the same dictionary that was used to print the marker
# measure the black marker side length in meters and store it in MARKER_LENGTH_METERS
ARUCO_DICTIONARY = "DICT_4X4_50"
MARKER_LENGTH_METERS = 0.076

ARUCO_OUTPUT_VIDEO_PATH = TASK_ROOT / "output" / "aruco_result.mp4"

# TODO(student): use this path if you want to render one of the provided OBJ models.
# start with cube.obj while debugging because its shape makes pose errors obvious
# after the pose is stable, switch to another OBJ model from res/models
MODEL_PATH = TASK_ROOT / "res" / "models" / "african_head.obj"
OUTPUT_DIR = TASK_ROOT / "output"


def load_camera_params(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
    return camera_matrix, dist_coeffs


def load_obj(model_path):
    """Load the vertices and faces from a small OBJ model."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for raw_line in Path(model_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("v "):
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            continue

        if line.startswith("f "):
            parts = line.split()[1:]
            if len(parts) < 3:
                continue

            indices: list[int] = []
            for token in parts:
                vertex_text = token.split("/")[0]
                if not vertex_text:
                    continue
                obj_index = int(vertex_text)
                if obj_index > 0:
                    indices.append(obj_index - 1)
                else:
                    indices.append(len(vertices) + obj_index)

            if len(indices) < 3:
                continue

            for i in range(1, len(indices) - 1):
                faces.append((indices[0], indices[i], indices[i + 1]))

    return vertices, faces


def get_aruco_dictionary(name):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_markers(frame, dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
    return corners, ids


def create_marker_object_points(marker_length_meters):
    half = marker_length_meters * 0.5
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def _is_valid_pose_result(result):
    if result is None:
        return False
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    rvec, tvec = result
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)
    return rvec.size == 3 and tvec.size == 3 and np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))


def estimate_marker_pose(marker_corners, marker_length_meters, camera_matrix, dist_coeffs):
    object_points = create_marker_object_points(marker_length_meters)

    # TODO(student): Estimate one marker pose with OpenCV solvePnP.
    # Input: detected 2D marker corners, marker size, camera_matrix, and dist_coeffs.
    # Output: rvec and tvec.
    # `object_points` has already been prepared for you.
    success, rotate_vec, transmit_vec = cv2.solvePnP(
        object_points, marker_corners.reshape(4, 2), camera_matrix, dist_coeffs
    )
    if not success:
        return None, None
    return rotate_vec, transmit_vec
    raise NotImplementedError("estimate_marker_pose is not implemented")


def render_virtual_object(frame, rvec, tvec, camera_matrix, dist_coeffs, vertices, faces):
    # TODO(student): Render the loaded OBJ model on top of the ArUco marker.
    # Input geometry from load_obj(...):
    #   vertices: list of 3D points, equivalent to an array with shape (N, 3).
    #   faces: list of triangle vertex indices, equivalent to an array with shape (M, 3).
    # Output: the rendered frame.
    #
    # 1. Convert vertices and faces to numpy arrays if needed.
    # 2. Normalize / scale / translate the model to fit above the marker.
    # 3. Use cv2.projectPoints(...) to project 3D vertices to 2D image points.
    # 4. For each face, collect its three projected 2D vertices.
    # 5. Draw the triangle edges or filled triangle on frame.
    #
    # Model size normalization can be tricky at first; we recommend asking AI for help.
    pts_3d = np.array(vertices, dtype=np.float32)

    # --- 1. 自动缩放 ---
    # 计算模型在三个轴上的跨度
    min_pt = np.min(pts_3d, axis=0)
    max_pt = np.max(pts_3d, axis=0)
    size = max_pt - min_pt
    max_dimension = np.max(size)
    
    # 缩放到 marker 长度的一半左右 (或者你觉得合适的比例)
    scale = MARKER_LENGTH_METERS / max_dimension
    pts_3d *= scale

    # --- 2. 居中与平移 ---
    # 让模型的中心在 (X=0, Y=0) 上，底部在 Z=0 上（让它站在纸上）
    # 注意：OBJ 里的坐标轴可能和 OpenCV 坐标系不一致，可能需要尝试交换轴
    # 假设 africanhead 的 Y 是高度轴 (Up-axis)
    pts_3d[:, 0] -= (np.min(pts_3d[:, 0]) + np.max(pts_3d[:, 0])) / 2  # X居中
    pts_3d[:, 1] -= (np.min(pts_3d[:, 1]) + np.max(pts_3d[:, 1])) / 2  # Y居中
    pts_3d[:, 2] -= np.min(pts_3d[:, 2])  # Z底部对齐 Z=0

    # --- 3. 投影 ---
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    # --- 4. 绘制 ---
    # africanhead 面数很多，全画 polylines 可能会卡顿，可以每隔几帧画一次，或者只画一部分
    # 也可以使用 cv2.fillPoly 来增加质感
    for face in faces:
        poly = np.array([pts_2d[face[0]], pts_2d[face[1]], pts_2d[face[2]]])
        cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=1)

    return frame
    raise NotImplementedError("render_virtual_object is not implemented")


def process_frame(frame, dictionary, camera_matrix, dist_coeffs, vertices, faces):
    output = frame.copy()
    corners, ids = detect_markers(frame, dictionary)

    if ids is None or len(ids) == 0:
        return output

    cv2.aruco.drawDetectedMarkers(output, corners, ids)

    for marker_corners, _ in zip(corners, ids):
        rvec, tvec = estimate_marker_pose(
            marker_corners,
            MARKER_LENGTH_METERS,
            camera_matrix,
            dist_coeffs,
        )
        render_virtual_object(
            output,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
            vertices,
            faces,
        )

    return output


def run_aruco_render(dictionary, camera_matrix, dist_coeffs, capture, vertices, faces):
    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Cannot read the first frame from the source.")

    height, width = frame.shape[:2]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(ARUCO_OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (width, height),
    )

    try:
        while ok:
            result = process_frame(frame, dictionary, camera_matrix, dist_coeffs, vertices, faces)
            writer.write(result)
            cv2.imshow("aruco", result)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            ok, frame = capture.read()
    finally:
        writer.release()
        capture.release()
        cv2.destroyAllWindows()

    print(f"Saved video result to: {ARUCO_OUTPUT_VIDEO_PATH}")


def main():
    if not CAMERA_PARAMS_PATH.exists():
        raise SystemExit(f"Camera parameters not found: {CAMERA_PARAMS_PATH}")

    camera_matrix, dist_coeffs = load_camera_params(CAMERA_PARAMS_PATH)
    vertices, faces = load_obj(MODEL_PATH)
    dictionary = get_aruco_dictionary(ARUCO_DICTIONARY)
    capture = cv2.VideoCapture(str(ARUCO_VIDEO_PATH))

    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {ARUCO_VIDEO_PATH}")

    run_aruco_render(dictionary, camera_matrix, dist_coeffs, capture, vertices, faces)


if __name__ == "__main__":
    main()
