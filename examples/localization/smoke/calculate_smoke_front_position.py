import os
import math
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
from ultralytics import YOLO


class SmokeFrontCalculator:
    def __init__(self):
        # Camera intrinsics
        self.fx = 6.94528802e+03
        self.fy = 6.94103996e+03
        self.u0 = 1.34412610e+03
        self.v0 = 7.59670553e+02

        # Tunnel and camera parameters
        self.H = 7.3  # Tunnel height (meters)
        self.h = 6.315  # Camera height (meters)
        self.pitch = math.radians(1.65)  # Pitch angle
        self.yaw = math.radians(2.5)  # Yaw angle

        # Detection box line thickness (pixels)
        self.line_width = 1

    def calculate_vault_line(self, x1, y1, x2, y2):
        """Calculate vault reference line from two points."""
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    def calculate_intersection(self, m, b, y_c):
        """Calculate intersection x-coordinate at given y."""
        x = (y_c - b) / m
        return x

    def locate_smoke_front(self, bbox, vault_x1, vault_y1, vault_x2, vault_y2):
        """
        Locate smoke backflow front.
        """
        x1, y1, x2, y2 = bbox

        y_upper_lower = y1 + self.line_width

        m, b = self.calculate_vault_line(vault_x1, vault_y1, vault_x2, vault_y2)
        u = self.calculate_intersection(m, b, y_upper_lower)
        v = y_upper_lower

        # Round to integer pixel coordinates
        u_rounded = int(round(u))
        v_rounded = int(round(v))

        # Compute distances using rounded integer coordinates
        if u_rounded >= self.u0:
            d1 = (self.H - self.h) / math.tan(math.atan(abs(v_rounded - self.v0) / self.fy) - self.pitch)
            l1 = abs(u_rounded - self.u0) * math.sqrt(d1 ** 2 + (self.H - self.h) ** 2) / math.sqrt(
                self.fy ** 2 + (v_rounded - self.v0) ** 2)
            d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(self.yaw - math.atan(l1 / d1))
            l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(self.yaw - math.atan(l1 / d1))
        else:
            d1 = (self.H - self.h) / math.tan(math.atan(abs(v_rounded - self.v0) / self.fy) - self.pitch)
            l1 = abs(u_rounded - self.u0) * math.sqrt(d1 ** 2 + (self.H - self.h) ** 2) / math.sqrt(
                self.fy ** 2 + (v_rounded - self.v0) ** 2)
            d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(self.yaw + math.atan(l1 / d1))
            l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(self.yaw + math.atan(l1 / d1))

        return d, l, u_rounded, v_rounded


def read_camera_matrix(filename):
    """Read camera intrinsic matrix from file."""
    with open(filename, 'r') as file:
        content = file.read().strip()
        content = content[1:-1].strip()
    lines = content.split('\n')
    camera_matrix = []
    for line in lines:
        clean_line = line.strip()[1:-1].strip()
        camera_matrix.extend(list(map(float, clean_line.split())))
    return np.array(camera_matrix, dtype=np.float64).reshape(3, 3)


def read_dist_coeffs(filename):
    """Read distortion coefficients from file."""
    with open(filename, 'r') as file:
        content = file.read().strip()[2:-2]
        dist_coeffs = list(map(float, content.split()))
    return np.array(dist_coeffs, dtype=np.float64)


def draw_visualization(image, bbox, vault_p1, vault_p2, intersection_point, distance_text, line_width=1):
    """
    Draw vault line, bounding box, upper edge, and intersection point on image.
    """
    x1, y1, x2, y2 = bbox
    u_inter, v_inter = intersection_point

    m = (vault_p2[1] - vault_p1[1]) / (vault_p2[0] - vault_p1[0])
    b = vault_p1[1] - m * vault_p1[0]

    h, w = image.shape[:2]
    x_start = 0
    x_end = w
    y_start = int(m * x_start + b)
    y_end = int(m * x_end + b)
    cv2.line(image, (x_start, y_start), (x_end, y_end), (0, 255, 255), line_width)

    cv2.circle(image, vault_p1, 6, (0, 255, 255), -1)
    cv2.circle(image, vault_p2, 6, (0, 255, 255), -1)
    cv2.putText(image, "Vault Ref 1", (vault_p1[0] + 10, vault_p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(image, "Vault Ref 2", (vault_p2[0] + 10, vault_p2[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), line_width)

    y_upper_line = int(y1 + line_width)
    cv2.line(image, (int(x1), y_upper_line), (int(x2), y_upper_line), (255, 255, 0), line_width)
    cv2.putText(image, "Upper Edge", (int(x1), y_upper_line - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    u_px = int(u_inter)
    v_px = int(v_inter)

    if 0 <= u_px < image.shape[1] and 0 <= v_px < image.shape[0]:
        image[v_px, u_px] = (255, 0, 255)
        for du in [-1, 0, 1]:
            if 0 <= u_px + du < image.shape[1]:
                image[v_px, u_px + du] = (255, 0, 255)
        for dv in [-1, 0, 1]:
            if 0 <= v_px + dv < image.shape[0]:
                image[v_px + dv, u_px] = (255, 0, 255)

    for i in range(0, int(u_inter), 15):
        cv2.line(image, (i, int(v_inter)), (min(i + 8, int(u_inter)), int(v_inter)), (255, 255, 0), 1)
    for i in range(int(u_inter), w, 15):
        cv2.line(image, (i, int(v_inter)), (min(i + 8, w), int(v_inter)), (255, 255, 0), 1)

    y_top = int(m * u_inter + b)
    for i in range(int(v_inter), int(y_top) - 1, -10):
        cv2.line(image, (int(u_inter), i), (int(u_inter), max(i - 8, int(y_top))), (255, 0, 255), 1)

    cv2.putText(image, distance_text, (int(u_inter) + 15, int(v_inter) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)

    coord_text = f"({int(u_inter)}, {int(v_inter)})"
    cv2.putText(image, coord_text, (int(u_inter) + 15, int(v_inter) + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths (all relative to script location)
    WEIGHTS = os.path.join(script_dir, 'weights.pt')
    SOURCE = os.path.join(script_dir, 'NVR_ch2_main_20250322233000_20250323000000.dav_20250403_125226.992.jpg')
    OUTPUT_PATH = os.path.join(script_dir, 'smoke_localization_result_C.jpg')
    CAMERA_MATRIX_FILE = os.path.join(script_dir, 'camera_matrix.txt')
    DIST_COEFFS_FILE = os.path.join(script_dir, 'dist_coeffs.txt')

    # Vault reference line points (on the undistorted image)
    VAULT_P1 = (1452, 470)
    VAULT_P2 = (868, 214)

    # Read calibration parameters
    print("📖 Reading calibration parameters...")
    cameraMatrix = read_camera_matrix(CAMERA_MATRIX_FILE)
    distCoeffs = read_dist_coeffs(DIST_COEFFS_FILE)

    # Load original image
    img_original = cv2.imread(SOURCE)
    if img_original is None:
        print(f"Error: Could not load image from {SOURCE}")
        return

    h, w = img_original.shape[:2]
    print(f"✅ Original image size: {w}x{h}")

    # Run object detection on original image
    print("🔍 Running object detection...")
    model = YOLO(WEIGHTS)
    results = model.predict(
        source=SOURCE,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        save=True,
        project='runs/predict_smoke',
    )

    # Undistort the entire image
    print("🔄 Undistorting image...")
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, (w, h), cv2.CV_32FC1)
    img_undistorted = cv2.remap(img_original, mapx, mapy, cv2.INTER_LINEAR)

    # Process detection results and draw on undistorted image
    print("📐 Drawing on undistorted image...")
    calculator = SmokeFrontCalculator()

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())

                if cls == 1:  # smoke
                    print(f"🔍 Detected smoke on original image: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")

                    points = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.float32).reshape(-1, 1, 2)

                    undistorted_points = cv2.undistortPoints(points, cameraMatrix, distCoeffs, P=newCameraMatrix)
                    undistorted_points = undistorted_points.reshape(-1, 2)

                    new_x1 = min(undistorted_points[:, 0])
                    new_y1 = min(undistorted_points[:, 1])
                    new_x2 = max(undistorted_points[:, 0])
                    new_y2 = max(undistorted_points[:, 1])

                    bbox_undistorted = [new_x1, new_y1, new_x2, new_y2]

                    print(f"   Undistorted bbox: ({new_x1:.0f}, {new_y1:.0f}) -> ({new_x2:.0f}, {new_y2:.0f})")

                    d, l, u_inter, v_inter = calculator.locate_smoke_front(
                        bbox_undistorted,
                        VAULT_P1[0], VAULT_P1[1],
                        VAULT_P2[0], VAULT_P2[1]
                    )

                    print(f"✅ Smoke front: Longitudinal={d:.2f}m, Lateral={l:.2f}m")
                    print(f"✅ Intersection: ({u_inter}, {v_inter})")

                    draw_visualization(
                        img_undistorted,
                        bbox_undistorted,
                        VAULT_P1,
                        VAULT_P2,
                        (u_inter, v_inter),
                        f"Smoke Front: {d:.1f}m",
                        line_width=calculator.line_width
                    )

    cv2.imwrite(OUTPUT_PATH, img_undistorted)
    print(f"✅ Result saved to: {OUTPUT_PATH}")

    cv2.imshow("Smoke Front Localization (Undistorted)", img_undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()