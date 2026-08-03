import os
import math
import warnings

warnings.filterwarnings('ignore')

import cv2
import numpy as np
from ultralytics import YOLO


class Localizer:
    """
    Unified localizer for flame and smoke with integrated object detection.
    """

    def __init__(self, weights_path=None):
        if weights_path is not None:
            self.model = YOLO(weights_path)
        else:
            self.model = None

        # ============================================================
        # USER CONFIGURATION - MODIFY ACCORDING TO YOUR SETUP
        # ============================================================

        # Camera intrinsics (shared by flame and smoke)
        self.fx = None
        self.fy = None
        self.u0 = None
        self.v0 = None

        # Camera mounting (shared)
        self.cam_height = None
        self.pitch = None
        self.yaw = None

        # Flame: vertical distance from camera to ground
        self.flame_height = 0.0

        # Smoke: vertical distance from camera to vault
        self.tunnel_height = None
        self.line_width = 1

        # Vault reference line (for smoke only)
        self.vault_p1 = (None, None)
        self.vault_p2 = (None, None)

        # Inference parameters
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.imgsz = 640

    def _compute_distance_smoke(self, u, v):
        """
        Compute smoke distance using camera geometry.
        Matches the standalone smoke script exactly.
        """
        h = self.tunnel_height - self.cam_height
        pitch = math.radians(self.pitch)
        yaw = math.radians(self.yaw)

        if u >= self.u0:
            d1 = h / math.tan(math.atan(abs(v - self.v0) / self.fy) - pitch)
            l1 = abs(u - self.u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(self.fy ** 2 + (v - self.v0) ** 2)
            d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw - math.atan(l1 / d1))
            l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw - math.atan(l1 / d1))
        else:
            d1 = h / math.tan(math.atan(abs(v - self.v0) / self.fy) - pitch)
            l1 = abs(u - self.u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(self.fy ** 2 + (v - self.v0) ** 2)
            d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw + math.atan(l1 / d1))
            l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw + math.atan(l1 / d1))

        return d, l

    def _compute_distance_flame(self, u, v):
        """
        Compute flame distance using camera geometry.
        """
        h = self.cam_height - self.flame_height
        pitch = math.radians(self.pitch)
        yaw = math.radians(self.yaw)

        if u >= self.u0:
            if v >= self.v0:
                d1 = h / math.tan(pitch + math.atan((v - self.v0) / self.fy))
                l1 = abs(u - self.u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(self.fy ** 2 + (v - self.v0) ** 2)
                d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw - math.atan(l1 / d1))
                l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw - math.atan(l1 / d1))
            else:
                d1 = h / math.tan(pitch - math.atan((self.v0 - v) / self.fy))
                l1 = abs(u - self.u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(self.fy ** 2 + (self.v0 - v) ** 2)
                d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw - math.atan(l1 / d1))
                l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw - math.atan(l1 / d1))
        else:
            if v >= self.v0:
                d1 = h / math.tan(pitch + math.atan((v - self.v0) / self.fy))
                l1 = abs(u - self.u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(self.fy ** 2 + (v - self.v0) ** 2)
                d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw + math.atan(l1 / d1))
                l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw + math.atan(l1 / d1))
            else:
                d1 = h / math.tan(pitch - math.atan((self.v0 - v) / self.fy))
                l1 = abs(u - self.u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(self.fy ** 2 + (self.v0 - v) ** 2)
                d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw + math.atan(l1 / d1))
                l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw + math.atan(l1 / d1))

        return d, l

    def calculate_vault_line(self, x1, y1, x2, y2):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

    def calculate_intersection(self, m, b, y_c):
        x = (y_c - b) / m
        return x

    def locate_smoke_front(self, bbox):
        x1, y1, x2, y2 = bbox

        y_upper_lower = y1 + self.line_width

        m, b = self.calculate_vault_line(
            self.vault_p1[0], self.vault_p1[1],
            self.vault_p2[0], self.vault_p2[1]
        )
        u = self.calculate_intersection(m, b, y_upper_lower)
        v = y_upper_lower

        u_rounded = int(round(u))
        v_rounded = int(round(v))

        d, l = self._compute_distance_smoke(u_rounded, v_rounded)

        return d, l, u_rounded, v_rounded

    def locate_flame(self, bbox):
        u = (bbox[0] + bbox[2]) / 2
        v = bbox[3]
        return self._compute_distance_flame(u, v)

    def read_camera_matrix(self, filename):
        with open(filename, 'r') as file:
            content = file.read().strip()
            content = content[1:-1].strip()
        lines = content.split('\n')
        camera_matrix = []
        for line in lines:
            clean_line = line.strip()[1:-1].strip()
            camera_matrix.extend(list(map(float, clean_line.split())))
        return np.array(camera_matrix, dtype=np.float64).reshape(3, 3)

    def read_dist_coeffs(self, filename):
        with open(filename, 'r') as file:
            content = file.read().strip()[2:-2]
            dist_coeffs = list(map(float, content.split()))
        return np.array(dist_coeffs, dtype=np.float64)

    def predict(self, source, save=True, visualize=True, undistort=False,
                camera_matrix_file=None, dist_coeffs_file=None):
        img = cv2.imread(source)
        if img is None:
            print(f"Error: Could not load image from {source}")
            return []

        h, w = img.shape[:2]
        print(f"✅ Image size: {w}x{h}")

        if undistort:
            print("🔄 Undistorting image...")
            cameraMatrix = self.read_camera_matrix(camera_matrix_file)
            distCoeffs = self.read_dist_coeffs(dist_coeffs_file)
            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, (w, h),
                                                     cv2.CV_32FC1)
            img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        else:
            cameraMatrix = None
            newCameraMatrix = None
            distCoeffs = None

        print("🔍 Running object detection...")
        if self.model is None:
            self.model = YOLO('weights/best.pt')

        results = self.model.predict(
            source=source,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save,
            project='runs/localize',
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    if undistort and cameraMatrix is not None:
                        points = np.array([
                            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                        ], dtype=np.float32).reshape(-1, 1, 2)
                        undistorted_points = cv2.undistortPoints(points, cameraMatrix, distCoeffs, P=newCameraMatrix)
                        undistorted_points = undistorted_points.reshape(-1, 2)
                        x1, y1 = min(undistorted_points[:, 0]), min(undistorted_points[:, 1])
                        x2, y2 = max(undistorted_points[:, 0]), max(undistorted_points[:, 1])

                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'class': 'flame' if cls == 0 else 'smoke',
                        'class_id': cls,
                        'confidence': conf
                    }

                    if cls == 0:  # Flame
                        d, l = self.locate_flame([x1, y1, x2, y2])
                        detection['longitudinal'] = d
                        detection['lateral'] = l
                        print(f"✅ flame: Longitudinal={d:.2f}m, Lateral={l:.2f}m")

                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
                        cv2.putText(img, f"Flame: {d:.1f}m", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        u = int((x1 + x2) / 2)
                        v = int(y2)
                        cv2.circle(img, (u, v), 4, (255, 0, 0), -1)

                    elif cls == 1:  # Smoke
                        d, l, u_inter, v_inter = self.locate_smoke_front([x1, y1, x2, y2])
                        detection['longitudinal'] = d
                        detection['lateral'] = l
                        detection['intersection'] = (u_inter, v_inter)
                        print(f"✅ smoke: Longitudinal={d:.2f}m, Lateral={l:.2f}m, Intersection=({u_inter}, {v_inter})")

                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                        cv2.putText(img, f"Smoke: {d:.1f}m", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        m = (self.vault_p2[1] - self.vault_p1[1]) / (self.vault_p2[0] - self.vault_p1[0])
                        b = self.vault_p1[1] - m * self.vault_p1[0]
                        cv2.line(img, (0, int(b)), (img.shape[1], int(m * img.shape[1] + b)), (0, 255, 255), 1)
                        cv2.circle(img, (u_inter, v_inter), 4, (255, 0, 255), -1)
                        cv2.putText(img, f"Front: {d:.1f}m", (u_inter + 10, v_inter - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                        y_upper = int(y1 + self.line_width)
                        cv2.line(img, (int(x1), y_upper), (int(x2), y_upper), (255, 255, 0), 1)

                    detections.append(detection)
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join('runs/localize', f'localization_result_{timestamp}.jpg')
        cv2.imwrite(output_path, img)
        print(f"✅ Result saved to: {output_path}")

        if visualize:
            cv2.imshow("Detection & Localization", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections


if __name__ == '__main__':
    # ============================================================
    # STEP 1: SET PATHS
    # ============================================================

    # Paths to weights, images, and calibration files
    WEIGHTS = 'weights/best.pt'
    SOURCE = 'examples/localization/smoke/sample_image.jpg'
    CAMERA_MATRIX_FILE = 'examples/localization/smoke/camera_matrix.txt'
    DIST_COEFFS_FILE = 'examples/localization/smoke/dist_coeffs.txt'

    # ============================================================
    # STEP 2: INITIALIZE LOCALIZER
    # ============================================================

    localizer = Localizer(WEIGHTS)

    # ============================================================
    # STEP 3: FILL CAMERA PARAMETERS (REQUIRED)
    # ============================================================

    # Camera intrinsics
    localizer.fx = None  # <-- FILL: Focal length x (pixels)
    localizer.fy = None  # <-- FILL: Focal length y (pixels)
    localizer.u0 = None  # <-- FILL: Principal point x (pixels)
    localizer.v0 = None  # <-- FILL: Principal point y (pixels)

    # Camera mounting
    localizer.cam_height = None  # <-- FILL: Camera height (meters)
    localizer.pitch = None  # <-- FILL: Pitch angle (degrees)
    localizer.yaw = None  # <-- FILL: Yaw angle (degrees)

    # Smoke (if using smoke localization)
    localizer.tunnel_height = None  # <-- FILL: Tunnel height (meters)
    localizer.line_width = 1  # <-- OPTIONAL: Bbox line thickness
    localizer.vault_p1 = (None, None)  # <-- FILL: Vault point 1 (u, v)
    localizer.vault_p2 = (None, None)  # <-- FILL: Vault point 2 (u, v)

    # ============================================================
    # STEP 4: RUN LOCALIZATION
    # ============================================================

    print("=" * 60)
    print("RUNNING LOCALIZATION")
    print("=" * 60)

    results = localizer.predict(
        source=SOURCE,
        save=True,
        visualize=False,
        undistort=True,
        camera_matrix_file=CAMERA_MATRIX_FILE,
        dist_coeffs_file=DIST_COEFFS_FILE
    )

    # ============================================================
    # STEP 5: PRINT RESULTS
    # ============================================================

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for det in results:
        if det['class'] == 'flame':
            print(f"\n🔥 FLAME")
            print(f"  Longitudinal: {det['longitudinal']:.2f}m")
            print(f"  Lateral: {det['lateral']:.2f}m")
        elif det['class'] == 'smoke':
            u, v = det.get('intersection', (None, None))
            print(f"\n💨 SMOKE")
            print(f"  Longitudinal: {det['longitudinal']:.2f}m")
            print(f"  Lateral: {det['lateral']:.2f}m")
            print(f"  Intersection: ({u}, {v})")