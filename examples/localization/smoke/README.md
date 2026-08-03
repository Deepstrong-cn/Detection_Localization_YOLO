# Smoke Front Localization Example

This script demonstrates monocular geometric localization for smoke backflow fronts using the proposed DL-YOLO method.

---

## Overview

The script performs the following steps:

1. **Object detection**: YOLO detects smoke in the original (distorted) image
2. **Image undistortion**: The entire image is corrected using camera calibration parameters
3. **Coordinate mapping**: Detection boxes are mapped from the distorted image to the undistorted image
4. **Vault reference line**: A virtual line along the tunnel crown is drawn from two reference points
5. **Smoke front localization**: The intersection of the upper edge (lower boundary) of the smoke bounding box with the vault reference line is calculated
6. **Visualization**: Results are drawn on the undistorted image

---

## Test Point Example

The following example corresponds to **Test Point C and F** from the smoke front localization experiment:
![NVR_ch2_main_20250322233000_20250323000000.dav_20250403_125226.992.jpg](NVR_ch2_main_20250322233000_20250323000000.dav_20250403_125226.992.jpg)
![smoke_localization_result_C.jpg](smoke_localization_result_C.jpg)
---
![NVR_ch2_main_20250322233000_20250323000000.dav_20250403_125247.026.jpg](NVR_ch2_main_20250322233000_20250323000000.dav_20250403_125247.026.jpg)
![smoke_localization_result_F.jpg](smoke_localization_result_F.jpg)
---
| Test Point | Intersection (u, v) | Ground Truth (m) | Estimated (m) | Error (m) |
|------------|---------------------|------------------|---------------|-----------|
| **C** | (1394, 445) | 60.00 | **59.63** | **-0.37** |
| **F** | (1308, 407) | 45.00 | **44.78** | **-0.22** |


**Vault Reference Line Points:**
- Vault Ref 1: `(1452, 470)`
- Vault Ref 2: `(868, 214)`

---

## Formula Reference

The localization follows the geometric model described in the paper (Section 3.2.2). For smoke fronts under the tunnel vault:

**Step 1: Compute intersection point**

The smoke front is defined as the intersection of:

- The vault reference line: $y = m \cdot x + b$
- The lower boundary of the upper edge of the smoke bounding box: $y = y_1 + w$

where $w$ is the detection box line thickness (1 pixel).

**Step 2: Compute intermediate distance**

$$d_y = \frac{H - h}{\tan\left(\arctan\left(\frac{|v - v_0|}{f_y}\right) - \alpha\right)}$$

**Step 3: Compute intermediate lateral offset**

$$d_x = \frac{|u - u_0| \cdot \sqrt{d_y^2 + (H - h)^2}}{\sqrt{f_y^2 + (v - v_0)^2}}$$

**Step 4: Compute final distances**

$$d_l = \frac{d_y}{\cos\left(\arctan\left(\frac{d_x}{d_y}\right)\right)} \cdot \cos\left(\beta \pm \arctan\left(\frac{d_x}{d_y}\right)\right)$$

$$d_t = \frac{d_y}{\cos\left(\arctan\left(\frac{d_x}{d_y}\right)\right)} \cdot \sin\left(\beta \pm \arctan\left(\frac{d_x}{d_y}\right)\right)$$

Where:
- $d_l$: Longitudinal distance (meters) - **final output**
- $d_t$: Lateral distance (meters) - **final output**
- $H$: Tunnel height (meters)
- $h$: Camera installation height (meters)
- $\alpha$: Pitch angle (radians)
- $\beta$: Yaw angle (radians)
- $(u, v)$: Intersection point coordinates (pixels)
- $(u_0, v_0)$: Principal point
- $f_y$: Focal length in pixels
- $\pm$ depends on the quadrant of the pixel relative to the principal point

---

## Usage

### Prerequisites

- Python 3.8+
- Required packages: `ultralytics`, `opencv-python`, `numpy`