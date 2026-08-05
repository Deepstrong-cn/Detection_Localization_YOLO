# Flame Localization Example

This script demonstrates monocular geometric localization for flame sources using the proposed DL-YOLO method.

---

## Overview

The script calculates the 3D spatial coordinates (longitudinal and lateral distances) of a flame source from its pixel coordinates in an image. The calculation uses:

![Uploading Fig. 7. Schematic diagram and process of tunnel .png…]()


- **Camera intrinsic parameters** (focal length, principal point)
- **Camera extrinsic parameters** (installation height, pitch angle, yaw angle)

---

## Formula Reference

The localization follows the geometric model described in the paper (Section 3.2.2). For flame sources on the tunnel road surface:

**Step 1: Compute intermediate distance along optical axis**

$$d_y = \frac{h}{\tan(\alpha \pm \beta)}$$

where $\beta = \arctan\left(\frac{|v - v_0|}{f_y}\right)$

**Step 2: Compute intermediate lateral offset**

$$d_x = \frac{(|u - u_0|) \cdot \sqrt{d_y^2 + h^2}}{\sqrt{f_y^2 + (v - v_0)^2}}$$

**Step 3: Compute final longitudinal and lateral distances**

$$d_l = \frac{d_y}{\cos\left(\arctan\left(\frac{d_x}{d_y}\right)\right)} \cdot \cos\left(\beta \pm \arctan\left(\frac{d_x}{d_y}\right)\right)$$

$$d_t = \frac{d_y}{\cos\left(\arctan\left(\frac{d_x}{d_y}\right)\right)} \cdot \sin\left(\beta \pm \arctan\left(\frac{d_x}{d_y}\right)\right)$$

Where:
- $d_l$: Longitudinal distance (meters) - **final output**
- $d_t$: Lateral distance (meters) - **final output**
- $h$: Camera installation height (meters)
- $\alpha$: Pitch angle (radians)
- $\beta$: Yaw angle (radians)
- $(u, v)$: Target pixel coordinates
- $(u_0, v_0)$: Principal point
- $f_y$: Focal length in pixels
- $\pm$ depends on the quadrant of the pixel relative to principal point

---
## Test Point Layout

The 16 test points (A–P) are arranged as shown in the figure below. The layout covers:
- **Longitudinal distances**: 27 m to 100 m from the camera
- **Lateral offsets**: 0 m to 6.6 m from the tunnel centerline
![marked_image_with_points.png](marked_image_with_points.png)
## Usage

### Prerequisites

- Python 3.8+
- No additional packages required (uses only `math` module)

### Run the script

```bash
cd examples/localization/flame
python calculate_flame_position.py

## Expected Output
============================================================
Point  u        v        Longitudinal (m)   Lateral (m)    
============================================================
A      2210     1163     26.25              0.50           
B      1825     1117     27.17              3.17           
C      1417     1066     28.37              6.38           
D      1919     977      33.55              3.12           
E      1582     939      34.97              6.32           
F      2021     817      45.50              3.09           
G      1895     803      46.55              4.64           
H      2070     740      54.74              3.06           
I      1929     726      56.29              5.14           
J      2024     636      74.59              5.06           
K      1942     628      76.35              6.75           
L      2093     621      79.08              4.02           
M      2158     603      85.08              2.97           
N      2046     593      87.97              5.51           
O      2179     571      97.57              2.92           
P      2060     561      101.45             6.01           
============================================================
```
