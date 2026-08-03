import math


def calculate_fire(u, v):
    """
    Calculate flame position from pixel coordinates.

    Args:
        u: x-coordinate of the point (pixels)
        v: y-coordinate of the point (pixels)

    Returns:
        d: Longitudinal distance (meters)
        l: Lateral distance (meters)
    """
    # Camera intrinsics
    fx = 3.88789992e+03
    fy = 3.90103551e+03
    u0 = 1.29660540e+03
    v0 = 7.29269604e+02

    # Camera mounting parameters
    h = 5.384  # Camera height (meters)
    pitch_angle = math.radians(5.56)  # Pitch angle (radians)
    yaw_angle = math.radians(14.47)  # Yaw angle (radians)

    # Compute distances based on pixel position
    if u >= u0:
        if v >= v0:
            d1 = h / math.tan(pitch_angle + math.atan((v - v0) / fy))
            l1 = abs(u - u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(fy ** 2 + (v - v0) ** 2)
            d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw_angle - math.atan(l1 / d1))
            l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw_angle - math.atan(l1 / d1))
        else:
            d1 = h / math.tan(pitch_angle - math.atan((v0 - v) / fy))
            l1 = abs(u - u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(fy ** 2 + (v0 - v) ** 2)
            d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw_angle - math.atan(l1 / d1))
            l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw_angle - math.atan(l1 / d1))
    else:
        if v >= v0:
            d1 = h / math.tan(pitch_angle + math.atan((v - v0) / fy))
            l1 = abs(u - u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(fy ** 2 + (v - v0) ** 2)
            d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw_angle + math.atan(l1 / d1))
            l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw_angle + math.atan(l1 / d1))
        else:
            d1 = h / math.tan(pitch_angle - math.atan((v0 - v) / fy))
            l1 = abs(u - u0) * math.sqrt(d1 ** 2 + h ** 2) / math.sqrt(fy ** 2 + (v0 - v) ** 2)
            d = (d1 / math.cos(math.atan(l1 / d1))) * math.cos(yaw_angle + math.atan(l1 / d1))
            l = (d1 / math.cos(math.atan(l1 / d1))) * math.sin(yaw_angle + math.atan(l1 / d1))

    return d, l


# 16 test points from flame localization experiment
points = [
    ("A", 2210, 1163),
    ("B", 1825, 1117),
    ("C", 1417, 1066),
    ("D", 1919, 977),
    ("E", 1582, 939),
    ("F", 2021, 817),
    ("G", 1895, 803),
    ("H", 2070, 740),
    ("I", 1929, 726),
    ("J", 2024, 636),
    ("K", 1942, 628),
    ("L", 2093, 621),
    ("M", 2158, 603),
    ("N", 2046, 593),
    ("O", 2179, 571),
    ("P", 2060, 561),
]

print("=" * 60)
print(f"{'Point':<6} {'u':<8} {'v':<8} {'Longitudinal (m)':<18} {'Lateral (m)':<15}")
print("=" * 60)

for name, u, v in points:
    d, l = calculate_fire(u, v)
    print(f"{name:<6} {u:<8} {v:<8} {d:<18.2f} {l:<15.2f}")

print("=" * 60)