\# Annotation Guidelines for DL-YOLO Dataset

This document describes the annotation protocol used for the tunnel fire dataset in our paper.

\---


\## Classes

| Class ID | Name | Description |

|----------|------|-------------|

| 0 | flame | Visible flame region from fire sources |

| 1 | smoke | Smoke plumes (white or black) in tunnel environments |

\---

\## Flame Annotation

\### Reference Point

Use the bottom center of the bounding box as the reference point for flame localization. The bottom center corresponds to the flame root on the tunnel road surface.

\### Bounding Box Rules

\- The bounding box should tightly enclose the visible flame region.

\- Include the entire flame area, including any protruding flames.

\- Do not include reflected light or glare from tunnel walls.

\### Examples

\- Small flame: Use a tight box around the flame core.

\- Large flame: Extend the box to cover the full flame height, with the bottom edge at the flame root.

\- Multiple flames: Annotate each visible flame separately.

\---

\## Smoke Annotation

\### Reference Point

Use the upper edge of the bounding box intersecting the tunnel vault reference line. The vault reference line is a virtual line along the tunnel crown.

\### Bounding Box Rules

\- The bounding box should cover the visible smoke region.

\- For low-contrast smoke, include the faint diffusion area.

\- Do not include background structures (tunnel walls, lights).

\### Smoke Types

\- White smoke: Annotate the main smoke plume.

\- Black smoke: Annotate the visible smoke region, including dense areas.

\- Thin/diffuse smoke: Extend the box to cover the visible diffusion boundary.

\---

\## General Rules

\### Tool

Use LabelImg for annotation. Export in YOLO format (.txt files).

\### File Format

Each annotation file should be a .txt file with the same name as the corresponding image. Each line contains:

class\_id x\_center y\_center width height

Where:

\- class\_id: 0 for flame, 1 for smoke

\- x\_center, y\_center: Center of bounding box (normalized to \[0, 1])

\- width, height: Bounding box size (normalized to \[0, 1])

Coordinates are normalized by dividing by image width and height.

\### Quality Control

\- Two independent annotators per image.

\- Inter-annotator agreement: IoU > 0.7.

\- Disagreements resolved by a third reviewer.

\### Exclusions

\- Do not annotate flame-like reflections from vehicle lights or tunnel walls.

\- Do not annotate smoke-like patterns from dust or steam.

\---

\## Folder Structure

DATA/dataset/

├── images/

│   ├── img\_001.jpg

│   └── ...

└── labels/

&#x20;   ├── img\_001.txt

&#x20;   └── ...

\---

\## References

\- LabelImg: https://github.com/tzutalin/labelImg

\- YOLO format: https://docs.ultralytics.com/datasets/detect/

