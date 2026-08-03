Detection\_Localization\_YOLO/

│

├── DATA/

│   ├── yolov8.yaml                    # Dataset configuration

│   │

│   └── dataset/

│       ├── images/                    

│       │   └── \*.jpg

│       │

│       ├── labels/                    

│       │   └── \*.txt

│       │

│       ├── train.txt                  # Training set image list

│       ├── val.txt                    # Validation set image list

│       ├── test.txt                   # Test set image list

│       ├── classes.txt                # Class names

│       └── generate_train_names.py          # Script to generate image lists


Note: The full dataset is not included due to institutional restrictions, but a few image and label samples are provided for reference. To use your own data, place images in DATA/dataset/images/ and labels in DATA/dataset/labels/, then run generate\_names.py to generate the image list. Manually split the list into train.txt, val.txt, and test.txt according to your experimental needs.

