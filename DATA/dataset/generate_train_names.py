import os

# Generate image list from labels folder
files = os.listdir('./labels/')
files.sort()

with open("./names.txt", "a") as output:
    for file in files:
        if file.endswith('txt'):
            image_file_name = './images/' + file[:-4] + '.jpg'
            output.write(image_file_name + '\n')