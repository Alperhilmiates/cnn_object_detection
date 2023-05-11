# %%
import numpy as np
import os
from ultralytics import YOLO
import torch
import cv2
import pandas as pd
import psycopg2

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
# % matplotlib inline
import seaborn as sns
from datetime import datetime

# %%
path_of_images = './train_yolo8/reguncelfotograflar_2/'
path_of_tiles = './train_yolo8/tiled_images/'

# %%
list_of_images = os.listdir(path_of_images)

# %% Load a model
model = YOLO("yolov8_segmentasyon_v3_28_april.pt")  # load our custom trained model

# %% Create a folder for each image
for images in list_of_images:
    print(images)
    try:
        if not os.path.exists(path_of_tiles + images):
            deneme = images.split(".")[0]
            os.makedirs(path_of_tiles + deneme)
    except:
        print("Unable to create " + images + " folder")


# %%

class Preprocess:
    def __init__(self, path_of_images, path_of_tiles):
        self.path_of_images = path_of_images
        self.path_of_tiles = path_of_tiles

    def list_images(self):
        self.list_of_images = os.listdir(self.path_of_images)
        return self.list_of_images

    def load_model(self):
        model = YOLO("yolov8_segmentasyon_v3_28_april.pt")  # load our custom trained model
        return model

    def create_folders(self):
        for images in self.list_of_images:
            try:
                if not os.path.exists(self.path_of_tiles + images):
                    deneme = images.split(".")[0]
                    os.makedirs(self.path_of_tiles + deneme)
            except:
                print("Unable to create " + images + " folder")


# %%

class PrepYolo:
    def __init__(self, path_of_images, path_of_tiles):
        self.path_of_images = path_of_images
        self.path_of_tiles = path_of_tiles
        self.tile_size = (512, 512)

    def re_image_yellow(self):
        # Load the image
        for images in list_of_images:
            img = cv2.imread(path_of_images + images)
            # Define the size of the tiles 256x256 to many tiles so 512x512 preferred

            image_name_ext = images.split(".")[0]
            # Loop through the image in tile_size increments
            try:
                for x in range(0, img.shape[1], self.tile_size[0]):
                    for y in range(0, img.shape[0], self.tile_size[1]):

                        # Crop the tile from the image
                        tile = img[y:y + self.tile_size[1], x:x + self.tile_size[0]]

                        # Convert the tile to HSV color space
                        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

                        # Define the yellow color range
                        lower_yellow = np.array([20, 100, 100])
                        upper_yellow = np.array([30, 255, 255])

                        # Threshold the image to only include yellow pixels
                        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

                        # Check if the tile contains any yellow pixels
                        if np.sum(mask_yellow) > 0:
                            # Save the tile as a new image
                            directory_same = path_of_tiles + image_name_ext
                            cv2.imwrite(f'{directory_same}/tile_{image_name_ext}_{x}_{y}.jpg', tile)

            except:
                print("Unable to load " + images + " file")

    def re_image_white(self):
        # Load the image
        for images in list_of_images:
            img = cv2.imread(path_of_images + images)
            # Define the size of the tiles 256x256 to many tiles so 512x512 preferred

            image_name_ext = images.split(".")[0]
            # Loop through the image in tile_size increments
            try:
                for x in range(0, img.shape[1], self.tile_size[0]):
                    for y in range(0, img.shape[0], self.tile_size[1]):

                        # Crop the tile from the image
                        tile = img[y:y + self.tile_size[1], x:x + self.tile_size[0]]

                        # Convert the tile to HSV color space
                        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)

                        # Define the white color range
                        lower_white = np.array([0, 0, 200])
                        upper_white = np.array([255, 50, 255])

                        # Threshold the image to only include white pixels
                        mask_white = cv2.inRange(hsv, lower_white, upper_white)

                        # Check if the tile contains any yellow pixels
                        if np.sum(mask_white) > 0:
                            # Save the tile as a new image
                            directory_same = path_of_tiles + image_name_ext
                            cv2.imwrite(f'{directory_same}/tile_{image_name_ext}_{x}_{y}.jpg', tile)
            except:
                print("Unable to load " + images + " file")

            # %%


# %% Create a class for prediction

# Define path to folder containing JPG files
jpg_folder_main = './tiled_images/test1/'


class Prediction:
    def __init__(self, jpg_folder):
        # Define classes dictionary
        self.classes = {0: 'galeri_sinegi', 1: 'beyaz_sinek', 2: 'tuta', 3: 'thrips'}
        # Initialize class count dictionary
        self.class_counts = {self.classes[i]: 0 for i in range(len(self.classes))}
        # Get list of JPG files in folder
        self.jpg_files = [f for f in os.listdir(jpg_folder) if f.endswith('.jpg')]

    def load_model(self, jpg_folder):
        for jpgs in self.jpg_files:
            deneme = model(f"{jpg_folder}/{jpgs}", conf=0.3)
            # save the result
            boxes = deneme[0].boxes.xyxy
            confs = deneme[0].boxes.conf

            # Iterate over boxes and count classes
            for i in range(len(boxes)):
                if confs[i] > 0.3:
                    cls = int(deneme[0].boxes.cls[i])
                    self.class_counts[self.classes[cls]] += 1
        return self.class_counts


# %%

test1 = Preprocess(path_of_images, path_of_tiles)
list_of_images = test1.list_images()
model = test1.load_model()
test1.create_folders()

# %%
test2 = PrepYolo(path_of_images, path_of_tiles)
test2.re_image_yellow()

# %%

test3 = Prediction(jpg_folder_main)
class_counts = test3.load_model(jpg_folder_main)

# %%
# Print class counts
print(class_counts)

# %%
conn = psycopg2.connect(
    dbname="dbname",
    user="user",
    password="password",
    host="localhost",
    port="5432"
)

df_pest = pd.read_sql_query("SELECT * FROM public.pests", conn)

conn.close()
# %%

print(df_pest)

# %%

# Define input variables
sector = 'hydro'
date_today = datetime.today().strftime('%Y-%m-%d')
print(date_today)
# %%
# manual input for date
date_today = '2023-03-10'


# %%

class update_database:
    def __init__(self, class_counts, sector, date_today):
        self.conn = psycopg2.connect(
            dbname="dbname",
            user="user",
            password="password",
            host="localhost",
            port="5432"
        )
        self.cur = self.conn.cursor()
        self.sec = sector
        self.class_counts = class_counts
        self.date_today = date_today
        self.total_pest_num = sum(self.class_counts.values())
        self.classes = ['galeri_sinegi', 'beyaz_sinek', 'tuta', 'thrips']

    def update(self):
        self.cur.execute(
            "INSERT INTO a_hydro1.pests (galeri_sinegi, beyaz_sinek, tuta, thrips, total_pest_num, sector, date_1) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (self.class_counts['galeri_sinegi'], self.class_counts['beyaz_sinek'], self.class_counts['tuta'],
             self.class_counts['thrips'], self.total_pest_num, self.sec, self.date_today))
        self.conn.commit()
        self.cur.close()
        self.conn.close()


# %%
update_database(class_counts, sector, date_today).update()

# %%
sector = "sector_1"
test4 = update_database(class_counts, sector)
test4.update()

# %%

# Define classes dictionary
classes = {0: 'galeri_sinegi', 1: 'beyaz_sinek', 2: 'tuta', 3: 'thrips'}

# Initialize class count dictionary
class_counts_v2 = {classes[i]: 0 for i in range(len(classes))}

# Define folder path
jpg_folder = './tiled_images/test2/'

# Iterate over JPG files in folder
class yolo_predict_output_images:
    def __init__(self, jpg_folder):
        for jpg_file in os.listdir(jpg_folder):
            if jpg_file.endswith('.jpg'):
                # Run YOLO detection on file
                !yolo detect predict task=segment model=yolov8_segmentasyon_v3_28_april.pt source={jpg_folder}{jpg_file} conf=0.3
                # Get results if they exist
                if 'result' in globals():
                    results = globals()['result']

                    # Iterate over boxes and count classes
                    for result in results:
                        # Get boxes and confidence values
                        boxes = result[0].boxes.xyxy
                        confs = result[0].boxes.conf

                        # Iterate over boxes and count classes
                        for i in range(len(boxes)):
                            if confs[i] > 0.3:
                                cls = int(result[0].boxes.cls[i])
                                class_counts_v2[classes[cls]] += 1
# %%
# Print class counts
print(class_counts_v2)

# %% Masks
# !yolo detect predict task=segment model=yolov8_segmentasyon_v2_27_april.pt source= "studio_project.mp4" conf=0.3
# %%
