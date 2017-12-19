import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def read_image(image_file, haarcascade_frontalface_file):
    image = cv2.imread(image_file)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(haarcascade_frontalface_file)

    bounding_boxes = face_cascade.detectMultiScale(grayscale_image, 1.25, 6)
    print(bounding_boxes[0])
    bb = bounding_boxes[0]
    x = bb[0]
    y = bb[1]
    w = bb[2]
    h = bb[3]


    # fig, ax = plt.subplots(1)
    # ax.imshow(grayscale_image, cmap="gray")
    # rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3]
    #                          ,linewidth=1,edgecolor='r',facecolor='none')
    # ax.add_patch(rect)

    za_treninanje_slika = grayscale_image[y:y+h, x:x+w]
    resized = cv2.resize(za_treninanje_slika, (96, 96))

    return resized
