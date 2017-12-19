import cv2
from pathlib import Path
import matplotlib.pyplot as plt


image = cv2.imread("/home/mihael/Documents/9. semestar/VIROKR/Projekt/Detecting-Facial-Features-CNN/dataset/slike/Ivo_Sanader.jpg")
grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
file = Path("/home/mihael/Documents/9. semestar/VIROKR/Projekt/Detecting-Facial-Features-CNN/dataset/haarcascade_frontalface_default.xml")
print(file.exists())
face_cascade = cv2.CascadeClassifier("/home/mihael/Documents/9. semestar/VIROKR/Projekt/Detecting-Facial-Features-CNN/dataset/haarcascade_frontalface_default.xml")

bounding_boxes = face_cascade.detectMultiScale(grayscale_image, 1.25, 6)
print(bounding_boxes[0])
bb = bounding_boxes[0]
x = bb[0]
y = bb[1]
w = bb[2]
h = bb[3]


plt.figure(dpi=250)
# fig, ax = plt.subplots(1)
# ax.imshow(grayscale_image, cmap="gray")
# rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3]
#                          ,linewidth=1,edgecolor='r',facecolor='none')
# ax.add_patch(rect)
plt.imshow(grayscale_image[y:y+h, x:x+w], cmap="gray")
plt.show()

za_treninanje_slika = grayscale_image[y:y+h, x:x+w]
