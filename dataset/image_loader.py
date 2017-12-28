import cv2
import matplotlib.pyplot as plt


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

    return resized, image, bb

def plot_image(network, image, load_file, normalize=True):
    plt.figure(dpi=250)
    if normalize:
        image = image / 255.0

    predicted = network.predict(image.reshape(1, -1), load_file=load_file)[0]
    plt.imshow(image, cmap="gray")
    predicted = predicted * 48 + 48
    plt.scatter(predicted[::2], predicted[1::2], c="r")
    plt.show()

    return predicted

def plot_original_image(original_image, predicted, bbox):
    plt.figure(dpi=250)
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]

    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.scatter(predicted[::2]*(121/96) + y, predicted[1::2]*(121/96) + x, c="w"
                ,s=2)
    plt.show()

