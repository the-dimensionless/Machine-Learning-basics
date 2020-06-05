# check scikit-image
import cv2
from matplotlib import pyplot as plt

imagePath = "dog.jpg"
image = cv2.imread(imagePath)

plt.imshow(image)
plt.show()
