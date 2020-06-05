import cv2
import os


path = os.getcwd()
url = os.getcwd() + "\ModelBuildingIntro\src\ps-scikit-learn-models\data\dog.jpg"
i = cv2.imread(url)
print(url)
print(i)
