import os
import cv2
import numpy as np
from scipy.spatial import distance
from collections import Counter


def process_images(folder):
    cats = {'cellphone': 1, 'flamingo': 2, 'Motorbikes': 3} #Listing categories
    xTrain, yTrain = [], []
    countPerCategory = {} #To check how many files has been count

    for cat in cats.keys():
        path = os.path.join(folder, cat)
        count = 0
        for i, img_name in enumerate(os.listdir(path)):
            image = cv2.imread(os.path.join(path, img_name)) #Reads files like image_0001.jpg and increments
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #changing image to gray format
            resized = cv2.resize(gray, (128, 128)) #resizing
            gradient = np.gradient(resized) #calculating gradient
            vector = np.float16(gradient).flatten()
            xTrain.append(vector)
            yTrain.append(cats[cat])
            count += 1

        countPerCategory[cat] = count

    return np.array(xTrain), np.array(yTrain), countPerCategory

#KNN Function
def KNN(xTrain, yTrain, sample_test, k, labels):
    distances = [distance.euclidean(xTrain[i], sample_test) for i in range(len(xTrain))]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [yTrain[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    predictedLabel = most_common[0][0]
    predictedCategory = [category for category, label in labels.items() if label == predictedLabel][0]
    return predictedLabel, predictedCategory



xTrain, yTrain, trainFileCount = process_images('/content/train')
sample_test, _, testFileCount = process_images('/content/test')

#categories
cats = {'cellphone': 1, 'flamingo': 2, 'Motorbikes': 3}

predictedLabel, predictedCategory = KNN(xTrain, yTrain, sample_test[0], 5, cats)

#Prints out results
print("\nResult:")
print(f"Predicted Label: {predictedLabel}")
print(f"Predicted Category: {predictedCategory}")


#Additional information
print("\nNumber of files in training set:")
for category, count in trainFileCount.items():
    print(f"{category}: {count} files")

print("\nNumber of files in test set:")
for category, count in testFileCount.items():
    print(f"{category}: {count} files")