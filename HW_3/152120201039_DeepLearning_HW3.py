import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from google.colab import drive
drive.mount('/content/drive')
     

def convertImages(path, label):
    images = [] #The proccessed images will be stored in here.
    labels = [] #Labels will be stored in here.

    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        if img_path.endswith(".jpg"):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128)) #resimler 128x128 = 16384'e çevrilir, RGB oldugu icin 3'le çarpılır = 49152
            img_vector = img.flatten() #Convert to 1D vector
            images.append(img_vector)
            labels.append(label)

    return np.array(images), np.array(labels)
     

#The functions which have served in the document, in order to avoid overflow np.clip has been used.
def tanh(x):
    return np.tanh(x)

def softplus(x):
    return np.log(1 + np.exp(np.clip(x, -700, 700)))

def mish(x):
    return x * tanh(softplus(x))

def dmish(x):
    omega = np.exp(3*x) + 4*np.exp(2*x) + (6+4*x)*np.exp(x) + 4*(1 + x)
    delta = 1 + pow((np.exp(x) + 1), 2)
    derivative = np.exp(x) * omega / pow(delta, 2)
    return derivative
     

def trainPerceptron(inputs, t, weights, rho, iterNo, mish):
    bias = np.c_[np.ones(inputs.shape[0]), inputs]  # Add bias
    bias, t = shuffle(bias, t, random_state=42)    # Shuffle
    n = bias.shape[1] #n is the # of features of perceptron added with 1

    # training loop for perceptron
    for _ in range(iterNo):
        for i in range(bias.shape[0]):
            prediction = mish(np.dot(bias[i], weights)) #in order to use mish function in document, mish arguments has been added to functions sign
            # Update weights based on the perceptron learning algorithm
            weights += rho * (t[i] - prediction) * bias[i]

    return weights

     

mainFolder = "/content"

#loads train data with using convertImages.
xTrainFlamingo, yTrainFlamingo = convertImages(os.path.join(mainFolder, "train", "flamingo"), label=0)
xTrainPizza, yTrainPizza = convertImages(os.path.join(mainFolder, "train", "pizza"), label=1)

#loads test data  with using convertImages.
xTestFlamingo, yTestFlamingo = convertImages(os.path.join(mainFolder, "test", "flamingo"), label=0)
xTestPizza, yTestPizza = convertImages(os.path.join(mainFolder, "test", "pizza"), label=1)


#the train data which is loaded separetly is being concatenating.
xTrain = np.concatenate((xTrainFlamingo, xTrainPizza), axis=0)
yTrain = np.concatenate((yTrainFlamingo, yTrainPizza), axis=0)

#the test data which is loaded separetly is being concatenating.
xTest = np.concatenate((xTestFlamingo, xTestPizza), axis=0)
yTest = np.concatenate((yTestFlamingo, yTestPizza), axis=0)

# Add bias
xTrainsBias = np.c_[np.ones(xTrain.shape[0]), xTrain]
xTestsBias = np.c_[np.ones(xTest.shape[0]), xTest]

#weights are being created randomly by using np.random
weights = np.random.rand(xTrainsBias.shape[1] +1) #adding 1 to get same dimension

rho = 0.0001
iterNo = 1000

weights = trainPerceptron(xTrainsBias, yTrain, weights, rho, iterNo, mish) # trainPerceptron call

np.save('weights.npy', weights) #save weights
weightsFile = np.load('weights.npy') #load weights


     

def testPerceptron(tempTest, weights, activation_function):
    # Add bias term to the test sample
    test_bias = np.insert(tempTest, 0, 1)

    # Calculate the predicted class (0 or 1) using Mish activation function
    prediction = int(activation_function(np.dot(test_bias, weights)) > 0)

    return prediction

# Test the perceptron and calculate accuracy
correctPredictions = 0
totalSamples = xTestsBias.shape[0]

for i in range(totalSamples):
    prediction = testPerceptron(xTestsBias[i], weightsFile, mish)
    if prediction == yTest[i]:
        correctPredictions += 1

accuracy = correctPredictions / totalSamples
print("Accuracy: {:.2f}%".format(accuracy * 100))
