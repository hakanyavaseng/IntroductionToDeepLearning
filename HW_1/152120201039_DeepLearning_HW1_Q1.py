import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from PIL import Image

#Read original IMG1 and IMG2
oImg1 = cv2.imread("/content/image1.jpg")
oImg2 = cv2.imread("/content/image2.jpg")

#Show original images
#cv2_imshow(oImg1)
#cv2_imshow(oImg2)

#Resize operations
Img1 = cv2.resize(oImg1,(256,256))
Img2 = cv2.resize(oImg2,(256,256))

#Show resized images
#cv2_imshow(Img1)
#cv2_imshow(Img2)

     

#Flips image 1 vertically
Img1 = cv2.flip(Img1,0)
#Flips image 1 horizontally
Img1 = cv2.flip(Img1,1)
#Rotate image 1-2 to left 90 degree
Img1 = cv2.rotate(Img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
Img2 = cv2.rotate(Img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
#Rotate image 1-2 to right 90 degree
Img1 = cv2.rotate(Img1, cv2.ROTATE_90_CLOCKWISE)
Img2 = cv2.rotate(Img2, cv2.ROTATE_90_CLOCKWISE)
     


#Get sizes of images
height1, width1 = Img1.shape[:2]
height2, width2 = Img2.shape[:2]

#Assign new sizes
new_width1 = int(width1 / 2)
new_height1 = int(height1 / 2)
new_width2 = int(width2 / 2)
new_height2 = int(height2 / 2)

#Images before half operations
#cv2_imshow(Img1)
#cv2_imshow(Img2)

#Half the image
halfedImg1 = cv2.resize(Img1,(new_width1, new_height1))
halfedImg2 = cv2.resize(Img2,(new_width2, new_height2))
     

#Crop operation

#Image1
height1, width1 = halfedImg1.shape[:2]
leftHalfedImg1 = halfedImg1[:, :width1//2]

#Image2
height2, width2 = halfedImg2.shape[:2]
rightHalfedImg2 = halfedImg2[:, :width2//2]

#Merge operation
finalImage = np.concatenate((leftHalfedImg1,rightHalfedImg2), axis = 1)
cv2_imshow(finalImage)