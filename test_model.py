import numpy as np
import pandas as pd

from functions import overlapScore
import cv2
import torch

from cnn_model import cnn_model


# load test data
# implement your code here
testdata=input("enter testing data")
pdTest = pd.read_csv(testdata,sep=",",header=None)
# pdTest = pd.read_csv('Dataset/testData.csv',sep=",",header=None)
# pdGroundTest = pd.read_csv('Dataset/ground-truth-test.csv',sep=",",header=None)
groundtest=input("enter ground truth for testing")
pdGroundTest = pd.read_csv(groundtest,sep=",",header=None)
pdTest = np.asarray(pdTest)
pdGroundTest = np.asarray(pdGroundTest)

print("please wait....")

# load model.pth and test model

model = cnn_model()
model.eval()
path=input("enter path for the model...or just model.pth")
model.load_state_dict(torch.load(path))
print("please wait...")
# reshape your text data and feed into your model
# implement your code here
lenTest = len(pdTest)
pdTest = np.reshape(pdTest,(lenTest,1,100,100))
pdTest=torch.Tensor(pdTest)
output = model(pdTest)
# use overlapscore function to calculate the average score
# implement your code here
output = output.detach().numpy()
output = output.astype(int)
avgScore,scores = overlapScore(output,pdGroundTest)

avgScore = avgScore/lenTest

print("Overlap Score for model : ",avgScore)

print("please wait for the image to show up....")
# save your output in a csv file in Result directory and draw an example with bounding box
# implement your code here
np.savetxt('Results/test-result.csv',output,delimiter=",")
torch.save(model.state_dict(),'cnn-test.pth')

# bounding box

img=np.zeros((512,512,3))
# ground truth coodinates 0
x1 = int(pdGroundTest[0][0])
y1 = int(pdGroundTest[0][1])
w1 = int(pdGroundTest[0][2])
h1 = int(pdGroundTest[0][3])
# ground truth coodinates 9
x2 = int(pdGroundTest[9][0])
y2 = int(pdGroundTest[9][1])
w2 = int(pdGroundTest[9][2])
h2 = int(pdGroundTest[9][3])

x1_p = int(output[0][0])
y1_p = int(output[0][1])
w1_p = int(output[0][2])
h1_p = int(output[0][3])

x2_p = int(output[9][0])
y2_p = int(output[9][1])
w2_p = int(output[9][2])
h2_p = int(output[9][3])

cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2) #green groundtruth 0
cv2.rectangle(img, (x1_p, y1_p), (x1_p+w1_p, y1_p+h1_p), (0, 0, 255), 2) #red ouptput 0

cv2.imshow("okay prediction", img)
cv2.waitKey(0)
print("image 1 displayed")

img = np.zeros((512, 512, 3))
cv2.rectangle(img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2) #green ground truth
cv2.rectangle(img, (x2_p, y2_p), (x2_p+w2_p, y2_p+h2_p), (0, 0, 255), 2) #green ground truth

cv2.imshow("accurate prediction", img)
cv2.waitKey(0)
print("image 2 displayed")

print("------finished------")