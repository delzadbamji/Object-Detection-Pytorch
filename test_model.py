import numpy as np
import pandas as pd

from functions import overlapScore
import cv2
import torch

from cnn_model import cnn_model



# implement your code here
t=input("enter t data")
t = np.asarray(pd.read_csv(t))
gt=input("enter gt data")
pdgt = np.asarray(pd.read_csv(gt))
model = cnn_model()
model.eval()
path=input("enter path for the model")
model.load_state_dict(torch.load(path))
lenTest = len(t)
t = np.reshape(t,(lenTest,1,100,100))
t=torch.Tensor(t)
output = model(t)
output = output.detach().numpy()
output = output.astype(int)
avgScore,scores = overlapScore(output,pdgt)
avgScore = avgScore/lenTest
print("Overlap: ",avgScore)
np.savetxt('test-result.csv')
torch.save(model.state_dict(),'test.pth')
img=np.zeros((512,512,3))
x1 = int(pdgt[0][0])
y1 = int(pdgt[0][1])
w1 = int(pdgt[0][2])
h1 = int(pdgt[0][3])
cv2.rectangle(img, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
cv2.imshow("okay prediction", img)
cv2.waitKey(0)

