# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Import all the required libraries such as PyTorch, Pandas, NumPy, Matplotlib, Scikit-learn, and Seaborn.

### STEP 2:

Load the customer segmentation dataset from the CSV file and remove unnecessary columns such as the customer ID.

### STEP 3:

Handle missing values in the dataset by replacing them with suitable values. Fill missing work experience values with 0 and family size values with the median.

### STEP 4:

Convert all categorical features into numerical values using Label Encoding so that they can be processed by the neural network.

### STEP 5:

Encode the target column (Segmentation) into numerical classes and separate the dataset into input features (X) and target labels (Y).

### STEP 6:

Split the dataset into training and testing sets. Apply feature scaling using StandardScaler to normalize the input data.

### STEP 7:

Convert the training and testing data into PyTorch tensors and create TensorDatasets and DataLoaders for batch processing.

### STEP 8:

Define a feedforward neural network with four fully connected layers and ReLU activation functions.

### STEP 9:

Initialize the model, define the Cross Entropy Loss function, and configure the Adam optimizer.

### STEP 10:

Train the neural network for multiple epochs by performing forward propagation, loss calculation, backpropagation, and weight updates.

### STEP 11:

Use the trained model to predict the segmentation classes for the test dataset.

### STEP 12:

Evaluate the model using Accuracy Score, Confusion Matrix, and Classification Report.

### STEP 13:

Visualize the confusion matrix using a Seaborn heatmap to analyze the classification performance of the model.






## PROGRAM

### Name: A.NABITHRA

### Register Number: 212224230172

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df = pd.read_csv("exp2.csv")
df
df=df.drop(columns=["ID"])
df
df.columns
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
cat_columns=["Gender","Ever_Married","Graduated","Profession","Spending_Score","Var_1"]
for col in cat_columns:
    df[col]=LabelEncoder().fit_transform(df[col])
lbe=LabelEncoder()
df["Segmentation"]=lbe.fit_transform(df["Segmentation"])
df
xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)
tr = TensorDataset(xt,yt)
tst = TensorDataset(xst,yst)
trl = DataLoader(tr,batch_size=16,shuffle=True) 
tstl=DataLoader(tst,batch_size=16)
## batch_size = 16 processes it as 16 rows instead of a huge dataset as a whole
class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,32)
        self.l2 = nn.Linear(32,16)
        self.l3 = nn.Linear(16,8)
        self.l4 = nn.Linear(8,4)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x
model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.0005)
epochs=300
#losses=[]
for i in range(epochs):
    for a,b in trl: #a refers to xt and b refers to yt
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b.long())
        loss.backward()
        op.step()
        
    if(i%10==0):
        print(f"{i}/{epochs}:",loss.item())   
            #losses.append(loss.item())
pre=[]
act=[]
with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)
    pre.extend(predicted.numpy())
    act.extend(yst.numpy())
    print(act,pre)
accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(act,pre,target_names=["A","B","C","D"])
print("Accuracy:\n",accuracy)
print("Confusion Matrix:\n",conf_matrix)
print("Classification Report:\n",cl_report)
import seaborn as sns
x1=["A","B","C","D"]
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap="Blues",xticklabels=x1,yticklabels=x1)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()





```

### Dataset Information
<img width="1305" height="953" alt="image" src="https://github.com/user-attachments/assets/a77194a8-94c7-4607-b7dc-2c5a556b1d8f" />


### OUTPUT

<img width="1637" height="564" alt="image" src="https://github.com/user-attachments/assets/f5fb381a-2b40-4df1-bca7-03dfa0a8568c" />

<img width="998" height="149" alt="image" src="https://github.com/user-attachments/assets/2ffb1ee6-8175-4698-b4d8-aa415544f108" />

<img width="1724" height="607" alt="image" src="https://github.com/user-attachments/assets/78268d99-ab8f-40b1-b80f-077a7789997b" />

<img width="1636" height="587" alt="image" src="https://github.com/user-attachments/assets/87617723-4d38-416c-a45b-da6d915e45dd" />

<img width="1608" height="604" alt="image" src="https://github.com/user-attachments/assets/bd574126-2ad3-4617-9b08-a7819b40a61b" />

<img width="360" height="852" alt="image" src="https://github.com/user-attachments/assets/9b66c1b6-6ae2-4c42-984c-d8f0d184f772" />






## Confusion Matrix

<img width="680" height="563" alt="image" src="https://github.com/user-attachments/assets/6cce4423-698c-4dd4-8b59-5b7aec29237e" />


## Classification Report
<img width="782" height="620" alt="image" src="https://github.com/user-attachments/assets/afffec0e-d640-4b5d-bf66-ed7050921751" />




## RESULT
Therefore, a neural network classification model has been developed for the given dataset.
