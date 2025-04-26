# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn tools.  
2. Load the Iris dataset and create a DataFrame with features and target.  
3. Separate features (`x`) and labels (`y`), then split into training and testing sets.  
4. Create and train an SGDClassifier on the training data.  
5. Use the trained model to predict labels on the test data.  
6. Calculate and print the accuracy of the model.  
7. Generate and print the confusion matrix to assess classification performance.  
8. Plot the true vs. predicted labels to visualize prediction distribution.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: V RAKSHITA
RegisterNumber: 212224100049 
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data,columns = data.feature_names)
df['target']=data.target
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
sgd = SGDClassifier(max_iter=1000,tol=1e-3)
sgd.fit(x_train,y_train)
y_pred = sgd.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)
confusion_mat = confusion_matrix(y_test,y_pred)
print("Confusion matrix: ",confusion_mat)
plt.scatter(y_test,y_pred)
```

## Output:

![Screenshot (276)](https://github.com/user-attachments/assets/332e8452-5ebb-466c-97f8-458168efd508)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
