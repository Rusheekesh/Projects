##importing the essesential libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
##loading the data
df = pd.read_csv("car_evaluation.csv")
##converting the string values to numerical values from the feature columns
### converting the buying_price - low:0,med:1,high:2,vhigh:3 ###
n = {"low":0,"med":1,"high":2,"vhigh":3}
df["buying_price"] = df["buying_price"].map(n)
### converting the maintenance_cost - low:0,med:1,high:2,vhigh:3 ###
n = {"low":0,"med":1,"high":2,"vhigh":3}
df["maintenance_cost"] = df["maintenance_cost"].map(n)
### converting the number_of_doors - 5more:6 ###
df.number_of_doors[df.number_of_doors == "5more" ] = 6
### converting the number_of_passengers - more:6 ###
df.number_of_passengers[df.number_of_passengers == "more" ] = 6
### converting the luggage_capacity - small:0,med:1,big:2 ###
n = {"small":0,"med":1,"big":2}
df["luggage_capacity"] = df["luggage_capacity"].map(n)
### converting safety - low:0,med:1,high:2 ###
n = {"low":0,"med":1,"high":2}
df["safety"] = df["safety"].map(n)
##Assigning the features and target
X = df.loc[:,"buying_price":"safety"]
Y = df.loc[:,"values"]
##Spliting the avilable data into training and testing phase
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8)
##fitting the training data into an algorithm
tree = DecisionTreeClassifier().fit(X_train,Y_train)
##creating the predicting variable
Y_predict = tree.predict(X_test)
##calculating the accuracy of the model for taining and testing data
print(tree.score(X_train,Y_train)) #1.0(must be 100% for training data)
print(tree.score(X_test,Y_test))   #0.9884(quite the good accuracy)
##predict the output for condition-
predict = tree.predict([[0,1,2,6,2,2]])
print(predict)




