#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import tree

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') 
# for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  
# model = tree.DecisionTreeRegressor() for regression

# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)

#Predict Output
predicted= model.predict(x_test)