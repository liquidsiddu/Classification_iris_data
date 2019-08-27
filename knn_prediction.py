#This Example program is to predict the accuracy by training the data.

#Importing the Packages of SciKit Learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.externals import joblib

#Loading the dataset
iris = load_iris()

X = iris.data
y = iris.target

#Spliting the Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

#print(X_train)
print(len(X_train))

#print(X_test)
print(len(X_test))

#print(y_test)
print(len(y_test))

#print(y_train)
print(len(y_train))

print(X_train.shape)
print(X_test.shape) 

print(y_train.shape)
print(y_test.shape)

#Training the data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#Predictions on Training Data
y_pred = knn.predict(X_test)

#Comparing Actual Responses with Predicted Responses (y_test, y_pred)
print("KNN Model Accuracy: ", metrics.accuracy_score(y_test, y_pred))

#Predictions out of sample data
sample = [[1, 5, 4, 2], [2, 3, 5, 4]]
preds = knn.predict(sample)
pred_spices = [iris.target_names[p] for p in preds]
print("Predictions: ", pred_spices)

#Saving the Model
joblib.dump(knn, 'iris_knn.pkl')
