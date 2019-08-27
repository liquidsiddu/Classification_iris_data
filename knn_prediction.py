from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.externals import joblib

iris = load_iris()

X = iris.data
y = iris.target

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

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("KNN Model Accuracy: ", metrics.accuracy_score(y_test, y_pred))

joblib.dump(knn, 'iris_knn.pkl')