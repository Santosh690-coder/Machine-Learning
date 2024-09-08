from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=load_iris()
x=iris.data
y=iris.target

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)


logreg=LogisticRegression()
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)

print("X\n",x)
print("\n")
print("Y-\n",y)
print("\n")
print("X train-\n",x_train)
print("\n")

print("Y train-\n",y_train)
print("\n")
print("Y test-\n",y_test)
print("\n")
print("Y pred\n",y_pred)
print("\n")