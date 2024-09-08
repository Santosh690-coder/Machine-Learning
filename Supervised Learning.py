import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df=pd.read_csv('iris.csv')

x=df[['sepal_length','sepal_width', 'peta_length', 'petal_width']]
y=df['species']

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)

model=SVC()
model.fit(X_train,y_train)

accuracy=model.score(X_test, y_test)

print('Test accuracy: ',accuracy)