import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


data = pd.read_csv('housing.csv')

print(data)

solo=data.dropna(inplace =True)
print(solo)
a=data.info()
print(a)
chandh=data.ocean_proximity.value_counts()
print(chandh)
hello=pd.get_dummies(data.ocean_proximity).astype(int)
print(hello)
data =data.join(pd.get_dummies(data.ocean_proximity).astype(int))

b = data.drop(columns=['ocean_proximity'],inplace=True)
print(b)
c=data.hist(figsize =(16,10))
print(c)
 


d=data.corr()
print(d)
plot=plt.figure(figsize=(15,8))
print(plot)

rotten=sns.heatmap(data.corr(),annot=True,cmap="YlGnBu")
print(rotten)

rooms=data['total_rooms']=np.log(data['total_rooms']+1)
bedrooms=data['total_rooms']=np.log(data['total_bedrooms']+1)
population=data['population']-np.log(data['population']+1)
households=data['households']=np.log(data['households']+1)

print(rooms)
print(bedrooms)
print(population)
print(households)

hist=data.hist(figsize=(16,10))

print(hist)

plotting=plt.figure(figsize=(15,8))
measure=sns.scatterplot(x='latitude',y='longitude',data=data,hue='median_house_value')

print(plotting)
print(measure)

ratio=data['bedroom_ratio']=data['total_bedrooms']/data['total_rooms']
rooms=data['household_rooms'] =data['total_rooms']/data['households']

print(ratio)
print(rooms)

figure=plt.figure(figsize=(15,8))

print(figure)

from sklearn.model_selection import train_test_split
X= data.drop(['median_house_value'], axis =1)
y = data['median_house_value']

train=X_train,X_test, y_train, y_test =  train_test_split(X,y, test_size=0.2)
print(train)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

Linear=reg.fit(X_train, y_train)
print(Linear)

accuracy=reg.score(X_test, y_test) ##accuracy check
print(accuracy)



from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier() 
go=model.fit(X_train,y_train)    
print(go)

model=model.score(X_test,y_test)
print(model)

predictions = forest.predict([[-122.23,37.88,41.0,6.781058,4.867534,5.777652,4.844187,8.3252,0,1,0,0,0,0.717813,1.399834]]) #ask it to predict
print(predictions)

import joblib
joblib.dump(forest, 'House-Price-Predictor.joblib')

forest = joblib.load('House-Price-Predictor.joblib')
predictions = forest.predict([[-122.23,37.88,41.0,6.781058,4.867534,5.777652,4.844187,8.3252,0,1,0,0,0,0.717813,1.399834]]) #ask it to predict
print(predictions