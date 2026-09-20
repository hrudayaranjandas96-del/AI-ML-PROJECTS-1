import pandas as pd
from sklearn.linear_model import LinearRegression

data = {'Hours': [1,2,3,4,5,6,7,8,9,10], 'Marks': [20,35,45,50,60,65,75,80,88,95]}
df = pd.DataFrame(data)

model = LinearRegression()
model.fit(df[['Hours']], df['Marks'])

hours = 6.5
pred = model.predict([[hours]])
print(f"Predicted Marks for {hours} hours: {pred[0]:.2f}")
