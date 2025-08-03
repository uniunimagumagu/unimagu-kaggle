import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())
print(train.info())
print(train.describe())
