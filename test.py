import pandas as pd
import numpy as np
diamonds = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv")

# print(diamonds.head())
# diamonds.drop(['carat', 'color', 'y'], axis=1, inplace=True)
# print(diamonds.head())

# diamonds.drop([2, 4, 5], axis=0, inplace=True)
# print(diamonds.head())

# sort the cut series
# result = diamonds.cut.sort_values(ascending=True)
# print(result.head())

# result = diamonds.price.sort_values(ascending=False)
# print(result)

# result = diamonds.sort_values('carat')
# result = diamonds.sort_values('carat', ascending=False)
# print(result)

# filter the dataframe rows to only how carat weight at least 0.3
# booleans = []
# for w in diamonds.carat:
#     if w >= .3:
#         booleans.append(True)
#     else:
#         booleans.append(False)

# print(booleans[0:20])


# convert a python list to a pandas series
l = [1, 3, 5, 7, 9, 11]
# print(l)

result = pd.Series(l, dtype='float32')
# print(result)

# result = diamonds[(diamonds.x > 5) & (diamonds.y > 5) & (diamonds.z > 5)]
# print(result)

# result = diamonds[(diamonds.cut == 'Premium') | (diamonds.cut == 'Ideal')]
# print(result)


# find the diamonds that are with a fair or good or premium
# result = diamonds[diamonds.cut.isin(['Fair', 'Good', 'Premium'])]
# print(result)

# columns list
# columns = diamonds.columns
# print(columns)

# read only a subset of 3 rows
# result = pd.read_csv("http://bit.ly/uforeports", nrows=3)
# print(result)

# iterate through diamonds
# for index, row in diamonds.iterrows():
#     print(index, row.carat, row.cut, row.color, row.price)


# add dtypes in the dataset
# print(diamonds.dtypes)

# include only numeric columns in the dataset
# result = diamonds.select_dtypes(include=[np.number]).dtypes
# print(result)

# pass a list of data types to only describe certain types of diamonds
# result = diamonds.describe(include=['object', 'float64'])
# print(result)

# caclulate the mean
# result = diamonds.head()
# print(result)

# calculate the mean of each row of diamonds
# result = diamonds.mean(axis= 1).head()
# print(result)

# mean price for each cut of diamonds
# result = diamonds.groupby('cut').price.mean()
# print(result)


# mean price for each carat of diamonds
# result = diamonds.groupby('carat').price.mean()
# print(result)

# mean price for each color of diamond
# result = diamonds.groupby('color').price.mean()
# print(result)

# mean price for each clarity of diamond
# result = diamonds.groupby('clarity').price.mean()
# print(result)


# calculate count, min, max price for each cut of diamonds
# result = diamonds.groupby('cut').price.agg(['count', 'min', 'max'])
# print(result)

# create side by side bar plot of the diamonds
# result = diamonds.groupby('cut').mean().plot(kind='bar')
# print(result)


# how many times each value in cut series of diamonds
# result = diamonds.cut.value_counts()
# print(result)

# display percentages of each value of cut series occurs in diamonds
# result = diamonds.cut.value_counts(normalize=True)
# print(result * 100)


# display the unique values in cut
# result = diamonds.cut.unique()
# print(result)

# display the unqiue values in carats
# result = diamonds.carat.unique()
# print(result)


# count the number of unique values in cut
# result = diamonds.carat.nunique()
# print(result)

# count the number of unique values in price
# result = diamonds.price.nunique()
# print(result)


# compute a cross tabulation of two series
# result = pd.crosstab(diamonds.cut, diamonds.clarity)
# print(result)


# various summary statistics of cut series of diamonds
result = diamonds.carat.describe()
print(result)
