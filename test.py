import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# result = diamonds.carat.describe()
# print(result)

# histogram of the carat series of diamonds
# result = diamonds.carat.plot(kind='hist')
# print(result)

# create a df of booleans from diamonds
# result = diamonds.isnull().head(20)
# print(result)


# count the number of missing values in each series
# result = diamonds.isnull().sum()
# print(result)


# check the number of rows and columns and drop those row if 'any' values are missing in a row of diamonds
# result = diamonds.dropna(how='any').shape
# print(result)

# drop a row if any or all values in a row are missing of diamonds
# result = diamonds.dropna(subset=['carat', 'cut'], how='any').shape
# print(result)


# drop a row if all values in a row are missing of diamonds
# result = diamonds.dropna(subset=['carat', 'cut'], how='any').shape
# print(result)


# set an existing column as the index of diamonds
# result = diamonds.set_index('color', inplace=True)
# print(diamonds.head())

# set an existing column as the index of diamonds dataframe and restore the index name,
# and move the index back to a column
# result = diamonds.set_index('color', inplace=True)
# # print(diamonds.head())
# diamonds.index.name = 'color2'
# diamonds.reset_index(inplace=True)
# print(diamonds.head())


# access a specified series index and the series values of diamonds
# result = diamonds.carat.value_counts().index
# diamonds.carat.value_counts().values
# print(result)

# diamonds.cut.value_counts().sort_values()
# print(diamonds)


# calculate the multiply of length, width and depth for each cut of diamonds
# result = ((diamonds.x * diamonds.y * diamonds.z)).head()
# print(result)


# concatenate the diamonds dataframe with the color series
# result = pd.concat([diamonds, diamonds.color], axis= 1).head()
# print(result)

# read specified rows and all columns of diamonds
# print(diamonds.head(7))

# real all columns
# print(diamonds.loc[0, :])

# read rows 0, 5, 7 and all columns of diamonds
# print(diamonds.loc[[0, 1, 2], :])

# read rows 2 through 5 and all columns of diamonds
# print(diamonds.loc[0:2, :])

# read rows 0 through 2 (inclusive), columns 'color' and 'price' of diamonds
# print(diamonds.loc[0: 2, ['color', 'price']])

# read rows 0 through 2 (inclusive) , columns color through price of diamonds
# result = diamonds.loc[0: 2, 'color': 'price']
# print(result)

# read rows in which the cut is premium, column , color of diamonds
# result = diamonds.loc[diamonds.cut == 'Premium', 'color']
# print(result)

# read rows in positions 0 and 1, columns in positions 0 and 3 of diamonds
# result = diamonds.iloc[[0, 1], [0, 3]]
# print(result)

# read rows in positions 0 through 4, columns in positions 1 through 4 of diamonds
# result = diamonds.iloc[0: 4, 1: 4]
# print(result)

# read rows in positions 0 through 4 (exlusive) and all columns of diamonds
# result = diamonds.iloc[0: 5, :]
# print(result)

# read rows 2 through 5, columns in positions 0 through 2
# result = diamonds.iloc[2: 5, 0: 2]
# print(result)

# getting the summary
# print(diamonds.info)

# get the true memory usage by diamonds dataframe
# print(diamonds.info(memory_usage='deep'))

# calculate the memory usage for each series
# print(diamonds.memory_usage(deep=True))

# get randomly sample rows from diamonds dataframe
# print(diamonds.sample(5))


# get sample 75% of the diamonds df rows without replacement and store the remaining 25% of rows in
# another dataframe
# result = diamonds.sample(frac=0.75, random_state=99)
# print(result)

# 
# print(diamonds.loc[~diamonds.index.isin(result.index), :])