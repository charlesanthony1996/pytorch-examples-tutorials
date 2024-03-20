import pandas as pd
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

