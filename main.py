import numpy as np

array_1 = np.array([[1, 2, 3], [4, 5, 6]])
array_2 = np.array([[7, 8, 9]])


result = array_1 + array_2
print(result)

first_row = result[0, :]
print(first_row)


transposed = result.T
print(transposed)


mask = result > 10
print(mask)

selected = result[mask]
print(selected)


last_element = result[-1, 1]
print(last_element)

vectorized_result = result + 1
print(vectorized_result)

