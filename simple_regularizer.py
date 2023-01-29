def l2_regularizer(weight_matrix):
    return 0.01 * sum([weight ** 2 for weight in weight_matrix])

weights = [1, 2, 3, 4]
print(l2_regularizer(weights))