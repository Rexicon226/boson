import numpy as np

# Define the two matrices
first_matrix = np.array([1, 3, 35, 9, 27, 30])  # 1x6 matrix
second_matrix = np.array([
    [636, 116, 636, 535],
    [8, 416, 5, 213],
    [0, 0, 0, 0],
    [635, 330, 637, 321],
    [4, 634, 324, 320],
    [640, 536, 640, 107]
])  # 6x4 matrix

# Perform matrix multiplication
result_matrix = first_matrix * second_matrix

print(result_matrix)
