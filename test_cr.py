from classes.cr import CR, Padding
import numpy as np

cr = CR(1, 3, Padding.Valid)

filter = [
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1]
]

cr.filters[0] = np.array(filter)

# 4 x 4
image = [
    [0, 50, 0, 29],
    [0, 80, 31, 2],
    [33, 90, 0, 75],
    [0, 9, 0, 95]
]

# 1(filter) x 2 x 2
expected = np.array([[[29, -192], [-35, -22]]])
actual = cr.foward_prop(np.array(image))
print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")

# TODO: propper assert