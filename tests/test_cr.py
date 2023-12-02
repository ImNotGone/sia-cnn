from layers.utils.optimization_methods import Adam, GradientDescent, Momentum
from layers.convolutional import Convolutional, Padding
import numpy as np

Convolutional = Convolutional(2, 3, Adam(0.001), (1, 4, 4), Padding.VALID)


filter = [
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1]
]

Convolutional.filters[0][0] = np.array(filter)
Convolutional.filters[1][0] = np.array(filter)

# 1 x 4 x 4
image = np.array(
    [
    [
    [0, 50, 0, 29],
    [0, 80, 31, 2],
    [33, 90, 0, 75],
    [0, 9, 0, 95]
    ]],
)

# 2(filter) x 2 x 2
expected = np.array([
    [
        [29, -192],
        [-35, -22]
    ],
    [
        [29, -192],
        [-35, -22]
    ]
])
actual = Convolutional.forward_prop(np.array(image))
#print(f"image: {image}")
#print(f"expected: {expected}")
#print(f"actual: {actual}")

# 2 x 4 x 4
image = np.array(
    [
    [
    [0, 50, 0, 29],
    [0, 80, 31, 2],
    [33, 90, 0, 75],
    [0, 9, 0, 95]
    ],
    [
    [0, 50, 0, 29],
    [0, 80, 30, 2],
    [33, 90, 0, 75],
    [0, 9, 0, 95]
    ],
    ],
)
Convolutional = Convolutional(2, 3, Adam(0.001), (2, 4, 4), Padding.VALID)
filter1 = [
    [
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1],
    ],
    [
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1],
    ]
]

filter2 = [
    [
    [-1, 0, -1], 
    [2, 0, -2], 
    [-1, 0, -1],
    ],
    [
    [-1, 0, -1], 
    [-2, 0, -2], 
    [-1, 0, -1],
    ]
]

Convolutional.filters[0] = np.array(filter1)
Convolutional.filters[1] = np.array(filter2)

actual = Convolutional.forward_prop(np.array(image))
#print(f"image: {image}")
#print(f"expected: {expected}")
print(f"actual: {actual}")