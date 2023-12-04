from layers.utils.optimization_methods import Adam, GradientDescent, Momentum
from layers.convolutional import Convolutional, Padding
import numpy as np

convolutional = Convolutional(2, 3, Adam(0.001), Padding.VALID)
convolutional.initialize((1, 4, 4))


filter = [
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1]
]

convolutional.filters[0][0] = np.array(filter)
convolutional.filters[1][0] = np.array(filter)

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
actual = convolutional.forward_prop(np.array(image))
#print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")

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
    [0, 80, 31, 2],
    [33, 90, 0, 75],
    [0, 9, 0, 95]
    ],
    ],
)
convolutional = Convolutional(2, 3, Adam(0.001), Padding.VALID)
convolutional.initialize((2, 4, 4))
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
    [1, 0, -1], 
    [2, 0, -2], 
    [1, 0, -1],
    ],
    [
    [1, 0, -1], 
    [2, 0, -2], 
    [1, 0, -1],
    ]
]

convolutional.filters[0] = np.array(filter1)
convolutional.filters[1] = np.array(filter2)

actual = convolutional.forward_prop(np.array(image))
expected[1] = expected[1]*-1
#print(f"image: {image}")
print(f"expected: {expected*2}")
print(f"actual: {actual}")
