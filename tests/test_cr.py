'''from layers.cr import CR, Padding
import numpy as np

cr = CR(1, 3, Padding.VALID)

filter = [
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1]
]

cr.filters[0] = np.array(filter)

# 4 x 4 x 1
image = np.array(
    [
    [[0], [50], [0], [29]],
    [[0], [80], [31], [2]],
    [[33], [90], [0], [75]],
    [[0], [9], [0], [95]]
    ],
)

# 1(filter) x 2 x 2
expected = np.array([[[29], [-192]], [[-35], [-22]]])
actual = cr.forward_prop(np.array(image))
print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")

# 4 x 4 x 2
image = np.array(
    [
    [[0, 0], [50, 50], [0, 0], [29, 29]],
    [[0, 0], [80, 80], [31, 31], [2, 2]],
    [[33, 33], [90, 90], [0, 0], [75, 75]],
    [[0, 0], [9, 9], [0, 0], [95, 95]]
    ],
)

# 1(filter) x 2 x 2
expected = np.array([[[29, 29], [-192, -192]], [[-35, -35], [-22, -22]]])
actual = cr.forward_prop(np.array(image))
print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")

# TODO: propper assert
'''

from layers.cr import CR, Padding
import numpy as np

cr = CR(2, 3, Padding.VALID)

filter = [
    [-1, 0, 1], 
    [-2, 0, 2], 
    [-1, 0, 1]
]

cr.filters[0] = np.array(filter)
cr.filters[1] = np.array(filter).T

# 4 x 4 x 1
image = np.array(
    [
    [[0], [50], [0], [29]],
    [[0], [80], [31], [2]],
    [[33], [90], [0], [75]],
    [[0], [9], [0], [95]]
    ],
)

# 1(filter) x 2 x 2
expected = np.array([
    [
        [29, -192],
        [-35, -22]
    ]
])
actual = cr.forward_prop(np.array(image))
print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")

# 4 x 4 x 2
image = np.array(
    [
    [[0, 0], [50, 50], [0, 0], [29, 29]],
    [[0, 0], [80, 80], [31, 31], [2, 2]],
    [[33, 33], [90, 90], [0, 0], [75, 75]],
    [[0, 0], [9, 9], [0, 0], [95, 95]]
    ],
)


expected = np.array([expected[0], expected[0]])
actual = cr.forward_prop(np.array(image))
print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")