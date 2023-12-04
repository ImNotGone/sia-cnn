from layers.pooling import Pooling, PoolingType
import numpy as np

# TEST MAX
pl = Pooling(PoolingType.MAX, 2)
pl.initialize((1, 4, 4))

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
expected = np.array([
    [[80],[31]], 
    [[90], [95]],
])

actual = pl.forward_prop(image)
print(f"image: {image}")
#print(f"expected: {expected}")
#print(f"actual: {actual}")
back = pl.back_prop(np.array([[[1, 2] , [3, 4]]]))
print(back)

# TEST MIN
pl = Pooling(PoolingType.MIN, 2)
pl.initialize((1, 4, 4))

# 4 x 4 x 1
image = np.array([[
    [0,50,3,29],
    [0,80,31,2],
    [33,90,102,75],
    [15,9,103,95],
]])

expected = np.array([
    [[0],[2]], 
    [[9], [75]],
])

actual = pl.forward_prop(image)
print(f"image: {image}")
#print(f"expected: {expected}")
#print(f"actual: {actual}")
back = pl.back_prop(np.array([[[1, 2] , [3, 4]]]))
print(back)

# TEST AVG
pl = Pooling(PoolingType.AVG, 2)
pl.initialize((1, 4, 4))

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


expected = np.array([
    [[32.5],[15.5]], 
    [[33], [42.5]],
])

actual = pl.forward_prop(image)
print(f"image: {image}")
#print(f"expected: {expected}")
#print(f"actual: {actual}")
back = pl.back_prop(np.array([[[1, 2] , [3, 4]]]))
print(back)
