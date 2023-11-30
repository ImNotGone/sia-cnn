from classes.pl import PL, PoolingType
import numpy as np

# TEST MAX
pl = PL(PoolingType.MAX, 2)

# 4 x 4 x 1
image = np.array([
    [[0],[50],[0],[29]],
    [[0],[80],[31],[2]],
    [[33],[90],[0],[75]],
    [[0],[9],[0],[95]],
])

expected = np.array([
    [[80],[31]], 
    [[90], [95]],
])

actual = pl.foward_prop(image)
print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")

# TEST MIN
pl = PL(PoolingType.MIN, 2)

# 4 x 4 x 1
image = np.array([
    [[0],[50],[3],[29]],
    [[0],[80],[31],[2]],
    [[33],[90],[102],[75]],
    [[15],[9],[103],[95]],
])

expected = np.array([
    [[0],[2]], 
    [[9], [75]],
])

actual = pl.foward_prop(image)
print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")

# TEST AVG
pl = PL(PoolingType.AVG, 2)

# 4 x 4 x 1
image = np.array([
    [[0],[50],[0],[29]],
    [[0],[80],[31],[2]],
    [[33],[90],[0],[75]],
    [[0],[9],[0],[95]],
])

expected = np.array([
    [[32.5],[15.5]], 
    [[33], [42.5]],
])

actual = pl.foward_prop(image)
print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")