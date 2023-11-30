from classes.pl import PL, PoolingType
import numpy as np

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
]
)
actual = pl.foward_prop(image)
print(f"image: {image}")
print(f"expected: {expected}")
print(f"actual: {actual}")