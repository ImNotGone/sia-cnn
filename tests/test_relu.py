from layers.relu import Relu        
import numpy as np

r = Relu()
r.initialize((1, 2, 2))
input = np.array([[
    [1, -1],
    [-3, 4],
]])
actual = r.forward_prop(input)
expected = np.array([[
    [1, 0],
    [0, 4],
]])
print(f"expected FP:\n{expected}")
print(f"actual:\n{actual}")
loss = np.array([
    [5, 12],
    [13, 7],
])
actual = r.back_prop(loss)
expected = np.array([
    [5, 0],
    [0, 7],
])
print(f"expected BP:\n{expected}")
print(f"actual:\n{actual}")
