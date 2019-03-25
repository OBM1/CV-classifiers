import numpy as np

ind = np.array([1, 3, 5, 7], dtype=int)

one_hot = np.zeros((4, 10))

one_hot[np.arange(4), ind] = 1

print(ind)
print(one_hot)
