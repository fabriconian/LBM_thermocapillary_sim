from LatFlow.utils import *
import time

vertices = np.array([
    [5,12],
    [8,18],
    [13,14],
    [11,6],
    [4,6],
])
t = time.time()
polygon_array = create_polygon_vek([2000,2000], vertices)
print(time.time() - t,'\n')
t = time.time()
polygon_array2 = create_polygon([2000,2000], vertices)
print(time.time() - t,'\n')
