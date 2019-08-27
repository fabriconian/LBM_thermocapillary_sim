from LatFlow.utils import *
import time

vertices = np.array([
    [1,10],
    [10,19],
    [19,10],
    [10,1],
])
vertices = np.random.uniform(0,10,[2000,2])
t = time.time()
polygon_array = create_polygon_vek([200,200], vertices)
print(time.time() - t,'\n')
t = time.time()
polygon_array2 = create_polygon([200,200], vertices)
print(time.time() - t,'\n')

polygon_array = create_polygon_vek([20,20], vertices)