import matplotlib.pyplot as plt
from numpy import array, sin, cos, pi, random

def random_walk(N=100):
    # Making random angels and radiuses
    random_angle = random.uniform(0, 2*pi, N)
    random_radius = random.uniform(0, 1, N)

    # Defining the starting position and the path array
    start_pos = array([0.0, 0.0])
    path = []

    # Defining the move funcion 
    def move(r, theta): 
        return r*array([cos(theta), sin(theta)])

    pos = start_pos

    for r, theta in zip(random_radius, random_angle):
        path.append(pos)
        pos = pos + move(r, theta)

    path_x = [i[0] for i in path]
    path_y = [i[1] for i in path]

    plt.plot(path_x, path_y)

for i in range(0, 10):
    random_walk(10000)

plt.show()
