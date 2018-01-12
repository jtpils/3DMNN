import os
import sys
from shape_generator import cube_coords_generator as generator
from shape_generator import cube_obj_wrapper as wrapper


if __name__ == "__main__":
    objects = 1000
    if not sys.argv[1] == None: objects = int(sys.argv[1])
    for i in range(0, objects):
        file = open("./../../data/concept/cube" + str(i) + ".obj", "w")
        file.write(wrapper(generator(cube=True)))
        file.close()