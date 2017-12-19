import os
import random as rand


def cube_coords_generator(scale=1, symmetry=False):
    # [x, y, z]
    vertices = [1, 2, 3, 4, 5, 6, 7, 8]
    lower_layer = vertices[0:4]
    upper_layer = vertices[4:8]

    highest_point = rand.randint(0, 100)/100
    lowest_point = round(highest_point - rand.randint(0, 100)/100, 4)

    for i in range(0, len(upper_layer)):
        x = rand.randint(0, 100)/100
        z = rand.randint(0.00, 100)/100
        upper_layer[i] = [x, highest_point, z]
        lower_layer[i] = [x, lowest_point, z]

    if not symmetry:
        for i in range(0, len(lower_layer)):
            x = (rand.randint(0, 100)/100)*scale
            z = (rand.randint(0, 100)/100)*scale

            lower_layer[i] = [x, lowest_point, z]

    return lower_layer + upper_layer

def cube_obj_wrapper(coords, name="object"):
    lines = ""
    for i in range(0, len(coords)):
        lines += "v " + str(coords[i][0]) + " " + str(coords[i][1]) + " " + str(coords[i][2]) + "\n"

    lines += '''f 1 4 8 5\nf 4 3 7 8\nf 3 2 6 7\nf 2 1 5 6\nf 1 2 3 4\nf 5 6 7 8'''
    file = open(name+".obj", "w")
    file.write(lines)
    file.close()
    return lines


print(cube_obj_wrapper(cube_coords_generator(symmetry=True)))
