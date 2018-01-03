import os
import random as rand

def cube_coords_generator(scale=1, symmetry=False):
    # [x, y, z]
    vertices = [1, 2, 3, 4, 5, 6, 7, 8]
    lower_plane = vertices[0:4]
    upper_plane = vertices[4:8]

    highest_point = rand.randint(0, 100)/100
    lowest_point = round(highest_point - rand.randint(30, 100)/100, 4)

    z_1 = rand.randint(50, 100)/100
    x_1 = rand.randint(0, 50)/100
    
    x_2 = rand.randint(0, 50)/100
    z_2 = rand.randint(50, 100)/100

    first_point = { "x": x_1,"y": lower_plane, "z": z_1 }
    second_point = { "x": x_2, "y": lower_plane, "z": z_1}

    first_plane = [0,0,0,0]

    first_plane[0] = first_point
    first_plane[1] = second_point
    first_plane[2] = { "x": x_1,"y": lower_plane, "z": z_2 }
    first_plane[3] = { "x": x_2,"y": lower_plane, "z": z_2 }

    for i in range(0,4):
        upper_plane[i] = { "x": first_plane[i]["x"], "y": highest_point, "z": first_plane[i]["z"]}
        lower_plane[3-i] = { "x": first_plane[i]["x"], "y": lowest_point, "z": first_plane[i]["z"]}

    if not symmetry:
        for i in range(0, len(lower_plane)):
            x = (rand.randint(0, 100)/100)*scale
            z = (rand.randint(0, 100)/100)*scale
            lower_plane[i] = { "x": x, "y": lowest_point, "z": z}

    print(lower_plane)
    print(upper_plane)
    return lower_plane + upper_plane

def cube_obj_wrapper(coords, name="object"):

    lines = ""
    for i in range(0, len(coords)):
        lines += "v " + str(coords[i]["x"]) + " " + str(coords[i]["y"]) + " " + str(coords[i]["z"]) + " #" + str(i + 1) + "\n"
    # Пълно шано, е това свързване, едно по едно ги пробвах.
    lines += 'f 2 1 3 4\nf 6 5 7 8\nf 1 2 7 8\nf 3 4 5 6\nf 2 4 5 7\nf 1 3 6 8'
    return lines
