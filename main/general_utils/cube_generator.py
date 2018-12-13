import os
import random as rand
import sys
# Absolute retardation. please spare me

def cube_coords_generator(scale=1, symmetry=True, cube=False):
    # [x, y, z]
    vertices = [1, 2, 3, 4, 5, 6, 7, 8]
    
    lower_plane = vertices[0:4]
    upper_plane = vertices[4:8]

    first_plane = []
    
    if cube:
        # Center 0,0,0
        center = 0
        distance = rand.randint(0,100*scale)/100
        x_1 = center - distance
        x_2 = center + distance

        highest_point = center + distance
        lowest_point = center - distance

        first_plane.append({ "x": x_1, "y": x_1, "z": x_1 })
        first_plane.append({ "x": x_2, "y": x_1, "z": x_1 })
        first_plane.append({ "x": x_1, "y": x_1, "z": x_2 })
        first_plane.append({ "x": x_2, "y": x_1, "z": x_2 })
    else:
        highest_point = rand.randint(0, 100)/100
        lowest_point = round(highest_point - rand.randint(30, 100)/100, 4)

        # Creating two points
        z_1 = rand.randint(50, 100*scale)/100
        x_1 = rand.randint(0, 50*scale)/100
        # Creating another two points
        x_2 = rand.randint(0, 50*scale)/100
        z_2 = rand.randint(50, 100*scale)/100

        # using them to create a line
        first_point = { "x": x_1, "y": lower_plane, "z": z_1 }
        # notice - z and y is the same
        second_point = { "x": x_2, "y": lower_plane, "z": z_1}

        # next - Creating a plane
        # that first line we created
        first_plane.append(first_point)
        first_plane.append(second_point)
        # then another one with the same x and z coordinates but different z
        first_plane.append({ "x": x_1,"y": lower_plane, "z": z_2 })
        first_plane.append({ "x": x_2,"y": lower_plane, "z": z_2 })
        # this makes them parallel
        # Now we have a quad. on 2 axises - x and z -> notice y is the same    
    #endif
        # a dict with all the points and the other parallel to the first plane.
    for i in range(0,4):
        upper_plane[i] = { "x": first_plane[i]["x"], "y": highest_point, "z": first_plane[i]["z"]}
        # this 3-i is because the order of the points, so I can traverse them easier later.
        lower_plane[3-i] = { "x": first_plane[i]["x"], "y": lowest_point, "z": first_plane[i]["z"]}

    # This is supposed to give any kind of a irregular polyhedron, not working
    if not symmetry:
        for i in range(0, len(lower_plane)):
            x = (rand.randint(0, 100)/100)*scale
            z = (rand.randint(0, 100)/100)*scale
            lower_plane[i] = { "x": x, "y": lowest_point, "z": z}

    # Connecting both planes, creating effectively a cuboid
    return lower_plane + upper_plane

def cube_obj_wrapper(coords, name="object"):

    lines = ""
    for i in range(0, len(coords)):
        lines += "v " + str(coords[i]["x"]) + " " + str(coords[i]["y"]) + " " + str(coords[i]["z"]) + " #" + str(i + 1) + "\n"
    # Traversing the vertices by hand. Ignore pls.
    lines += 'f 2 1 3 4\nf 6 5 7 8\nf 1 2 7 8\nf 3 4 5 6\nf 2 4 5 7\nf 1 3 6 8'
    return lines

if __name__ == "__main__":
    objects = 1000
    if not sys.argv[1] == None: objects = int(sys.argv[1])
    for i in range(0, objects):
        file = open("./../../data/concept/cube" + str(i) + ".obj", "w")
        file.write(cube_obj_wrapper(cube_coords_generator(cube=True)))
        file.close()