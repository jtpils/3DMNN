import pyglet
import pywavefront as pyw
from viewport import obj_viewport

def update(dt):
    windw.file_name = "model.obj"    
    
    windw.rotation += 45 * dt

    if windw.rotation > 720:
       windw.rotation = 0

if __name__ == "__main__":
    windw = obj_viewport(1280, 720, ".obj file viewer", resizable=True)
    windw.mesh = pyw.Wavefront("model.obj")
    
    obj_base = pyw.ObjParser(windw.mesh, "model.obj")
    obj_vertices = obj_base.vertices
    trianglesCount = 1

    # first element (0) is empty
    triangles = []

    triangle = []
    for i, vertex in enumerate(obj_vertices,start = 0):
        triangle.append(vertex)
        if i % 3 == 0:
            trianglesCount += 1
            print(trianglesCount)
            print(triangle)
            triangles.append(triangle)
            triangle = []
            
    print(obj_vertices[0])
    pyglet.clock.schedule(update)
    pyglet.app.run()
