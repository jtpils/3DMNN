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
    windw.mesh = pyw.Wavefront("../model.obj")
    obj_defined = pyw.ObjParser(windw.mesh,"../model.obj")
    
    normals = []
    
    for i, vertex in enumerate(obj_defined.vertices, start=0):      
        #calculate normals
        #https://stackoverflow.com/questions/6656358/calculating-normals-in-a-triangle-mesh/6661242#6661242        
        # x,y,z
        vertex_normal = [.0, .0, .0]
        
        normals.append(vertex_normal)



    pyglet.clock.schedule(update)
    pyglet.app.run()
