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

    pyglet.clock.schedule(update)
    pyglet.app.run()
