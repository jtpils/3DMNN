import ctypes
import pyglet
from pyglet.gl import *
import pywavefront as pyw

class obj_viewport(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.set_minimum_size(150, 150)
        glClearColor(0.1, 0.1, 0.1, 1.0)  # Background

        self.clipping = 1.0
        self.field_of_view = 40
        self.lightfv = ctypes.c_float * 4
        self.rotation = 0
        self.file_name = ""
        self.render_mode = GL_TRIANGLES
        self.mesh = None

        #Objects and properties inside the viewport define here:
        #  . . .
        # Test objects while developing the viewport
        #                                                              x_1,y_2,z_2, x_2,y_2,z_2 ,x_3...
        self.test_triangle = pyglet.graphics.vertex_list(3, ('v3f', [-0.5,-0.5,0.0, 0.5,-0.5,0.0, 0.0,0.5,0.0]),
                                                            ('c3B', [100,200,200, 200,100,100, 100,200,100]))

                                                                #indices
        self.test_quad = pyglet.graphics.vertex_list_indexed(4, [0,1,2, 2,3,0],
                                                    #vertices
                                                    ('v3f', [-0.5,-0.5,0.0, 0.5,-0.5,0.0, 0.5,0.5,0.0, -0.5,0.5,0.0]),
                                                    #colors
                                                    ('c3B', [100,200,200, 200,100,100, 100,200,100, 200,100,200]))

    def on_draw(self):

        to_draw = self.mesh

        self.clear()
        glLoadIdentity()

        glLightfv(GL_LIGHT0, GL_POSITION, self.lightfv(-40, 200, 100, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, self.lightfv(0.8, 0.8, 0.8, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.lightfv(0.5, 0.5, 0.5, 1.0))

        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)

        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_MODELVIEW)

        glTranslated(0, 0, -2)
        glRotatef(45, 0, 0, 1)
        glRotatef(0, 0, 0, 1)
        glRotatef(self.rotation, 0, 1, 0)
        glRotatef(45, 1, 0, 0)

        to_draw.draw()

    def on_resize(self, width, height):

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.field_of_view, float(width)/height, self.clipping, 100)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glViewport(0, 0, width, height)
