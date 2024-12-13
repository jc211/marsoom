import pyglet
from pyglet import gl
from imgui_bundle import imgui, imguizmo
from imgui_bundle.python_backends.pyglet_backend import (
    PygletProgrammablePipelineRenderer,
)

gizmo = imguizmo.im_guizmo

class Window:
    def __init__(self, 
                 width: int = 1280, 
                 height: int = 720, 
                 caption: str = "Demo Window", 
                 resizable: bool = True, 
                 vsync: bool = True, 
                 visible: bool = True,
                 docking: bool = True,
                 ):

        self.window = pyglet.window.Window(
            width=width,
            height=height,
            caption=caption,
            resizable=resizable,
            vsync=vsync,
            visible=visible
        )
        self.bg_color = (1.0, 1.0, 1.0, 1.0)
        self.window.switch_to()        
        imgui.create_context()              
        io = imgui.get_io()
        if docking: 
            io.config_flags |= imgui.ConfigFlags_.docking_enable
            io.config_windows_move_from_title_bar_only = True  
        self.imgui_renderer = PygletProgrammablePipelineRenderer(
                self.window, attach_callbacks=True
        )
        self.window.push_handlers(self)
    
    def should_exit(self):
        return self.window.has_exit

    def imgui_active(self):
        return (
            imgui.is_any_item_hovered()
            or imgui.is_any_item_focused()
            or imgui.is_any_item_active()
        )

    def draw(self): # Overwrite this method
        imgui.begin("Hello, world!")
        imgui.end()
    
    def on_draw(self):
        self.check_gl_error()
        gl.glClearColor(*self.bg_color)
        self.window.clear()
        imgui.new_frame()
        gizmo.begin_frame()
        imgui.dock_space_over_viewport(flags=imgui.DockNodeFlags_.passthru_central_node)
        self.draw()
        imgui.render()
        self.imgui_renderer.render(imgui.get_draw_data())
    
    def check_gl_error(self):
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"OpenGL error: {error}")

    def run(self, fps:float=60.0):
        pyglet.app.run(1.0 / fps) 
    
    def step(self):
        pyglet.clock.tick(poll=True)
        for w in pyglet.app.windows:
            w.switch_to()
            w.dispatch_events()
            w.dispatch_event('on_draw')
            w.flip()

