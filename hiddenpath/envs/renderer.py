import pyglet
from pyglet import graphics as pg
from pyglet import clock
from pyglet.gl import *
from math import tan, radians


class PygletRenderer(pyglet.window.Window):
    def __init__(self, width=1280, height=720, pathfinder=None, queue=None, semaphore=None):
        if pathfinder:
            self.win_size = {'width': width,
                             'height': height}
            self.pathfinder = pathfinder
            self.queue = queue
            self.semaphore = semaphore
        else:
            self.win_size = {'width': width, 'height': height}

        # create window
        super(PygletRenderer, self).__init__(
            width=self.win_size['width'], height=self.win_size['height'],
            resizable=True, fullscreen=False, caption="hiddenpath")
        self.set_minimum_size(32, 32)

        self.decorate_events()

        # setup the tick counter and Frames Per Second counter
        self.dt_count = clock.tick()
        self.fps_display = pyglet.clock.ClockDisplay()

        # rendering variables
        pyglet.options['debug_gl'] = False
        self.primitives = {'points': GL_POINTS,
                           'square_points': GL_QUADS,
                           'stripe': GL_LINE_STRIP,
                           'line_loop': GL_LINE_LOOP,
                           'quads': GL_QUADS}
        self.points_mode = {'mode': 'square_points',
                            'points': 1,
                            'square_points': 2}
        self.fov_y = 60.0
        self.camera = {'x_pan': 0.0,
                       'y_pan': 0.0,
                       'z_zoom': 1.0,
                       'z_offset': 0,  # z coordinate at which primitives are drawn
                       'zoom_factor': 1.1,
                       'scene_scale': 0}
        self.set_camera_scale(self.points_mode[self.points_mode['mode']])

        # objects to draw
        self.const_drawings = []
        self.batches = {'const': pg.Batch()}

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)

    def new_batch(self, name: str):
        self.batches[name] = pg.Batch()

    def reset_to_const(self):
        for name in self.batches:
            if name != 'const':
                self.batches[name] = pg.Batch()

    def reset_all(self):
        for name in self.batches:
            self.batches[name] = pg.Batch()

    def app_run(self):
        self.semaphore.release()
        self.semaphore.release()
        # pyglet.app.run()
        self.events_loop()

    def set_points_mode(self, mode: str):
        if mode not in self.points_mode:
            raise ValueError(f"There is no such points mode: '{mode}'")
        self.points_mode['mode'] = mode
        self.set_camera_scale(self.points_mode[mode])

    def set_camera_scale(self, scale):
        # if self.camera['scene_scale'] == scale:
        #     return
        self.camera['scene_scale'] = scale
        self.camera['x_default'] = self.win_size['width'] * scale * (-0.5)
        self.camera['y_default'] = self.win_size['height'] * scale * (-0.5)
        self.camera['z_default'] = -1 * (self.win_size['height'] * scale / 2.0) / tan(radians(self.fov_y / 2.0))
        self.camera['z_max'] = self.camera['z_default'] * 5  # zoom out limit, no reason to zoom out too far

    def decorate_events(self):

        @self.event
        def on_resize(width, height):
            width_diff, height_diff = width - self.win_size['width'], height - self.win_size['height']
            self.win_size['width'] = width
            self.win_size['height'] = height

            self.set_camera_scale(self.camera['scene_scale'])
            self.camera['x_pan'] += width_diff * self.camera['scene_scale'] / 2
            self.camera['y_pan'] += height_diff * self.camera['scene_scale'] / 2

            # sets the viewport
            glViewport(0, 0, width, height)

            # sets the projection
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(self.fov_y,  # field of view across y coord
                           width / float(height),  # aspect ratio
                           0.0,  # near clip
                           self.camera['z_max'])  # far clip
            # sets the model view
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

            return pyglet.event.EVENT_HANDLED

        # TODO adapt for devices with different scroll sensitivity
        @self.event
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            zoom = self.camera['z_zoom'] * self.camera['zoom_factor'] ** (scroll_y * (-1))
            if self.camera['z_default'] * zoom > self.camera['z_max']:
                self.camera['z_zoom'] = zoom

        @self.event
        def on_mouse_drag(x, y, dx, dy, button, modifiers):
            # press the LEFT MOUSE BUTTON to pan in screen pixel
            if button == pyglet.window.mouse.LEFT:
                screen_to_scene_ratio = self.camera['z_zoom'] * self.camera['scene_scale']
                self.camera['x_pan'] += dx * screen_to_scene_ratio
                self.camera['y_pan'] += dy * screen_to_scene_ratio
            # press the RIGHT MOUSE BUTTON pan in canvas pixels
            elif button == pyglet.window.mouse.RIGHT:
                self.camera['x_pan'] += dx
                self.camera['y_pan'] += dy

        @self.event
        def on_close():
            self.parallel_is_live = False
            self.close()

        # @self.event
        # def on_key_press(symbol, modifiers):
        #     self.tread_is_live = False
        #     if symbol == key.ESCAPE:
        #         self.close()

        # if self.pathfinder.settings['visualization'] == 'multiprocess':
        #     @self.event
        #     def on_draw():
        #         self.on_draw_decor()

    def read_queue(self):
        if self.queue.empty():
            return
        for drawings_tuple in self.queue.get():
            points, draw_as, color, batch_name = drawings_tuple
            if draw_as == 'new_batch' or draw_as == 'reset_batch':
                self.new_batch(batch_name)
            elif draw_as == 'reset_to_const':
                self.reset_to_const()
            elif draw_as == 'reset_all':
                self.reset_all()
            elif draw_as == 'close':
                self.has_exit = True
            else:
                self.add_to_drawings(points, draw_as, color, batch_name)

    # deprecated
    # def on_draw_decor(self):
    #
    #     self.dt_count += clock.tick()
    #
    #     glClearColor(0.2, 0.2, 0.2, 1)
    #     glClear(GL_COLOR_BUFFER_BIT)
    #
    #     glLoadIdentity()
    #     glPushMatrix()
    #     glTranslatef(self.camera['x_default'] + self.camera['x_pan'],
    #                  self.camera['y_default'] + self.camera['y_pan'],
    #                  self.camera['z_default'] * self.camera['z_zoom'])
    #
    #     # self.batch_lock.acquire()
    #     self.batches.draw()
    #     # self.batch_lock.release()
    #
    #     glPopMatrix()
    #     # draw PFS and flip buffer
    #     glTranslatef(self.camera['x_default'] / self.camera['scene_scale'],
    #                  self.camera['y_default'] / self.camera['scene_scale'],
    #                  self.camera['z_default'] / self.camera['scene_scale'])
    #     self.fps_display.draw()
    #
    #     self.flip()

    def events_loop(self):
        while not self.has_exit:
            self.read_queue()
            self.switch_to()
            self.dispatch_events()
            self.draw_frame()
        self.close()

    def draw_frame(self):
        self.dt_count += clock.tick()
        # timer rate for updating code
        # if self.dt_count >= 0.025:
        #     self.dt_count = 0

        glClearColor(0.2, 0.2, 0.2, 1)
        glClear(GL_COLOR_BUFFER_BIT)

        glLoadIdentity()
        glPushMatrix()
        glTranslatef(self.camera['x_default'] + self.camera['x_pan'],
                     self.camera['y_default'] + self.camera['y_pan'],
                     self.camera['z_default'] * self.camera['z_zoom'])

        for _, batch in self.batches.items():
            batch.draw()

        glPopMatrix()
        # draw PFS and flip buffer
        glTranslatef(self.camera['x_default'] / self.camera['scene_scale'],
                     self.camera['y_default'] / self.camera['scene_scale'],
                     self.camera['z_default'] / self.camera['scene_scale'])
        self.fps_display.draw()

        self.flip()

    def points_to_drawing(self, points_coords: list, draw_as='points', color=(0.0, 1.0, 0.0, 1.0)):
        coords, colors = [], []
        for coord in points_coords:
            if draw_as == 'square_points':
                point_size = self.points_mode['square_points']
                left = coord[0] * point_size
                bottom = coord[1] * point_size
                coords.extend([left, bottom, 0,
                               left, bottom + point_size, 0,
                               left + point_size, bottom + point_size, 0,
                               left + point_size, bottom, 0])
                colors.extend(color * 4)
            elif draw_as == 'stripe' or draw_as == 'line_loop':
                point_size = self.points_mode[self.points_mode['mode']]
                half_size = round(point_size / 2)
                coords.extend([coord[0] * point_size + half_size,
                               coord[1] * point_size + half_size, 0])
                colors.extend(color)
            elif draw_as == 'points':
                point_size = self.points_mode[self.points_mode['mode']]
                coords.extend([coord[0] * point_size, coord[1] * point_size, 0])
                colors.extend(color)
            elif draw_as == 'quads':
                point_size = self.points_mode[self.points_mode['mode']]
                coords.extend([coord[0] * point_size, coord[1] * point_size, 0])
                colors.extend(color)

        return {'number': round(len(coords) / 3),
                'coords': tuple(coords),
                'colors': tuple(colors),
                'draw_as': draw_as}

    # add objects to constant batch that is drawn at every frame
    def add_to_drawings(self, points_coords: list, draw_as='square_points',
                        color=(0.0, 1.0, 0.0, 1.0), batch_name='const'):
        if draw_as == 'walk':
            self.add_to_drawings(points_coords, draw_as='square_points', color=color, batch_name=batch_name)
            self.add_to_drawings(points_coords, draw_as='stripe', color=color, batch_name=batch_name)
            return
        drawing = self.points_to_drawing(points_coords, draw_as, color)
        self.add_to_batch_as(drawing, batch_name)
        # if const:
        #     self.const_drawings.append(drawing)

    # adds primitive to a batch (constant or temporary)
    def add_to_batch_as(self, drawing: dict, batch_name='const'):
        self.batches[batch_name].add(drawing['number'], self.primitives[drawing['draw_as']], None,
                                     ('v3f', drawing['coords']), ('c4f', drawing['colors']))
