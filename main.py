import pygame as pg
import numpy as np
import taichi as ti
import taichi_glsl as ts
from taichi_glsl import vec2, vec3

ti.init(arch=ti.cpu)  # ti.cpu ti.vulkan ti.opengl ti.metal(macOS) ti.cuda
resolution = width, height = vec2(1360, 768)
# load texture
texture = pg.image.load("img/noshov.jpg")  # texture res - 2^n x 2^n (512 x 512, 1024 x 1024 e.t.c)
texture_size = texture.get_size()[0]
# texture color normallization 0 - 255 --> 0.0 - 1.0
texture_array = pg.surfarray.array3d(texture).astype(np.float32) / 255


@ti.data_oriented
class PyShader:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.full((width, height, 3), [0, 0, 0], np.uint8)
        # taichi fields
        self.screen_field = ti.Vector.field(3, ti.uint8, (width, height))
        self.texture_field = ti.Vector.field(3, ti.float32, texture.get_size())
        self.texture_field.from_numpy(texture_array)

    @ti.kernel
    def render(self, time: ti.float32):
        for frag_coord in ti.grouped(self.screen_field):
            # normalized pixel coords
            uv = (frag_coord - 0.5 * resolution) / resolution.y
            col = vec3(0.0)

            uv += vec2(0.2 * ts.sin(time / 2), 0.3 * ts.cos(time / 3))  # плавное перемещение координат
            # polar coords
            phi = ts.atan(uv.y, uv.x)

            rho = ts.length(uv)  # для круглой формы тоннеля
            # rho = pow(pow(uv.x ** 2, 4) + pow(uv.y ** 2, 4), 0.125)  # для квадратной формы тоннеля

            st = vec2(phi / ts.pi, 0.25 / rho)
            st.x += time / 14  # эффект вращения
            st.y += time / 2  # эффект движения
            col += self.texture_field[st * texture_size]

            col *= rho + 0.2  # глубина затемнения тоннеля
            col += 0.1 / rho * vec3(0.3, 0.1, 0.0)  # свет в конце тоннеля

            col = ts.clamp(col, 0.0, 1.0)
            self.screen_field[frag_coord.x, resolution.y - frag_coord.y] = col * 255

    def update(self):
        time = pg.time.get_ticks() * 1e-03  # time in sec
        self.render(time)
        self.screen_array = self.screen_field.to_numpy()

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.screen_array)

    def run(self):
        self.update()
        self.draw()


class App:
    def __init__(self):
        self.screen = pg.display.set_mode(resolution)
        self.clock = pg.time.Clock()
        self.shader = PyShader(self)

    def run(self):
        while True:
            self.shader.run()
            pg.display.flip()

            # [exit() for i in pg.event.get() if i.type == pg.QUIT] #  выход
            for i in pg.event.get():  # выход №2
                if i.type == pg.KEYDOWN:
                    if i.key == pg.K_ESCAPE:
                        exit()

            self.clock.tick(60)
            pg.display.set_caption(f'FPS: {self.clock.get_fps() : .2f}')


if __name__ == '__main__':
    app = App()
    app.run()
