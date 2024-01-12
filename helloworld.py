import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

n = 768
pixels = ti.field(dtype=ti.f32, shape=(n * 2, n))


@ti.func
def complex_sqr(z):
    return tm.vec2(z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2)


@ti.kernel
def paint(t: float):
    for i, j in pixels:
        c = tm.vec2(-0.8, ti.cos(t) * 0.2)
        z = tm.vec2(i / n - 1, j / n - 0.5) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


gui = ti.GUI("Julia Set", res=(n * 2, n))

i = 0
while gui.running:
    paint(i)
    gui.set_image(pixels)
    gui.show()
    i += 0.03125
