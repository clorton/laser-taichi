import time

import numpy as np

N = 200_000_000


def numpy_sines(n):
    data = np.random.rand(n)
    return np.sin(data)


sines = numpy_sines(N)

t0 = time.perf_counter_ns()
sines = numpy_sines(N)
t1 = time.perf_counter_ns()

print(f"numpy_sines({N=}) took {(t1-t0)/1_000_000_000} seconds")
index = np.random.randint(0, N)
print(f"sines[{index=}] = {sines[index]}")

# Numba "out-of-the-box" version

import numba as nb


@nb.jit(nopython=True, nogil=True, cache=False)
def numba_sines(N):
    data = np.random.rand(N)
    sines = np.sin(data)
    return sines


sines = numba_sines(np.uint32(N))

t0 = time.perf_counter_ns()
sines = numba_sines(np.uint32(N))
t1 = time.perf_counter_ns()

print(f"numba_sines({N=}) took {(t1-t0)/1_000_000_000} seconds")
index = np.random.randint(0, N)
print(f"sines[{index=}] = {sines[index]}")

# Numba optimized version

from numba import float64, prange, uint32


@nb.jit(float64[:](uint32), nopython=True, nogil=True, parallel=True)
def numba_opt_sines(N):
    sines = np.empty(N)
    for i in prange(N):
        sines[i] = np.sin(np.random.rand())
    return sines


sines = numba_opt_sines(np.uint32(N))

t0 = time.perf_counter_ns()
sines = numba_opt_sines(np.uint32(N))
t1 = time.perf_counter_ns()

print(f"numba_opt_sines({N=}) took {(t1-t0)/1_000_000_000} seconds")
index = np.random.randint(0, N)
print(f"sines[{index=}] = {sines[index]}")

# Taichi CPU version

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu)

sines = ti.field(dtype=ti.f32, shape=N)


@ti.kernel
def taichi_cpu_sines():
    for i in sines:
        sines[i] = tm.sin(ti.random(float))


taichi_cpu_sines()
t0 = time.perf_counter_ns()
taichi_cpu_sines()
t1 = time.perf_counter_ns()

print(f"taichi_cpu_sines({N=}) took {(t1-t0)/1_000_000_000} seconds")
index = np.random.randint(0, N)
print(f"sines[{index=}] = {sines[index]}")

# Taichi GPU version

ti.init(arch=ti.gpu)

sines = ti.field(dtype=ti.f32, shape=N)


@ti.kernel
def taichi_gpu_sines():
    for i in sines:
        sines[i] = tm.sin(ti.random(float))


taichi_gpu_sines()
t0 = time.perf_counter_ns()
taichi_gpu_sines()
ti.sync()
t1 = time.perf_counter_ns()

print(f"taichi_gpu_sines({N=}) took {(t1-t0)/1_000_000_000} seconds")
index = np.random.randint(0, N)
print(f"sines[{index=}] = {sines[index]}")
