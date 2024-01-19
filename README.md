# SIR/SEIR/Spatial SEIR Models in Python with Taichi Acceleration

These are ports of the models in this prototyping branch of [LASER](https://github.com/InstituteforDiseaseModeling/laser/tree/clorton/well-mixed-abc).

I found "compiler" (Taichi) errors somewhat opaque, but keeping the kernels small made it easier to locate problems. Taichi can be a bit fiddly about data types and doesn't automatically cast as often as C/C++.

Debugging my code (vs. "compiler errors") took two approaches:

1) Break the larger transmission kernel into a number of smaller kernels and inspect the results of each kernel invocation.
2) Rely on `<field>.to_numpy()` heavily in the debugger shell to investigate results. E.g., for "summing across rows" and "summing down columns" I had the `i` and `j` indices reversed. Pro-tip `ndarray.nonzero()[0]` and `<array>[<array>.nonzero()[0]]` are useful for finding non-zero values in large arrays.

Once I had everything debugged and working, initially on Apple Silicon with Metal, I was pleasantly surprised to run the same code, unmodified on x86-64 WSL on Windows with an Nvidia GPU.

I was also able to run, nearly unmodified, on a GPU enabled VM from IDM's self-service VM portal. I did have to make one change to the Taichi initialization: `ti.init(arch=ti.gpu, device_memory_GB=8)`. The GPU in that system has 12GB of RAM but I chose 8 since I was sure that was enough and wouldn't push the limits of the board.

Running Taichi on the CPU hasn't met my expectations\* (doesn't match Numpy+Numba multicore) but I may be missing something since I am new to Taichi. On the other hand, the GPU utilization has great results and, while fiddly as mentioned above, is easier to write and debug than PyCUDA.

Scenario: ~183M agents (Nigeria 2015 census) distributed across 774 LGAs. On each timestep incubating agents are updated, infectious agents are updated, and the force of infection in each LGA is determined from the number of local infectious agents +/- a portion of contagion coming in or going out to connected LGAs. The simulation runs for 365 timesteps with an initially fully susceptible population and 10 initial infections randomly placed.

|System|CPU|GPU|Wall Clock Time (seconds)|
|:-----|---|---|---------------|
|IDM Self-Service VM|Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz|NVIDIA A40-12Q|0:00:05.5|
|Apple MacBook Pro|Apple M1 Max|Apple M1 Max|0:00:11.3|
|Lenovo ThinkPad X1 Extreme|Intel Core i7-10850H @ 2.70GHz|NVIDIA GeForce GTX 1650|0:00:29.8|

\* Taichi on CPU:

|System|# Cores|Wall Clock Time (_minutes_)|
|------|:-----:|---------------------------|
|AMD EPYC 7763|16|~7:30|
|Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz|8|~9:30|
|Apple M1 Max - **NumPy+Numba**|10 (8/2)|**~1:15**|
