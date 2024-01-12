#! /usr/bin/env python3

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import taichi as ti
from tqdm import tqdm

ti.init(arch=ti.gpu)


def set_params():
    TIMESTEPS = np.uint32(128)
    POP_SIZE = np.uint32(200_000_000)
    INF_MEAN = np.float32(5)
    INF_STD = np.float32(1)
    INIT_INF = np.uint32(10)
    R_NAUGHT = np.float32(2.5)

    SEED = np.uint32(20231205)

    parser = ArgumentParser()
    parser.add_argument("-t", "--timesteps", type=np.uint32, default=TIMESTEPS)
    parser.add_argument("-p", "--pop_size", type=np.uint32, default=POP_SIZE)
    parser.add_argument("--inf_mean", type=np.float32, default=INF_MEAN)
    parser.add_argument("--inf_std", type=np.float32, default=INF_STD)
    parser.add_argument("--initial_inf", type=np.uint32, default=INIT_INF)
    parser.add_argument("--r_naught", type=np.float32, default=R_NAUGHT)
    parser.add_argument("--seed", type=np.uint32, default=SEED)
    parser.add_argument(
        "-f", "--filename", type=Path, default=Path(__file__).parent / "sir.csv"
    )

    args = parser.parse_args()
    args.__setattr__("beta", np.float32(args.r_naught / args.inf_mean))

    return args  # might use vars(args) here if we need to return a dictionary


def test_sir(params):
    susceptibility = ti.field(dtype=ti.u8, shape=params.pop_size)
    susceptibility.fill(1)
    infected = ti.field(dtype=ti.u8, shape=params.pop_size)
    infected.fill(0)
    itimers = ti.field(dtype=ti.u8, shape=params.pop_size)
    itimers.fill(0)

    # initial_infs = np.random.choice(params.pop_size, params.initial_inf, replace=False)
    initial_infs = np.random.randint(
        0, high=params.pop_size, size=params.initial_inf, dtype=np.uint32
    )

    @ti.kernel
    def init_infs(initial_infs: ti.types.ndarray(ti.u32, 1)):
        for i in initial_infs:
            j = ti.cast(initial_infs[i], ti.i32)
            print(i, j)
            susceptibility[j] = ti.cast(0, ti.u8)
            infected[j] = ti.cast(1, ti.u8)
            itimers[j] = ti.round(
                ti.randn() * params.inf_std + params.inf_mean, ti.types.u8
            )

    init_infs(initial_infs)

    @ti.kernel
    def inf_update():
        for i in itimers:
            if itimers[i] > 0:
                tmp = itimers[i] - ti.cast(1, ti.u8)
                itimers[i] = tmp
                if tmp == 0:
                    infected[i] = ti.cast(0, ti.u8)

    @ti.func
    def sum(x):
        s = 0
        for i in x:
            s += x[i]
        return s

    @ti.kernel
    def transmission():
        contagion = sum(infected)
        force = contagion * params.beta / params.pop_size
        for i in susceptibility:
            if ti.random() < force * susceptibility[i]:
                susceptibility[i] = ti.cast(0, ti.u8)
                infected[i] = ti.cast(1, ti.u8)
                itimers[i] = ti.round(
                    ti.randn() * params.inf_std + params.inf_mean, ti.types.u8
                )

    results = ti.field(dtype=ti.u32, shape=(params.timesteps + 1, 4))
    results.fill(0)

    @ti.func
    def recovered(susceptibility, infected):
        r = 0
        for i in susceptibility:
            if (susceptibility[i] == 0) and (infected[i] == 0):
                r += 1
        return r

    @ti.kernel
    def record(t: ti.i32):
        results[t, 0] = t
        results[t, 1] = sum(susceptibility)
        results[t, 2] = sum(infected)
        results[t, 3] = recovered(susceptibility, infected)

    record(0)

    tstart = datetime.now()
    for t in tqdm(range(params.timesteps)):
        inf_update()
        transmission()
        record(t + 1)

    report = results.to_numpy()
    tfinish = datetime.now()
    print(f"Simulation took {tfinish - tstart} seconds")
    np.savetxt(params.filename, report, fmt="%u", delimiter=",", header="t,s,i,r")

    return


if __name__ == "__main__":
    params = set_params()
    print(f"Running SIR simulation with {params}")
    test_sir(params)
