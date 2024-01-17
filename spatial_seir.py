#! /usr/bin/env python3

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import taichi as ti
from tqdm import tqdm

import nigeria

ti.init(arch=ti.gpu)


def set_params():
    TIMESTEPS = np.uint32(128)
    INC_MEAN = np.float32(4)
    INC_STD = np.float32(1)
    INF_MEAN = np.float32(5)
    INF_STD = np.float32(1)
    INIT_INF = np.uint32(10)
    R_NAUGHT = np.float32(2.5)

    SEED = np.uint32(20231205)

    parser = ArgumentParser()
    parser.add_argument("-t", "--timesteps", type=np.uint32, default=TIMESTEPS)
    parser.add_argument("--inc_mean", type=np.float32, default=INC_MEAN)
    parser.add_argument("--inc_std", type=np.float32, default=INC_STD)
    parser.add_argument("--inf_mean", type=np.float32, default=INF_MEAN)
    parser.add_argument("--inf_std", type=np.float32, default=INF_STD)
    parser.add_argument("--initial_inf", type=np.uint32, default=INIT_INF)
    parser.add_argument("--r_naught", type=np.float32, default=R_NAUGHT)
    parser.add_argument("--seed", type=np.uint32, default=SEED)
    parser.add_argument(
        "-f", "--filename", type=Path, default=Path(__file__).parent / "seir.csv"
    )

    args = parser.parse_args()
    args.__setattr__("beta", np.float32(args.r_naught / args.inf_mean))

    return args  # might use vars(args) here if we need to return a dictionary


def load_populations():
    # filter out the state and national level LGAs
    lgas = {k: v for k, v in nigeria.lgas.items() if len(k.split(":")) == 5}
    # from dictionary with key and values ((population, year), (lat, lon), area) extract population sizes
    POP_INDEX = 0  # first item in value tuple is population info tuple
    SIZE_INDEX = 0  # first item in population info tuple is population size
    pops = np.array([v[POP_INDEX][SIZE_INDEX] for v in lgas.values()], dtype=np.uint32)

    # pops = np.array([10_000, 10_000, 10_000], dtype=np.uint32)

    return pops


def load_network():
    gravity = np.array(nigeria.gravity, dtype=np.float32)

    # gravity = np.array([[0.85, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]], dtype=np.float32)

    return gravity


def spatial_seir(params):
    pops_np = load_populations()
    pop_size = pops_np.sum()
    f_node_populations = ti.field(dtype=ti.u32, shape=len(pops_np))
    f_node_populations.from_numpy(pops_np)
    num_pops = len(pops_np)
    f_network = ti.field(dtype=ti.f32, shape=(num_pops, num_pops))
    network_np = load_network()
    f_network.from_numpy(network_np)

    f_susceptibility = ti.field(dtype=ti.u8, shape=pop_size)
    f_susceptibility.fill(1)
    f_etimers = ti.field(dtype=ti.u8, shape=pop_size)
    f_etimers.fill(0)
    f_itimers = ti.field(dtype=ti.u8, shape=pop_size)
    f_itimers.fill(0)

    nodeid_np = np.zeros(pop_size, dtype=np.uint16)
    nodeidx = 0
    for i, pop in enumerate(pops_np):
        nodeid_np[nodeidx : nodeidx + pop] = i
        nodeidx += pop

    f_nodeids = ti.field(dtype=ti.u16, shape=pop_size)
    f_nodeids.from_numpy(nodeid_np)

    f_results = ti.field(dtype=ti.u32, shape=(params.timesteps + 1, 5))
    f_results.fill(0)

    f_contagion = ti.field(dtype=ti.u32, shape=num_pops)
    # f_contagion.fill(0)

    f_forces = ti.field(dtype=ti.f32, shape=num_pops)
    # f_forces.fill(0)

    f_history = ti.field(dtype=ti.u32, shape=(params.timesteps + 1, num_pops))

    f_transfer = ti.field(dtype=ti.u32, shape=(num_pops, num_pops))
    # f_transfer.fill(0)

    f_axis_sums = ti.field(dtype=ti.u32, shape=num_pops)

    # initial_infs = np.random.choice(pop_size, params.initial_inf, replace=False)
    initial_infs = np.random.randint(
        0, high=pop_size, size=params.initial_inf, dtype=np.uint32
    )
    print(f"initial_infs = {initial_infs}")

    @ti.kernel
    def init_infs(initial_infs: ti.types.ndarray(ti.u32, 1)):
        for i in initial_infs:
            j = ti.cast(initial_infs[i], ti.i32)
            # print(i, j)
            f_susceptibility[j] = ti.cast(0, ti.u8)
            duration = ti.round(ti.randn() * params.inf_std + params.inf_mean)
            if duration <= 0:
                duration = 1
            f_itimers[j] = ti.cast(duration, ti.u8)

    @ti.kernel
    def inf_update():
        for i in f_itimers:
            if f_itimers[i] > 0:
                tmp = f_itimers[i] - ti.cast(1, ti.u8)
                f_itimers[i] = tmp

    @ti.kernel
    def inc_update():
        for i in f_etimers:
            if f_etimers[i] > 0:
                tmp = f_etimers[i] - ti.cast(1, ti.u8)
                f_etimers[i] = tmp
                if tmp == 0:
                    duration = ti.round(ti.randn() * params.inf_std + params.inf_mean)
                    if duration <= 0:
                        duration = 1
                    f_itimers[i] = ti.cast(duration, ti.u8)

    @ti.func
    def sum(x):
        s = 0
        for i in x:
            s += x[i]
        return s

    @ti.func
    def infectious(susceptibility, itimers):
        inf = 0
        for i in susceptibility:
            if (susceptibility[i] == 0) and (itimers[i] > 0):
                inf += 1
        return inf

    ########################################

    @ti.kernel
    def tx0(t: ti.i32):
        # zero out the f_contagion array
        for i in f_contagion:
            f_contagion[i] = 0

    @ti.kernel
    def tx1(t: ti.i32):
        # accumulate contagion for each node
        for i in f_susceptibility:
            if (f_susceptibility[i] == 0) and (f_itimers[i] > 0):
                f_contagion[ti.cast(f_nodeids[i], ti.i32)] += 1

    @ti.kernel
    def tx2(t: ti.i32):
        # multiple accumulated contagion by the network
        for i, j in f_transfer:
            f_transfer[i, j] = ti.cast(
                ti.round(f_contagion[i] * f_network[i, j]), ti.u32
            )

    @ti.kernel
    def tx3(t: ti.i32):
        # accumulate across rows for incoming contagion
        for i in f_axis_sums:
            f_axis_sums[i] = 0
        for i, j in f_transfer:
            f_axis_sums[j] += f_transfer[i, j]
        for i in f_axis_sums:
            f_contagion[i] = f_contagion[i] + f_axis_sums[i]

    @ti.kernel
    def tx4(t: ti.i32):
        # accumulate down columns for outgoing contagion
        for i in f_axis_sums:
            f_axis_sums[i] = 0
        for i, j in f_transfer:
            f_axis_sums[i] += f_transfer[i, j]
        for i in f_axis_sums:
            if f_axis_sums[i] <= f_contagion[i]:
                f_contagion[i] = f_contagion[i] - f_axis_sums[i]
            else:
                f_contagion[i] = ti.u32(0xDEADBEEF)

    @ti.kernel
    def tx5(t: ti.i32):
        # record total contagion for each node
        for i in f_contagion:
            f_history[t, i] = f_contagion[i]

    @ti.kernel
    def tx6(t: ti.i32):
        # multiply contagion by beta
        for i in f_forces:
            f_forces[i] = params.beta * f_contagion[i]

    @ti.kernel
    def tx7(t: ti.i32):
        # divide node contagion by node population
        for i in f_forces:
            f_forces[i] = f_forces[i] / f_node_populations[i]

    @ti.kernel
    def tx8(t: ti.i32):
        # visit each individual determining transmision by node force of infection and individual susceptibility
        for i in f_susceptibility:
            if ti.random() < (
                f_forces[ti.cast(f_nodeids[i], ti.i32)] * f_susceptibility[i]
            ):
                f_susceptibility[i] = ti.cast(0, ti.u8)
                duration = ti.round(ti.randn() * params.inc_std + params.inc_mean)
                if duration <= 0:
                    duration = 1
                f_etimers[i] = ti.cast(duration, ti.u8)

    ########################################

    @ti.kernel
    def transmission(t: ti.i32):
        # zero out the f_contagion array
        for i in f_contagion:
            f_contagion[i] = 0

        # accumulate contagion for each node
        for i in f_susceptibility:
            if (f_susceptibility[i] == 0) and (f_itimers[i] > 0):
                f_contagion[ti.cast(f_nodeids[i], ti.i32)] += 1

        # multiple accumulated contagion by the network
        for i, j in f_transfer:
            f_transfer[i, j] = ti.cast(
                ti.round(f_contagion[i] * f_network[i, j]), ti.u32
            )

        # accumulate across rows for incoming contagion
        for i in f_axis_sums:
            f_axis_sums[i] = 0
        for i, j in f_transfer:
            f_axis_sums[i] += f_transfer[i, j]
        for i in f_axis_sums:
            f_contagion[i] = f_contagion[i] + f_axis_sums[i]

        # accumulate down columns for outgoing contagion
        for i in f_axis_sums:
            f_axis_sums[i] = 0
        for i, j in f_transfer:
            f_axis_sums[j] += f_transfer[i, j]
        for i in f_axis_sums:
            f_contagion[i] = f_contagion[i] - f_axis_sums[i]

        # record total contagion for each node
        for i in f_contagion:
            f_history[t, i] = f_contagion[i]

        # multiply contagion by beta
        for i in f_contagion:
            f_contagion[i] = ti.cast(f_contagion[i] * params.beta, ti.u32)

        # divide node contagion by node population
        for i in f_contagion:
            f_contagion[i] = ti.cast(f_contagion[i] / f_node_populations[i], ti.u32)

        # visit each individual determining transmision by node force of infection and individual susceptibility
        for i in f_susceptibility:
            if ti.random() < (
                f_contagion[ti.cast(f_nodeids[i], ti.i32)] * f_susceptibility[i]
            ):
                f_susceptibility[i] = ti.cast(0, ti.u8)
                duration = ti.round(ti.randn() * params.inc_std + params.inc_mean)
                if duration <= 0:
                    duration = 1
                f_etimers[i] = ti.cast(duration, ti.u8)

    @ti.func
    def exposed(susceptibility, etimers):
        e = 0
        for i in susceptibility:
            if (susceptibility[i] == 0) and (etimers[i] > 0):
                e += 1
        return e

    @ti.func
    def recovered(susceptibility, etimers, itimers):
        r = 0
        for i in susceptibility:
            if (susceptibility[i] == 0) and (etimers[i] == 0) and (itimers[i] == 0):
                r += 1
        return r

    @ti.kernel
    def record(t: ti.i32):
        f_results[t, 0] = t
        f_results[t, 1] = sum(f_susceptibility)
        f_results[t, 2] = exposed(f_susceptibility, f_etimers)
        f_results[t, 3] = infectious(f_susceptibility, f_itimers)
        f_results[t, 4] = recovered(f_susceptibility, f_etimers, f_itimers)

    init_infs(initial_infs)
    record(0)

    tstart = datetime.now()
    for t in tqdm(range(params.timesteps)):
        inf_update()
        inc_update()

        # _c = np.zeros(num_pops, dtype=np.uint32)
        # np.add.at(_c, nodeid_np[f_itimers.to_numpy() != 0], 1)
        # _t = (_c * network_np).round().astype(np.uint32)
        # print(f"{_t.sum(axis=1)}")
        # print(f"{_t.sum(axis=0)}")

        # transmission(t)
        tx0(t)  # zero out the f_contagion array
        tx1(t)  # accumulate contagion for each node (into f_contagion[f_nodeids])
        tx2(t)  # multiple accumulated contagion by the network (into f_transfer)
        tx3(t)  # accumulate across rows for incoming contagion (into f_contagion)
        tx4(t)  # accumulate down columns for outgoing contagion (out of f_contagion)
        tx5(t)  # record total contagion for each node (into f_history[t,:])
        tx6(t)  # multiply contagion by beta (into f_forces)
        tx7(t)  # divide node contagion by node population (in f_forces)
        tx8(
            t
        )  # visit each individual determining transmision by node force of infection and individual susceptibility
        record(t + 1)
        ti.sync()

    report = f_results.to_numpy()
    tfinish = datetime.now()
    print(f"Simulation took {tfinish - tstart} seconds")
    print(f"Writing SEIR results to {params.filename}")
    np.savetxt(
        params.filename, report, fmt="%u", delimiter=",", header="timestep,S,E,I,R"
    )

    history = f_history.to_numpy()
    print(f"Writing history to {params.filename.parent / 'history.csv'}")
    np.savetxt(
        params.filename.parent / "history.csv",
        history,
        fmt="%u",
        delimiter=",",
        header=",".join([f"node{i}" for i in range(num_pops)]),
    )

    return


if __name__ == "__main__":
    params = set_params()
    print(f"Running SIR simulation with {params}")
    spatial_seir(params)
