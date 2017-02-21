from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np


def emd(source, target, cost):
    epsilon = 1e-8

    a, b, c = np.copy(source), np.copy(target), np.copy(cost)
    assert np.abs(a.sum() - b.sum()) < epsilon
    assert np.all(c >= 0.0)

    unvisited = -1
    boundary = -2

    m, n = len(a), len(b)

    # initialization
    d, dx, dy = np.zeros([m, n]), np.zeros(m), np.zeros(n)
    queue = np.zeros(m, dtype=np.int32)
    fx = np.min(c, axis=1)
    fy = np.min(c - fx[:, np.newaxis], axis=0)

    delta, delta_x, delta_y = np.zeros(n), np.zeros(m), np.zeros(n)

    for i in range(m):
        while a[i] > dx[i] + epsilon:
            start = i

            # <editor-fold desc="Initialization for Breath First Search">
            trace_x = unvisited * np.ones(m, dtype=np.int32)
            trace_y = unvisited * np.ones(n, dtype=np.int32)
            front, rear = 0, 0
            queue[0] = start
            trace_x[start] = boundary
            delta_x[start] = a[start] - dx[start]
            delta = c[start] - fx[start] - fy
            arg = start * np.ones(n, dtype=np.int32)
            finish = unvisited
            # </editor-fold>

            while finish == unvisited:

                # <editor-fold desc="Find augmenting path to increase the flow">
                while front <= rear and finish == unvisited:
                    x = queue[front]
                    front += 1
                    for y in range(n):
                        if trace_y[y] == unvisited:
                            w = c[x, y] - fx[x] - fy[y]
                            if w == 0:
                                trace_y[y] = x
                                delta_y[y] = delta_x[x]
                                if dy[y] < b[y]:
                                    finish = y
                                    delta_y[y] = np.minimum(delta_y[y], b[y] - dy[y])
                                    break
                                for xx in range(m):
                                    if (trace_x[xx] == unvisited and
                                                c[xx, y] == fx[xx] + fy[y] and
                                                d[xx, y]):
                                        trace_x[xx] = y
                                        delta_x[xx] = np.minimum(delta_y[y], d[xx, y])
                                        rear += 1
                                        queue[rear] = xx
                            if delta[y] > w:
                                delta[y] = w
                                arg[y] = x
                # </editor-fold>

                if finish == unvisited:
                    # <editor-fold desc="subX and addY">
                    inc_value = np.min(delta[np.where(trace_y == unvisited)[0]])
                    fx[np.where(trace_x != unvisited)[0]] += inc_value
                    fy[np.where(trace_y != unvisited)[0]] -= inc_value
                    delta[np.where(trace_y == unvisited)[0]] -= inc_value
                    for j in range(n):
                        if delta[j] == 0.0 and trace_y[j] == unvisited:
                            k = arg[j]
                            trace_y[j] = k
                            delta_y[j] = delta_x[k]
                            if dy[j] < b[j]:
                                finish = j
                                delta_y[j] = np.minimum(delta_y[j], b[j] - dy[j])
                                break
                            for ii in range(m):
                                if (trace_x[ii] == unvisited and
                                            c[ii, j] == fx[ii] + fy[j] and
                                        d[ii, j]):
                                    trace_x[ii] = j
                                    delta_x[ii] = np.minimum(delta_y[j], d[ii, j])
                                    rear += 1
                                    queue[rear] = ii
                                    # </editor-fold>

            # <editor-fold desc="Enlarge">
            inc_value = delta_y[finish]
            dy[finish] += inc_value
            while True:
                x = trace_y[finish]
                d[x, finish] += inc_value
                finish = trace_x[x]
                if finish == boundary:
                    break
                d[x, finish] -= inc_value
            dx[x] += inc_value
            # </editor-fold>

    return np.sum(c * d)
