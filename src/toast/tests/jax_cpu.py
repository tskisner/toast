# Copyright (c) 2024-2024 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import time

use_jax = False
try:
    import jax
    import jax.numpy as jnp
    import jax.lax as jlax
    use_jax = True
except:
    pass

import numpy as np
import numpy.testing as nt

from .._libtoast import (
    cov_accum_diag,
    cov_mult_diag,
    filter_polynomial,
)

from ..utils import dtype_to_aligned
from ..timing import Timer
from ._helpers import close_data, create_comm
from .mpi import MPITestCase


def jax_polyfilter(order, flags, signals, starts, stops):
    max_intr = max([x - y for x, y in zip(stops, starts)])
    out = list()

    n_samp = len(flags)
    n_bad = jnp.count_nonzero(flags)
    n_good = n_samp - n_bad
    n_order = min(order, n_good)
    dx = 2.0 / n_samp
    xstart = 0.5 * dx - 1
    orderinv = 1.0 / n_order

    @jax.jit
    def build_templates(ordr, ns, flgs):
        templ = jnp.zeros((ordr, ns))
        for iord in range(ordr):
            if iord == 0:
                templ = jlax.dynamic_update_slice(
                    templ,
                    jnp.ones(ns),
                    (0, 0),
                )
            elif iord == 1:
                templ = jlax.dynamic_update_slice(
                    templ,
                    xstart + dx * jnp.arange(ns),
                    (1, 0),
                )
            else:
                x = xstart + dx * jnp.arange(ns)
                row = orderinv * (
                    templ[iord - 1] * x * (2 * iord - 1) -
                    templ[iord - 2] * (iord - 1)
                )
                templ = jlax.dynamic_update_slice(
                    templ,
                    row,
                    (iord, 0),
                )
        # Apply flags
        templ = templ.at[:, flgs != 0].set(0)
        return templ


    templates = build_templates(n_order, n_samp, flags)

    # Build flagged signal array
    sigarray = jnp.vstack(signals)
    sigarray = sigarray.at[:, flags != 0].set(0)

    # Solve for filtered timestreams
    filtered = jax.scipy.linalg.solve(templates, sigarray.T).to_py()

    # Update inputs
    for isig, sig in enumerate(signals):
        sig[:] = filtered[isig]


class JaxCpuTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        # self.outdir = create_outdir(self.comm, fixture_name)
        self.rank = 0
        self.nproc = 1
        if self.comm is not None:
            self.rank = self.comm.rank
            self.nproc = self.comm.size

    def test_polyfilter(self):
        if not (use_jax):
            if self.rank == 0:
                print("Not running with JAX support- skipping memory test")
            return
        print(jax.devices())
        order = 9
        nsamp = 5000000
        nsig = 5
        nintr = 10
        starts = list()
        stops = list()
        cur = nsamp // nintr
        off = 0
        for intr in range(nintr):
            starts.append(off)
            if off + cur >= nsamp:
                # Last interval
                stops.append(nsamp)
                break
            else:
                stops.append(off + cur)
            off += cur
            cur += 1
        starts = np.array(starts, dtype=np.int64)
        stops = np.array(stops, dtype=np.int64)
        flags = np.zeros(nsamp, dtype=np.uint8)
        sigs_compiled = list()
        sigs_jax = list()
        for isig in range(nsig):
            sdata = np.random.normal(loc=isig, scale=isig, size=nsamp)
            sigs_compiled.append(sdata)
            sigs_jax.append(np.array(sdata))

        tm = Timer()
        tm.start()
        filter_polynomial(order, flags, sigs_compiled, starts, stops, False)
        tm.report_clear("filter_polynomial: libtoast")
        jax_polyfilter(order, flags, sigs_jax, starts, stops)
        tm.report_clear("filter_polynomial: jax")
        tm.stop()


    def test_covariance(self):
        if not (use_jax):
            if self.rank == 0:
                print("Not running with JAX support- skipping memory test")
            return
        pass
