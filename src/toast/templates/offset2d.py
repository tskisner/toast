# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

from collections import OrderedDict

import h5py
import numpy as np
import scipy
import scipy.signal
import traitlets
from astropy import units as u
from astropy.table import Column, QTable
from astropy.io.misc.hdf5 import read_table_hdf5, write_table_hdf5

from .._libtoast import (
    template_offset_add_to_signal,
    template_offset_apply_diag_precond,
    template_offset_project_signal,
)
from ..data import Data
from ..dist import DistRange
from ..intervals import IntervalList
from ..mpi import MPI
from ..observation import default_values as defaults
from ..ops.operator import Operator
from ..pointing_utils import scan_range_lonlat
from ..timing import function_timer
from ..traits import Bool, Float, Instance, Int, Quantity, Unicode, trait_docs
from ..utils import AlignedF64, Logger, rate_from_times
from ..vis import set_matplotlib_backend
from .. import qarray as qa
from .amplitudes import Amplitudes
from .template import Template


@trait_docs
class Offset2D(Template):
    """This class represents 2D spatial fluctuations in Az/El.

    This template is a grid of amplitudes representing the spatial atmosphere amplitudes
    for grid common to all detectors for a limited span of time.  The grid resolution
    is specified as well as the intervals to use for the chunking in time.  Note that
    the resolution in azimuth is specified at the horizon and is scaled by the cosine
    of the central elevation value.

    For experiments making large throws in azimuth, the atmosphere will likely be
    different between sweeps and the intervals representing these throws should be
    used.  For instruments tracking a field and scanning in Az and El, a maximum time
    step should be specified that corresponds to the shorter of: the time for the patch
    to drift substantially in Az/El, OR the typical time for the atmosphere to change
    (due to wind) across the field.

    """

    # Notes:  The TraitConfig base class defines a "name" attribute.  The Template
    # class (derived from TraitConfig) defines the following traits already:
    #    data             : The Data instance we are working with
    #    view             : The timestream view we are using
    #    det_data         : The detector data key with the timestreams
    #    det_data_units   : The units of the detector data
    #    det_flags        : Optional detector solver flags
    #    det_flag_mask    : Bit mask for detector solver flags
    #

    resolution_azimuth = Quantity(
        0.5 * u.degree,
        help="The approximate angular width of a grid cell in azimuth",
    )

    resolution_elevation = Quantity(
        0.5 * u.degree,
        help="The approximate angular width of a grid cell in elevation",
    )

    max_step_time = Quantity(600.0 * u.second, help="Maximum time step")

    times = Unicode(defaults.times, help="Observation shared key for timestamps")

    detector_pointing = Instance(
        klass=Operator,
        allow_none=True,
        help="Operator that translates boresight Az/El pointing into detector frame",
    )

    noise_model = Unicode(
        None,
        allow_none=True,
        help="Observation key with noise model for weighting.",
    )

    min_hits = Int(
        10,
        help="The minimum number of hits in a grid cell to keep the cell amplitude",
    )

    field_of_view = Quantity(
        None,
        allow_none=True,
        help="Override the focalplane field of view",
    )

    @traitlets.validate("detector_pointing")
    def _check_detector_pointing(self, proposal):
        detpointing = proposal["value"]
        if detpointing is not None:
            if not isinstance(detpointing, Operator):
                raise traitlets.TraitError(
                    "detector_pointing should be an Operator instance"
                )
            # Check that this operator has the traits we expect
            for trt in [
                "view",
                "boresight",
                "shared_flags",
                "shared_flag_mask",
                "quats",
            ]:
                if not detpointing.has_trait(trt):
                    msg = f"detector_pointing operator should have a '{trt}' trait"
                    raise traitlets.TraitError(msg)
        return detpointing

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def clear(self):
    #     """Delete the underlying C-allocated memory."""
    #     if hasattr(self, "_offsetvar"):
    #         del self._offsetvar
    #     if hasattr(self, "_offsetvar_raw"):
    #         self._offsetvar_raw.clear()
    #         del self._offsetvar_raw

    # def __del__(self):
    #     self.clear()

    @function_timer
    def _initialize(self, new_data):
        log = Logger.get()
        self._all_dets = set()

        comm = new_data.comm

        # Build the global list of all sessions
        data_sessions = new_data.split(obs_session_name=True, require_full=True)

        local_sessions = list(data_sessions.keys())
        all_sessions = None
        if comm.comm_group_rank is not None:
            if comm.group_rank == 0:
                psessions = comm.comm_group_rank.gather(local_sessions, root=0)
                if comm.group == 0:
                    # Build up the list of sessions ordered by first occurrance.
                    # This should be a small number of strings and a relatively
                    # small number of process groups, so no need to use a set.
                    all_sessions = list()
                    for ps in psessions:
                        for s in ps:
                            if s not in all_sessions:
                                all_sessions.append(s)
                all_sessions = comm.comm_group_rank.bcast(all_sessions, root=0)
            all_sessions = comm.comm_group.bcast(all_sessions, root=0)
        else:
            # There is only one group so all processes have the list of sessions
            all_sessions = local_sessions
        # print(f"{comm.world_rank}:  all_sessions = {all_sessions}")

        self._local_sessions = local_sessions
        self._all_sessions = all_sessions

        # Every group checks their interval list (view) for each observation and
        # splits it as needed to keep the maximum time step below the requested value.

        session_intervals = dict()
        for sname, sdata in data_sessions.items():
            check_intervals = None
            for ob in sdata.obs:
                if sname not in session_intervals:
                    # First observation in this session
                    samplespans = list()
                    for ivw, vw in enumerate(ob.intervals[self.view]):
                        n_split = 1
                        time_diff = (
                            ob.shared[self.times][vw.last]
                            - ob.shared[self.times][vw.first]
                        )
                        tdiff = time_diff
                        while tdiff > self.max_step_time.to_value(u.s):
                            n_split += 1
                            tdiff = time_diff / n_split
                        if n_split == 1:
                            samplespans.append((vw.first, vw.last))
                        else:
                            incr = (vw.last - vw.first + 1) // n_split
                            off = vw.first
                            for ic in range(n_split - 1):
                                samplespans.append((off, off + incr - 1))
                                off += incr
                            samplespans.append((off, vw.last))
                    #print(f"View {self.view} --> {samplespans}")
                    session_intervals[sname] = IntervalList(
                        timestamps=ob.shared[self.times],
                        samplespans=samplespans,
                    )
                    check_intervals = IntervalList(
                        timestamps=ob.shared[self.times],
                        intervals=ob.intervals[self.view],
                    )
                else:
                    # Check that the original interval lists agree between
                    # observations in the same session.
                    if ob.intervals[self.view] != check_intervals:
                        msg = f"Interval lists '{self.view}' disagree between obs "
                        msg += f"{ob.name} and {sdata.obs[0].name} in the same session"
                        raise RuntimeError(msg)

        # Every group finds the scan range for its session data

        session_ranges = dict()
        for sname, sdata in data_sessions.items():
            session_ranges[sname] = dict()
            for ob in sdata.obs:
                self._all_dets.update(ob.local_detectors)
                for ivw, vw in enumerate(session_intervals[sname]):
                    vslice = slice(vw.first, vw.first + vw.last + 1, 1)
                    azmin, azmax, elmin, elmax = scan_range_lonlat(
                        ob,
                        self.detector_pointing.boresight,
                        flags=self.detector_pointing.shared_flags,
                        flag_mask=self.detector_pointing.shared_flag_mask,
                        # is_azimuth=True,
                        samples=vslice,
                        field_of_view=self.field_of_view,
                    )
                    print(f"proc {ob.comm.group_rank}: {sname}:{vw.first}-{vw.last} {np.degrees(azmin)}, {np.degrees(azmax)}, {np.degrees(elmin)}, {np.degrees(elmax)}")
                    if ivw in session_ranges[sname]:
                        # Update the range extent for this view
                        azmin = min(session_ranges[sname][ivw]["azmin"], azmin)
                        azmax = max(session_ranges[sname][ivw]["azmax"], azmax)
                        elmin = min(session_ranges[sname][ivw]["elmin"], elmin)
                        elmax = min(session_ranges[sname][ivw]["elmax"], elmax)
                    session_ranges[sname][ivw] = {
                        "azmin": azmin,
                        "azmax": azmax,
                        "elmin": elmin,
                        "elmax": elmax,
                        "sample_off": vw.first,
                        "n_sample": vw.last - vw.first + 1,
                    }

        # Compute the maximum extent for each session and interval across all
        # groups.  If there is only one group, then nothing needed.
        if comm.comm_group_rank is not None:
            full_ranges = None
            if comm.group_rank == 0:
                full_ranges = dict()
                for ses in all_sessions:
                    props = None
                    if ses in session_ranges:
                        props = session_ranges[ses]
                    sranges = comm.comm_group_rank.gather(props, root=0)
                    if comm.group == 0:
                        full_ranges[ses] = dict()
                        for pr in sranges:
                            if pr is None:
                                continue
                            for ivw, vwprops in pr.items():
                                if ivw in full_ranges[ses]:
                                    azmin = min(
                                        full_ranges[ses][ivw]["azmin"], vwprops["azmin"]
                                    )
                                    azmax = max(
                                        full_ranges[ses][ivw]["azmax"], vwprops["azmax"]
                                    )
                                    elmin = min(
                                        full_ranges[ses][ivw]["elmin"], vwprops["elmin"]
                                    )
                                    elmax = max(
                                        full_ranges[ses][ivw]["elmax"], vwprops["elmax"]
                                    )
                                    full_ranges[ses][ivw]["azmin"] = azmin
                                    full_ranges[ses][ivw]["azmax"] = azmax
                                    full_ranges[ses][ivw]["elmin"] = elmin
                                    full_ranges[ses][ivw]["elmax"] = elmax
                                else:
                                    full_ranges[ses][ivw] = vwprops
                                
                                azmin = full_ranges[ses][ivw]["azmin"]
                                azmax = full_ranges[ses][ivw]["azmax"]
                                elmin = full_ranges[ses][ivw]["elmin"]
                                elmax = full_ranges[ses][ivw]["elmax"]
                    comm.comm_group_rank.barrier()
                full_ranges = comm.comm_group_rank.bcast(full_ranges, root=0)
            session_ranges = comm.comm_group.bcast(full_ranges, root=0)

        print(f"{comm.world_rank}:  session_ranges = {session_ranges}")

        # Now every process has the Az / El range for every time slice of every
        # session.  Each process computes the grid spacing and number of local
        # and global amplitudes.

        amp_off = 0
        local_amp_off = 0
        self._local_ranges = list()
        for sname, sprops in session_ranges.items():
            for ivw, vprops in sprops.items():
                azmin = vprops["azmin"].to_value(u.rad)
                azmax = vprops["azmax"].to_value(u.rad)
                elmin = vprops["elmin"].to_value(u.rad)
                elmax = vprops["elmax"].to_value(u.rad)

                print(f"{sname} {ivw}: {np.degrees(azmin)} {np.degrees(azmax)} {np.degrees(elmin)} {np.degrees(elmax)}")
                
                azcenter = 0.5 * (azmin + azmax)
                elcenter = 0.5 * (elmin + elmax)
                
                # print(f"{sname} {ivw}:  center {np.degrees(azcenter)} {np.degrees(elcenter)}")
                
                elincr = self.resolution_elevation.to_value(u.radian)
                azincr = self.resolution_azimuth.to_value(u.radian) / np.cos(elcenter)
                print(f"{sname} {ivw}: elincr = {np.degrees(elincr)} azincr = {np.degrees(azincr)}")

                # Round the resolutions to fit evenly between min / max values
                n_az = int(0.5 + (azmax - azmin) / azincr)
                n_el = int(0.5 + (elmax - elmin) / elincr)
                if n_az == 0:
                    n_az = 1
                if n_el == 0:
                    n_el = 1
                azincr = (azmax - azmin) / n_az
                elincr = (elmax - elmin) / n_el
                # print(f"{sname} {ivw}:  incr {np.degrees(elincr)} {np.degrees(azincr)}")

                # Increase the grid by a buffer of cells around the border
                n_az += 2
                n_el += 2
                azmin -= azincr
                azmax += azincr
                elmin -= elincr
                elmax += elincr

                # print(f"{sname} {ivw}: {np.degrees(azmin)} {np.degrees(azmax)} {np.degrees(elmin)} {np.degrees(elmax)}")

                # print(f"{sname} {ivw}:  n_az={n_az} n_el={n_el}")
                vprops["n_az"] = n_az
                vprops["n_el"] = n_el
                vprops["amp_off"] = amp_off
                vprops["n_amp"] = n_az * n_el
                vprops["azinc_rad"] = azincr
                vprops["elinc_rad"] = elincr
                # print(
                #     f"{sname} {ivw}:  incr_rad {vprops['azinc_rad']} {vprops['elinc_rad']}"
                # )
                if sname in data_sessions:
                    vprops["local_amp_off"] = local_amp_off
                    self._local_ranges.append((amp_off, vprops["n_amp"]))
                    local_amp_off += vprops["n_amp"]
                amp_off += vprops["n_amp"]

        # print(f"{comm.world_rank}:  session_ranges = {session_ranges}")

        self._session_ranges = session_ranges
        self._n_global = amp_off
        self._n_local = local_amp_off
        self._n_ranges = len(self._local_ranges)
        self._all_dets = list(sorted(self._all_dets))

        # Flagging.  Every process expands its local detector pointing and uses
        # the local amplitudes to accumulate the hits.  This is then reduced and
        # used to flag amplitude values based on the minimum required hits.

        self._amp_flags = np.zeros(self._n_local, dtype=bool)
        hits = self._zeros()

        for iob, ob in enumerate(self.data.obs):
            sname = ob.session.name
            det_ones = np.ones(ob.n_local_samples)
            dets = ob.select_local_detectors()
            if len(dets) == 0:
                continue
            for det in dets:
                # Expand pointing
                subdata = self.data.select(obs_index=iob)
                self.detector_pointing.apply(
                    subdata,
                    detectors=[det],
                )
                for ivw, vprops in self._session_ranges[sname].items():
                    slc = slice(
                        vprops["sample_off"], 
                        vprops["sample_off"] + vprops["n_sample"], 
                        1,
                    )
                    amp_index = self._amp_indices(
                        vprops,
                        slc,
                        ob.detdata[self.detector_pointing.quats][det, :, :],
                    )
                    np.add.at(
                        hits.local,
                        amp_index, 
                        det_ones[slc],
                    )
        hits.sync()

        for sname, sprops in self._session_ranges.items():
            if sname not in data_sessions:
                # This process does not have this session locally
                continue
            for ivw, vprops in sprops.items():
                n_az = vprops["n_az"]
                n_el = vprops["n_el"]
                amp_off = vprops["local_amp_off"]
                n_grid = n_az * n_el
                off = 0
                for iel in range(n_el):
                    for iaz in range(n_az):
                        amp_index = amp_off + iel * n_az + iaz
                        if hits.local[amp_index] < self.min_hits:
                            self._amp_flags[amp_index] = True
        del hits
        # print(
        #     f"{comm.world_rank}:  {self._n_global} amplitudes ({self._n_local} local)"
        # )

    def _detectors(self):
        return self._all_dets

    def _zeros(self):
        z = Amplitudes(
            self.data.comm, 
            self._n_global, 
            self._n_local, 
            local_ranges=self._local_ranges,
        )
        z.local_flags[:] = np.where(self._amp_flags, 1, 0)
        return z

    def _amp_indices(self, props, slc, quats):
        # print(f"DBG slice: {slc}")
        # print(f"DBG quats: {quats[slc, :]}")
        det_az, det_el, _ = qa.to_lonlat_angles(quats[slc, :])
        # print(f"DBG az / el: {det_az} / {det_el}")
        n_az = props["n_az"]
        azmin = props["azmin"].to_value(u.rad)
        elmin = props["elmin"].to_value(u.rad)
        azinc = props["azinc_rad"]
        elinc = props["elinc_rad"]
        # print(
        #     f"DBG n_az={n_az} azmin={azmin} elmin={elmin} azinc={azinc} elinc={elinc}"
        # )
        amp_off = props["local_amp_off"]
        grid_az = np.trunc((det_az - azmin) / azinc).astype(np.int32)
        grid_el = np.trunc((det_el - elmin) / elinc).astype(np.int32)
        # print(f"DBG grid_az = {grid_az}")
        # print(f"DBG grid_el = {grid_el}")
        return amp_off + grid_el * n_az + grid_az

    @function_timer
    def _add_to_signal(self, detector, amplitudes, **kwargs):
        for iob, ob in enumerate(self.data.obs):
            sname = ob.session.name
            dets = set(ob.select_local_detectors())
            if detector not in dets:
                continue
            if detector not in ob.detdata[self.det_data].detectors:
                continue
            # Expand pointing
            subdata = self.data.select(obs_index=iob)
            self.detector_pointing.apply(
                subdata,
                detectors=[
                    detector,
                ],
            )
            # FIXME:  Use flags here
            # print(f"add_to_signal {detector} for {ob.n_local_samples} local samples")
            for ivw, vprops in self._session_ranges[sname].items():
                slc = slice(
                    vprops["sample_off"], vprops["sample_off"] + vprops["n_sample"], 1
                )
                amp_index = self._amp_indices(
                    vprops,
                    slc,
                    ob.detdata[self.detector_pointing.quats][detector, :, :],
                )
                ob.detdata[self.det_data][detector, slc] += amplitudes.local[amp_index]

    @function_timer
    def _project_signal(self, detector, amplitudes, **kwargs):
        for iob, ob in enumerate(self.data.obs):
            sname = ob.session.name
            dets = set(ob.select_local_detectors())
            if detector not in dets:
                continue
            if detector not in ob.detdata[self.det_data].detectors:
                continue
            # Expand pointing
            subdata = self.data.select(obs_index=iob)
            self.detector_pointing.apply(
                subdata,
                detectors=[
                    detector,
                ],
            )
            # print(f"project_signal {detector} for {ob.n_local_samples} local samples")
            # print(
            #     f"quats shape = {ob.detdata[self.detector_pointing.quats].data.shape}"
            # )
            for ivw, vprops in self._session_ranges[sname].items():
                slc = slice(
                    vprops["sample_off"], vprops["sample_off"] + vprops["n_sample"], 1
                )
                amp_index = self._amp_indices(
                    vprops,
                    slc,
                    ob.detdata[self.detector_pointing.quats][detector, :, :],
                )
                np.add.at(
                    amplitudes.local, 
                    amp_index, 
                    ob.detdata[self.det_data][detector, slc],
                )

    @function_timer
    def _add_prior(self, amplitudes_in, amplitudes_out, **kwargs):
        return

    @function_timer
    def _apply_precond(self, amplitudes_in, amplitudes_out, **kwargs):
        amplitudes_out.local[:] = amplitudes_in.local[:]

    @function_timer
    def write(self, amplitudes, out):
        """Write out amplitude values.

        This stores the amplitudes to a file for debugging / plotting.

        Args:
            amplitudes (Amplitudes):  The amplitude data.
            out (str):  The output file.

        Returns:
            None

        """
        if self._local_sessions != self._all_sessions:
            raise NotImplementedError(
                "Can only write amplitudes from jobs with one process group"
            )

        if self.data.comm.group_rank == 0:
            with h5py.File(out, "w") as hf:
                for sname, sprops in self._session_ranges.items():
                    hg = hf.create_group(sname)
                    for ivw, vprops in sprops.items():
                        samp_off = vprops["sample_off"]
                        n_samp = vprops["n_sample"]
                        last_samp = samp_off + n_samp - 1
                        dname = f"{samp_off:09d}_{last_samp:09d}"
                        n_az = vprops["n_az"]
                        n_el = vprops["n_el"]
                        azmin = vprops["azmin"].to_value(u.deg)
                        elmin = vprops["elmin"].to_value(u.deg)
                        azinc = vprops["azinc_rad"] * 180.0 / np.pi
                        elinc = vprops["elinc_rad"] * 180.0 / np.pi
                        amp_off = vprops["local_amp_off"]
                        n_grid = n_az * n_el
                        amp_data = QTable(
                            [
                                Column(name="az_index", length=n_grid, dtype=np.int32),
                                Column(name="el_index", length=n_grid, dtype=np.int32),
                                Column(name="az_center", length=n_grid, unit=u.deg),
                                Column(name="el_center", length=n_grid, unit=u.deg),
                                Column(name="value", length=n_grid, dtype=np.float64),
                                Column(name="flag", length=n_grid, dtype=np.uint8),
                            ]
                        )
                        off = 0
                        for iel in range(n_el):
                            cent_el = elmin + (iel + 0.5) * elinc
                            for iaz in range(n_az):
                                cent_az = azmin + (iaz + 0.5) * azinc
                                amp_index = amp_off + iel * n_az + iaz
                                amp_data[off]["az_index"] = iaz
                                amp_data[off]["el_index"] = iel
                                amp_data[off]["az_center"] = cent_az * u.degree
                                amp_data[off]["el_center"] = cent_el * u.degree
                                amp_data[off]["value"] = amplitudes.local[amp_index]
                                amp_data[off]["flag"] = amplitudes.local_flags[amp_index]
                                off += 1

                        amp_data.meta["n_az"] = n_az
                        amp_data.meta["n_el"] = n_el
                        amp_data.meta["azmin"] = azmin * u.degree
                        amp_data.meta["azmax"] = (n_az * azinc + azmin) * u.degree
                        amp_data.meta["azinc"] = azinc * u.degree
                        amp_data.meta["elmin"] = elmin * u.degree
                        amp_data.meta["elmax"] = (n_el * elinc + elmin) * u.degree
                        amp_data.meta["elinc"] = elinc * u.degree
                        write_table_hdf5(amp_data, hg, dname)


def plot(amp_file, out_root=None):
    """Plot an amplitude dump file.
    
    Args:
        amp_file (str):  The path to the input file of amplitudes.
        out_root (str):  The root of the output files.

    Returns:
        None
    
    """

    if out_root is not None:
        set_matplotlib_backend(backend="pdf")

    import matplotlib.pyplot as plt

    figdpi = 100

    with h5py.File(amp_file, "r") as hf:
        for gname, grp in hf.items():
            for dname, dset in grp.items():
                amp_data = read_table_hdf5(grp, path=dname)
                meta = amp_data.meta
                n_az = meta["n_az"]
                n_el = meta["n_el"]
                imarray = np.zeros((n_el, n_az), dtype=np.float64)
                for row in amp_data:
                    x = row["az_index"]
                    y = row["el_index"]
                    z = row["value"]
                    if row["flag"] != 0:
                        z = np.nan
                    imarray[y, x] = z
                #print(imarray)
                fig = plt.figure(dpi=figdpi, figsize=(8, 4))
                ax = fig.add_subplot(1, 1, 1)
                im = ax.imshow(
                    imarray, 
                    cmap="jet", 
                    extent=(
                        meta["azmin"], 
                        meta["azmax"],
                        meta["elmin"],
                        meta["elmax"],
                    ),
                )
                fig.colorbar(im, orientation="vertical")

                if out_root is None:
                    plt.show()
                else:
                    outfile = f"{out_root}_{gname}_{dname}.pdf"
                    plt.savefig(
                        outfile, 
                        dpi=figdpi, 
                        bbox_inches="tight", 
                        format="pdf"
                    )
                    plt.close()
