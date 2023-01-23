# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
from collections import OrderedDict

import numpy as np
import traitlets
from astropy import units as u
from astropy.table import Column, QTable, Table

from .. import qarray as qa
from ..observation import default_values as defaults
from ..timing import function_timer
from ..traits import Bool, Int, Unicode, trait_docs, Quantity, Tuple
from ..utils import Environment, Logger
from .operator import Operator

from .pointing_model_utils import find_source, make_cal_map



@trait_docs
class PointingModelFit(Operator):
    """Reconstruction of focalplane geometry from a source observation.

    This operator takes data containing bright source observations and attempts
    to find the focalplane quaternion offset for each detector.

    """

    # Class traits

    API = Int(0, help="Internal interface version for this operator")

    source_lon = Quantity(
        None,
        allow_none=True,
        help="Point source longitude",
    )

    source_lat = Quantity(
        None,
        allow_none=True,
        help="Point source latitude",
    )

    view = Unicode(
        None, allow_none=True, help="Use this view of the data in all observations"
    )

    noise_model = Unicode(
        "noise_model", help="Observation key containing the noise model"
    )

    shared_flags = Unicode(
        defaults.shared_flags,
        allow_none=True,
        help="Observation shared key for telescope flags to use",
    )

    shared_flag_mask = Int(
        defaults.shared_mask_invalid, help="Bit mask value for optional flagging"
    )

    boresight = Unicode(
        defaults.boresight_radec, help="Observation shared key for boresight"
    )

    det_data = Unicode(
        defaults.det_data, help="Observation detdata key for the timestream data"
    )

    det_flags = Unicode(
        defaults.det_flags,
        allow_none=True,
        help="Observation detdata key for flags to use",
    )

    det_flag_mask = Int(
        defaults.det_mask_invalid | defaults.det_mask_processing,
        help="Bit mask value for optional detector flagging",
    )

    resolution = Tuple(
        (),
        help="The Lon/Lat projection resolution (Quantities)",
    )

    relcal_column = Unicode(
        "relcal", help="Column of the focalplane table for relative gain values"
    )

    state_column = Unicode(
        "pmodel", help="Column of focalplane table for pointing model fit state"
    )

    state_default_good = Bool(
        False, help="If True, when creating state column mark all dets as good"
    )

    debug_dir = Unicode(
        "pointing_model_out", help="Output directory for pointing model debug plots"
    )

    @traitlets.validate("shared_flag_mask")
    def _check_shared_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    @traitlets.validate("det_flag_mask")
    def _check_det_flag_mask(self, proposal):
        check = proposal["value"]
        if check < 0:
            raise traitlets.TraitError("Flag mask should be a positive integer")
        return check

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @function_timer
    def _exec(self, data, detectors=None, use_accel=False, **kwargs):
        log = Logger.get()

        # Set up mapmaking operators.

        if len(self.resolution) == 0:
            raise RuntimeError("You must specify the mapmaking projection resolution")

        if self.debug_dir is not None:
            if data.comm.group_rank == 0:
                if not os.path.isdir(self.debug_dir):
                    os.makedirs(self.debug_dir)
            if data.comm.comm_group is not None:
                data.comm.comm_group.barrier()

        # Get the superset of all detectors, and also build the list of
        # detectors which have an existing pointing solution.  Because we
        # do this in one pass through the data, we do not use the
        # data.all_local_detectors() method.  While passing through the
        # data, we also create the focalplane columns if they do not exist.

        all_dets = OrderedDict()
        slv_dets = OrderedDict()
        for ob in data.obs:
            fp = ob.telescope.focalplane
            n_rows = len(fp.detector_data)
            if self.relcal_column not in fp.detector_data.colnames:
                fp.detector_data.add_column(
                    Column(
                        name=self.relcal_column,
                        data=np.ones(n_rows, dtype=np.float64),
                    )
                )
            if self.state_column not in fp.detector_data.colnames:
                fp.detector_data.add_column(
                    Column(
                        name=self.state_column,
                        data=np.zeros(n_rows, dtype=np.uint8),
                    )
                )
                if self.state_default_good:
                    fp.detector_data[self.state_column][:] = 1
            dets = ob.select_local_detectors(selection=detectors)
            for d in dets:
                if d not in all_dets:
                    # First occurrence of this detector
                    all_dets[d] = None
                    slv_dets[d] = False
                if fp[d][self.state_column] != 0:
                    # We have at least one observation with a pointing solution
                    # for this detector
                    slv_dets[d] = True

        all_dets = list(all_dets.keys())
        solved_dets = list([x for x, y in slv_dets.items() if y])
        unsolved_dets = list([x for x, y in slv_dets.items() if not y])
        print(all_dets)
        print(solved_dets)

        # Make a map combining all detectors with an existing pointing
        # solution.

        wcs = None
        img_data = None
        if len(solved_dets) > 0:
            wcs, img_data = make_cal_map(
                "already_solved",
                data,
                solved_dets,
                self.resolution,
                self.view,
                self.boresight,
                self.det_data, 
                self.det_flags,
                self.det_flag_mask,
                self.shared_flags,
                self.shared_flag_mask,
                self.noise_model,
                poly_order=5,
                debug_dir=self.debug_dir,
            )

            # Find the source on one process
            src_lon = None
            src_lat = None
            if data.comm.group_rank == 0:
                src_debug = None
                if self.debug_dir is not None:
                    src_debug = os.path.join(self.debug_dir, "findpeaks_solved")
                    import matplotlib.pyplot as plt
                    fig = plt.figure(dpi=100, figsize=(8, 4))
                    ax = fig.add_subplot(1, 1, 1)
                    im = ax.imshow(
                        img_data, 
                        cmap="jet", 
                    )
                    fig.colorbar(im, orientation="vertical")
                    outfile = f"{src_debug}_input.pdf"
                    plt.savefig(
                        outfile, 
                        dpi=100, 
                        bbox_inches="tight", 
                        format="pdf"
                    )
                    plt.close()
                (src_img_x, src_img_y) = find_source(
                    img=img_data, 
                    debug_root=src_debug
                )
                src_lon, src_lat = wcs.wcs_pix2world(
                    np.array([[src_img_x, src_img_y]]),
                    0,
                )[0]
                print(f"source at {src_img_x}, {src_img_y} = {src_lon}, {src_lat}")
                # for i in range(-2, 3):
                #     for j in range(-2, 3):
                #         x = src_img_x + i
                #         y = src_img_y + j
                #         sn, st = wcs.wcs_pix2world(np.array([[x, y]]), 0)[0]
                #         print(f"{x}, {y} = {sn}, {st}")
            if data.comm.comm_group is not None:
                src_lon = data.comm.comm_group.bcast(src_lon, root=0)
                src_lat = data.comm.comm_group.bcast(src_lat, root=0)
        
        # Make single detector maps of unsolved detectors.

        for det in unsolved_dets:
            wcs, img_data = make_cal_map(
                det,
                data,
                [det,],
                self.resolution,
                self.view,
                self.boresight,
                self.det_data, 
                self.det_flags,
                self.det_flag_mask,
                self.shared_flags,
                self.shared_flag_mask,
                self.noise_model,
                poly_order=5,
                debug_dir=self.debug_dir,
            )

            # Find the source on one process
            src_lon = None
            src_lat = None
            if data.comm.group_rank == 0:
                src_debug = None
                if self.debug_dir is not None:
                    src_debug = os.path.join(self.debug_dir, f"findpeaks_{det}")
                    import matplotlib.pyplot as plt
                    fig = plt.figure(dpi=100, figsize=(8, 4))
                    ax = fig.add_subplot(1, 1, 1)
                    im = ax.imshow(
                        img_data, 
                        cmap="jet", 
                    )
                    fig.colorbar(im, orientation="vertical")
                    outfile = f"{src_debug}_input.pdf"
                    plt.savefig(
                        outfile, 
                        dpi=100, 
                        bbox_inches="tight", 
                        format="pdf"
                    )
                    plt.close()
                (src_img_x, src_img_y) = find_source(
                    img=img_data, 
                    debug_root=src_debug
                )
                src_lon, src_lat = wcs.wcs_pix2world(
                    np.array([[src_img_x, src_img_y]]),
                    0,
                )[0]
                print(f"{det} source at {src_img_x}, {src_img_y} = {src_lon}, {src_lat}")
                # for i in range(-2, 3):
                #     for j in range(-2, 3):
                #         x = src_img_x + i
                #         y = src_img_y + j
                #         sn, st = wcs.wcs_pix2world(np.array([[x, y]]), 0)[0]
                #         print(f"{x}, {y} = {sn}, {st}")
            if data.comm.comm_group is not None:
                src_lon = data.comm.comm_group.bcast(src_lon, root=0)
                src_lat = data.comm.comm_group.bcast(src_lat, root=0)

        # If a known source position is given, attempt to find a source in
        # the map.  Otherwise attempt to fit against the map made from
        # already-solved detectors.



        # for iob, ob in enumerate(data.obs):
        #     # Get the detectors we are using for this observation
        #     dets = ob.select_local_detectors(detectors)
        #     if len(dets) == 0:
        #         # Nothing to do for this observation
        #         continue

        #     # Focalplane to update
        #     focalplane = ob.telescope.focalplane

        #     # Data container with just this observation
        #     obs_data = data.select(obs_index=iob)

        #     # Make single detector maps
        #     for det in dets:
        #         mapper.name = f"cal_{det}"
        #         mapper.apply(obs_data, detectors=[det,])

        return

    def _finalize(self, data, **kwargs):
        return

    def _requires(self):
        req = {
            "meta": list(),
            "shared": [self.boresight],
            "detdata": [self.det_data],
            "intervals": list(),
        }
        if self.shared_flags is not None:
            req["shared"].append(self.shared_flags)
        if self.det_flags is not None:
            req["detdata"].append(self.det_flags)
        if self.view is not None:
            req["intervals"].append(self.view)
        return req

    def _provides(self):
        prov = {
            "meta": list(),
            "shared": list(),
            "detdata": [self.quats],
        }
        return prov
