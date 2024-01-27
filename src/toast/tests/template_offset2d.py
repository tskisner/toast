# Copyright (c) 2015-2020 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import numpy.testing as nt
from astropy import units as u

from .. import ops
from ..atm import available_atm
from ..observation import default_values as defaults
from ..pixels_io_wcs import write_wcs_fits
from ..templates import Offset, Offset2D
from ..templates.offset2d import plot as plot_offset2d
from ..utils import rate_from_times
from ._helpers import (
    close_data,
    create_fake_sky,
    create_ground_data,
    create_outdir,
    plot_wcs_maps,
)
from .mpi import MPITestCase


class TemplateOffset2DTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)
        # For debugging, change this to True
        self.write_extra = True

    def test_atm_reduce(self):
        if not available_atm:
            print("TOAST was built without atmosphere support, skipping Offset2D tests")
            return

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm, pixel_per_process=10, single_group=True)

        # Simple detector pointing
        detpointing_azel = ops.PointingDetectorSimple(
            boresight=defaults.boresight_azel, quats="quats_azel"
        )
        detpointing_radec = ops.PointingDetectorSimple(
            boresight=defaults.boresight_radec, quats="quats_radec"
        )

        # Create a noise model from focalplane detector properties
        default_model = ops.DefaultNoiseModel()
        default_model.apply(data)

        # Make an elevation-dependent noise model
        el_model = ops.ElevationNoise(
            noise_model="noise_model",
            out_model="el_weighted",
            detector_pointing=detpointing_azel,
        )
        el_model.apply(data)

        # Set up sky pixelization
        pixels = ops.PixelsWCS(
            detector_pointing=detpointing_radec,
            projection="CAR",
            resolution=(0.1 * u.degree, 0.1 * u.degree),
            auto_bounds=True,
            use_astropy=True,
        )
        weights = ops.StokesWeights(
            mode="IQU",
            detector_pointing=detpointing_radec,
        )

        # Build the pixel distribution
        pix_dist = ops.BuildPixelDistribution(
            pixel_dist="pixel_dist",
            pixel_pointing=pixels,
        )
        pix_dist.apply(data)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Write it out
        truthfile = os.path.join(self.outdir, f"input_sky.fits")
        write_wcs_fits(data["fake_map"], truthfile)
        if rank == 0:
            plot_wcs_maps(mapfile=truthfile)
        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        # Scan map into timestreams
        scanner = ops.Pipeline(
            operators=[
                pixels,
                weights,
                ops.ScanMap(
                    det_data=defaults.det_data,
                    pixels=pixels.pixels,
                    weights=weights.weights,
                    map_key="fake_map",
                ),
            ],
            detsets=["SINGLE"],
        )
        scanner.apply(data)

        # Simulate noise and accumulate to signal
        sim_noise = ops.SimNoise(noise_model=el_model.out_model)
        sim_noise.apply(data)

        # Simulate atmosphere signal and accumulate
        sim_atm = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            lmin_center=0.001 * u.meter,
            lmin_sigma=0.0 * u.meter,
            lmax_center=1.0 * u.meter,
            lmax_sigma=0.0 * u.meter,
            gain=6.0e-5,
            zatm=40000 * u.meter,
            zmax=200 * u.meter,
            xstep=10 * u.meter,
            ystep=10 * u.meter,
            zstep=10 * u.meter,
            nelem_sim_max=10000,
            wind_dist=3000 * u.meter,
            z0_center=2000 * u.meter,
            z0_sigma=0 * u.meter,
        )
        sim_atm.apply(data)

        sim_atm_coarse = ops.SimAtmosphere(
            detector_pointing=detpointing_azel,
            lmin_center=300 * u.meter,
            lmin_sigma=30 * u.meter,
            lmax_center=10000 * u.meter,
            lmax_sigma=1000 * u.meter,
            xstep=100 * u.meter,
            ystep=100 * u.meter,
            zstep=100 * u.meter,
            zmax=2000 * u.meter,
            nelem_sim_max=30000,
            gain=6.0e-4,
            realization=1000000,
            wind_dist=10000 * u.meter,
        )
        sim_atm_coarse.apply(data)

        if self.write_extra and rank == 0:
            import matplotlib.pyplot as plt

            ob = data.obs[0]
            det = ob.local_detectors[0]
            times = np.array(ob.shared[defaults.times])

            fig = plt.figure(figsize=(12, 8), dpi=72)
            ax = fig.add_subplot(1, 1, 1, aspect="auto")
            ax.plot(times, ob.detdata[defaults.det_data][det])
            ax.set_title(f"Detector {det} Atmosphere + Noise TOD")
            outfile = os.path.join(self.outdir, f"{det}_atm-noise_tod.pdf")
            plt.savefig(outfile)
            plt.close()

        # Make a binned map for comparison

        binner = ops.BinMap(
            pixel_pointing=pixels,
            stokes_weights=weights,
            noise_model=el_model.out_model,
        )

        mapmaker = ops.MapMaker(
            name="test_binned",
            binning=binner,
            solve_rcond_threshold=1.0e-3,
            map_rcond_threshold=1.0e-3,
            output_dir=self.outdir,
            reset_pix_dist=True,
        )
        mapmaker.apply(data)

        if rank == 0:
            hitfile = os.path.join(self.outdir, f"{mapmaker.name}_hits.fits")
            mapfile = os.path.join(self.outdir, f"{mapmaker.name}_map.fits")
            plot_wcs_maps(
                hitfile=hitfile,
                mapfile=mapfile,
                truth=truthfile,
                range_I=(-10.0, 10.0),
                range_Q=(-10.0, 10.0),
                range_U=(-10.0, 10.0),
            )

        # Now use the Offset2D template

        tmatrix = ops.TemplateMatrix(
            templates=[
                Offset2D(
                    det_data=defaults.det_data,
                    times=defaults.times,
                    detector_pointing=detpointing_azel,
                    noise_model=el_model.out_model,
                    resolution_azimuth=0.5 * u.degree,
                    resolution_elevation=0.5 * u.degree,
                    view="throw",
                    # max_step_time=10.0 * u.second,
                ),
            ]
        )

        mapmaker.name = "test_offset2d"
        mapmaker.template_matrix = tmatrix
        mapmaker.apply(data)

        amps = data[f"{mapmaker.name}_solve_amplitudes"]
        amp_file = os.path.join(self.outdir, f"{mapmaker.name}_offset2d.h5")
        tmatrix.templates[0].write(
            amps[tmatrix.templates[0].name], 
            amp_file,
        )

        if rank == 0:
            hitfile = os.path.join(self.outdir, f"{mapmaker.name}_hits.fits")
            mapfile = os.path.join(self.outdir, f"{mapmaker.name}_map.fits")
            plot_wcs_maps(
                hitfile=hitfile,
                mapfile=mapfile,
                truth=truthfile,
                range_I=(-5.0, 5.0),
                range_Q=(-5.0, 5.0),
                range_U=(-5.0, 5.0),
            )
            plot_offset2d(
                amp_file, 
                os.path.join(self.outdir, f"{mapmaker.name}_offset2d"),
            )
            

        # Now try only 1D offset

        tmatrix = ops.TemplateMatrix(
            templates=[
                Offset(
                    det_data=defaults.det_data,
                    times=defaults.times,
                    noise_model=el_model.out_model,
                    view="throw",
                ),
            ]
        )

        mapmaker.name = "test_offset1d"
        mapmaker.template_matrix = tmatrix
        mapmaker.apply(data)

        if rank == 0:
            hitfile = os.path.join(self.outdir, f"{mapmaker.name}_hits.fits")
            mapfile = os.path.join(self.outdir, f"{mapmaker.name}_map.fits")
            plot_wcs_maps(
                hitfile=hitfile,
                mapfile=mapfile,
                truth=truthfile,
                range_I=(-5.0, 5.0),
                range_Q=(-5.0, 5.0),
                range_U=(-5.0, 5.0),
            )
            

        # Now try combined 1D and 2D offsets

        tmatrix = ops.TemplateMatrix(
            templates=[
                Offset(
                    det_data=defaults.det_data,
                    times=defaults.times,
                    noise_model=el_model.out_model,
                    view="throw",
                ),
                Offset2D(
                    det_data=defaults.det_data,
                    times=defaults.times,
                    detector_pointing=detpointing_azel,
                    noise_model=el_model.out_model,
                    resolution_azimuth=0.5 * u.degree,
                    resolution_elevation=0.5 * u.degree,
                    view="throw",
                    # max_step_time=10.0 * u.second,
                ),
            ]
        )

        mapmaker.name = "test_both"
        mapmaker.template_matrix = tmatrix
        mapmaker.apply(data)

        amps = data[f"{mapmaker.name}_solve_amplitudes"]
        amp_file = os.path.join(self.outdir, f"{mapmaker.name}_offset2d.h5")
        tmatrix.templates[1].write(
            amps[tmatrix.templates[1].name], 
            amp_file,
        )

        if rank == 0:
            hitfile = os.path.join(self.outdir, f"{mapmaker.name}_hits.fits")
            mapfile = os.path.join(self.outdir, f"{mapmaker.name}_map.fits")
            plot_wcs_maps(
                hitfile=hitfile,
                mapfile=mapfile,
                truth=truthfile,
                range_I=(-5.0, 5.0),
                range_Q=(-5.0, 5.0),
                range_U=(-5.0, 5.0),
            )
            plot_offset2d(
                amp_file, 
                os.path.join(self.outdir, f"{mapmaker.name}_offset2d"),
            )

        close_data(data)
