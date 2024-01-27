# Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
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


class PointingModelFitTest(MPITestCase):
    def setUp(self):
        fixture_name = os.path.splitext(os.path.basename(__file__))[0]
        self.outdir = create_outdir(self.comm, fixture_name)
        np.random.seed(123456)
        # For debugging, change this to True
        self.write_extra = True

    def create_cal_data(self):
        """Create calibration data"""
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data = create_ground_data(self.comm, pixel_per_process=10, single_group=True)
        fwhm = data.obs[0].telescope.focalplane.detector_data["fwhm"][0]

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

        # Set up sky pixelization- use high resolution.  Also
        # use one submap so that every process has the full
        # map data.
        pix_per_fwhm = 20.0
        sim_res = fwhm / pix_per_fwhm
        pixels = ops.PixelsWCS(
            detector_pointing=detpointing_radec,
            projection="CAR",
            resolution=(sim_res, sim_res),
            submaps=1,
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
        print(f"Simulating sky with {pixels.wcs_shape} pixels", flush=True)

        # Create fake polarized sky pixel values locally
        create_fake_sky(data, "pixel_dist", "fake_map")

        # Get the WCS shape of the projection
        wcs_shape = pixels.wcs_shape

        print(f"Simulating fake beam convolved gaussian", flush=True)
        # Insert a symmetric gaussian at the center of the projection.
        # We assume a point source convolved with the detector beam
        background = np.amax(data["fake_map"].data[:, :, 0])
        src_mag = 5000.0 * background
        coeff = src_mag / (pix_per_fwhm**2 * 2.0 * np.pi)
        n_dim = int(3.0 * pix_per_fwhm)
        half = n_dim // 2
        off_x = wcs_shape[0] // 2
        off_y = wcs_shape[1] // 2
        source_lon, source_lat = pixels.wcs.wcs_pix2world(
            np.array([[off_x, off_y]]), 0
        )[0]
        print(f"Source at {off_x}, {off_y} == {source_lon}, {source_lat}")
        for x in range(n_dim):
            for y in range(n_dim):
                xpix = off_x - half + x
                ypix = off_y - half + y
                xdelt = (xpix - off_x)**2
                ydelt = (ypix - off_y)**2
                xterm = float(xdelt) / (2.0 * pix_per_fwhm**2)
                yterm = float(ydelt) / (2.0 * pix_per_fwhm**2)
                pix = xpix * wcs_shape[1] + ypix
                sval = coeff * np.exp(-(xterm + yterm))
                # print(f"Accum {sval} to pix {xpix}, {ypix}")
                data["fake_map"].data[0, pix, 0] += sval

        print(f"writing truth map", flush=True)
        # Write it out
        truthfile = os.path.join(self.outdir, f"input_sky.fits")
        write_wcs_fits(data["fake_map"], truthfile)
        if self.write_extra and rank == 0:
            plot_wcs_maps(mapfile=truthfile)
        if data.comm.comm_world is not None:
            data.comm.comm_world.barrier()

        print(f"Scan map to tod", flush=True)
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

        print(f"Simulating noise", flush=True)
        # Simulate noise and accumulate to signal
        # sim_noise = ops.SimNoise(noise_model=el_model.out_model)
        # sim_noise.apply(data)

        # print(f"Simulating atmosphere", flush=True)
        # # Simulate atmosphere signal and accumulate
        # sim_atm = ops.SimAtmosphere(
        #     detector_pointing=detpointing_azel,
        #     lmin_center=0.001 * u.meter,
        #     lmin_sigma=0.0 * u.meter,
        #     lmax_center=1.0 * u.meter,
        #     lmax_sigma=0.0 * u.meter,
        #     gain=6.0e-5,
        #     zatm=40000 * u.meter,
        #     zmax=200 * u.meter,
        #     xstep=10 * u.meter,
        #     ystep=10 * u.meter,
        #     zstep=10 * u.meter,
        #     nelem_sim_max=10000,
        #     wind_dist=3000 * u.meter,
        #     z0_center=2000 * u.meter,
        #     z0_sigma=0 * u.meter,
        # )
        # sim_atm.apply(data)

        # print(f"Simulating coarse atm", flush=True)
        # sim_atm_coarse = ops.SimAtmosphere(
        #     detector_pointing=detpointing_azel,
        #     lmin_center=300 * u.meter,
        #     lmin_sigma=30 * u.meter,
        #     lmax_center=10000 * u.meter,
        #     lmax_sigma=1000 * u.meter,
        #     xstep=100 * u.meter,
        #     ystep=100 * u.meter,
        #     zstep=100 * u.meter,
        #     zmax=2000 * u.meter,
        #     nelem_sim_max=30000,
        #     gain=6.0e-4,
        #     realization=1000000,
        #     wind_dist=10000 * u.meter,
        # )
        # sim_atm_coarse.apply(data)

        # Clear temp objects
        for ob in data.obs:
            del ob.detdata[detpointing_azel.quats]
            del ob.detdata[detpointing_radec.quats]
            del ob.detdata[pixels.pixels]
            del ob.detdata[weights.weights]
        del data["pixel_dist"]

        # print(f"Writing TOD plot", flush=True)

        # if self.write_extra:
        #     import matplotlib.pyplot as plt

        #     for ob in data.obs:
        #         times = np.array(ob.shared[defaults.times].data)
        #         for det in ob.local_detectors:
        #             outfile = os.path.join(
        #                 self.outdir, 
        #                 f"{ob.name}_{det}_tod.pdf"
        #             )
        #             fig = plt.figure(figsize=(12, 8), dpi=72)
        #             ax = fig.add_subplot(1, 1, 1, aspect="auto")
        #             ax.plot(times, ob.detdata[defaults.det_data][det])
        #             ax.set_title(f"Detector {det} Atmosphere + Noise + Source TOD")
        #             plt.savefig(outfile)
        #             plt.close()
        return data, (source_lon, source_lat)
    
    def test_gauss_fit(self):
        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create an input gaussian
        nx = 1000
        ny = 1000
        lon_min = - np.pi / 8
        lon_max = np.pi / 8
        lat_min = - np.pi / 8
        lat_max = np.pi / 8
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        sigma_major = 0.05 * lon_range
        sigma_minor = 0.02 * lat_range
        center_lon = 0.5 * (lon_min + lon_max) + 3 * sigma_major
        center_lat = 0.5 * (lat_min + lat_max) + 3 * sigma_minor
        rot = np.pi / 8
        img = ops.pointing_model_utils.evaluate_gaussian(
            nx,
            ny,
            lon_min, 
            lon_max, 
            lat_min, 
            lat_max, 
            center_lon, 
            center_lat,
            sigma_major,
            sigma_minor,
            rot,
            1.0,
        )
        if rank == 0:
            import matplotlib.pyplot as plt
            fig = plt.figure(dpi=100, figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(
                img, 
                cmap="jet",
                origin="lower", 
            )
            fig.colorbar(im, orientation="vertical")
            outfile = os.path.join(self.outdir, f"gauss_fit_input.pdf")
            plt.savefig(
                outfile, 
                dpi=100, 
                bbox_inches="tight", 
                format="pdf"
            )
            plt.close()
        
        # Fit for it
        fit = ops.pointing_model_utils.fit_gaussian(
            img, 
            lon_min, 
            lon_max, 
            lat_min, 
            lat_max,
            center_lon,
            center_lat,
            0.5 * (sigma_major + sigma_minor),
        )
        print(fit)

        if rank == 0:
            import matplotlib.pyplot as plt

            best = ops.pointing_model_utils.evaluate_gaussian(
                nx,
                ny, 
                lon_min, 
                lon_max, 
                lat_min, 
                lat_max, 
                fit["center_lon"], 
                fit["center_lat"],
                fit["sigma_major"],
                fit["sigma_minor"],
                fit["angle"],
                fit["amplitude"],
            )

            fig = plt.figure(dpi=100, figsize=(6, 4))
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(
                best, 
                cmap="jet",
                origin="lower", 
            )
            fig.colorbar(im, orientation="vertical")
            outfile = os.path.join(self.outdir, f"gauss_fit_result.pdf")
            plt.savefig(
                outfile, 
                dpi=100, 
                bbox_inches="tight", 
                format="pdf"
            )
            plt.close()
        

    def test_source_fit(self):
        if not available_atm:
            print(
                "No atmosphere sim support, skipping pointing model tests"
            )
            return

        rank = 0
        if self.comm is not None:
            rank = self.comm.rank

        # Create fake observing of a small patch
        data, (source_lon, source_lat) = self.create_cal_data()

        # Fit to the source
        fitter = ops.PointingModelFit(
            source_lon=source_lon * u.degree,
            source_lat=source_lat * u.degree,
            noise_model="el_weighted",
            resolution=(0.05 * u.degree, 0.05 * u.degree),
            debug_dir=os.path.join(self.outdir, "pointing_model"),
            state_default_good=False,
        )
        fitter.apply(data)

        print(data)


        close_data(data)
