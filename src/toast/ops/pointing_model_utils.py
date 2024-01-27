# Copyright (c) 2023-2023 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np
from astropy.wcs import WCS
from scipy.optimize import curve_fit, least_squares, Bounds

from ..mpi import MPI
from ..timing import function_timer
from ..utils import Environment, Logger

from .copy import Copy
from .delete import Delete
from .mapmaker import MapMaker
from .mapmaker_binning import BinMap
from .pixels_wcs import PixelsWCS
from .pointing_detector import PointingDetectorSimple
from .polyfilter import PolyFilter
from .stokes_weights import StokesWeights

try:
    import findpeaks
    have_peaks = True
except ImportError:
    have_peaks = False


@function_timer
def find_source(
    img, 
    search_origin=None,
    search_radius=None,
    debug_root=None,
):
    """Locate a single point source in an image.

    This uses the findpeaks package to locate the best source candidate.  The search
    may be restricted to within some radius from a given pixel.

    Args:
        img (array):  The 2D image
        search_origin (tuple):  For restricting the search range, the (x, y) origin
            of the search.
        search_radius (float):  For restricting the search range, the radius in pixels
            to consider.
        debug_root (str):  Root path to debug output plots.

    Returns:
        (tuple):  The best estimate of the source location in pixel coordinates.

    """
    log = Logger.get()
    if not have_peaks:
        msg = "Cannot import the 'findpeaks' package, needed for source finding."
        raise RuntimeError(msg)

    env = Environment.get()
    verbosity = 2
    if env.log_level() == "DEBUG":
        verbosity = 2
    elif env.log_level() == "VERBOSE":
        verbosity = 3

    # Set denoising window to 10% of shortest dimension
    short = min(img.shape[0], img.shape[1])
    window = int(0.1 * short)
    if window < 10:
        window = 10

    interp = window

    print(f"window set to {window}")
    print(f"interp set to {interp}")

    fpk = findpeaks.findpeaks(
        method="topology",
        togray=False,
        interpolate=interp,
        denoise="fastnl",
        window=window,
        whitelist=["peak"],
        verbose=verbosity,
    )
    result = fpk.fit(img)
    #print(result)

    import matplotlib.pyplot as plt

    fig = plt.figure(dpi=100, figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        result["Xproc"], 
        cmap="jet", 
    )
    fig.colorbar(im, orientation="vertical")
    outfile = f"{debug_root}_proc.pdf"
    plt.savefig(
        outfile, 
        dpi=100, 
        bbox_inches="tight", 
        format="pdf"
    )
    plt.close()

    fig_ax = fpk.plot(cmap="jet")
    #print(fig_ax)
    outfile = f"{debug_root}_result.pdf"
    plt.savefig(
        outfile, 
        dpi=100, 
        bbox_inches="tight", 
        format="pdf"
    )
    plt.close()

    pers = result["persistence"]
    if len(pers) == 0:
        return (None, None)
    else:
        pix_off = -1.5
        cent_x = result["persistence"]["x"][0] + pix_off
        cent_y = result["persistence"]["y"][0] + pix_off
        return (cent_x, cent_y)


@function_timer
def make_cal_map(
    name,
    data,
    detectors,
    resolution,
    view,
    boresight,
    det_data, 
    det_flags,
    det_flag_mask,
    shared_flags,
    shared_flag_mask,
    noise_model,
    poly_order=5,
    debug_dir=None,
):
    """Make a simple, intensity-only map for use in calibration.

    This applies a 1D polynomial filter to a copy of the data and makes a
    binned map.  Then it extracts this and returns it as a 2D image along with
    wcs information.

    Args:

    Returns:

    
    """
    # Copy data to temp object
    temp_name = "temp_cal"
    cp = Copy(detdata=[(det_data, temp_name)])
    cp.apply(data, detectors=detectors)

    # Filter
    polyfilt = PolyFilter(
        det_data=temp_name,
        order=poly_order,
        det_flags=det_flags,
        det_flag_mask=det_flag_mask,
        shared_flags=shared_flags,
        shared_flag_mask=shared_flag_mask,
        view=view,
    )
    polyfilt.apply(data, detectors=detectors)

    det_pointing_cal = PointingDetectorSimple(
        view=view,
        shared_flags=shared_flags,
        shared_flag_mask=shared_flag_mask,
        boresight=boresight,
        quats="quats_cal",
    )
    pixels_cal = PixelsWCS(
        detector_pointing=det_pointing_cal,
        projection="CAR",
        resolution=resolution,
        auto_bounds=True,
        view=view,
        submaps=1,
    )
    stokes_cal = StokesWeights(
        detector_pointing=det_pointing_cal,
        mode="I",
        view=view,
    )
    binner_cal = BinMap(
        det_data=temp_name,
        det_flags=det_flags,
        det_flag_mask=det_flag_mask,
        shared_flags=shared_flags,
        shared_flag_mask=shared_flag_mask,
        pixel_pointing=pixels_cal,
        stokes_weights=stokes_cal,
        noise_model=noise_model,
    )
    mapper = MapMaker(
        name=name,
        det_data=temp_name,
        convergence=1.0e-8,
        iter_min=10,
        iter_max=100,
        solve_rcond_threshold=1.0e-3,
        map_rcond_threshold=1.0e-3,
        binning=binner_cal,
        template_matrix=None,
        write_binmap=False,
        write_map=True,
        write_hits=True,
        write_cov=False,
        write_rcond=False,
        keep_final_products=True,
        reset_pix_dist=True,
        output_dir=debug_dir,
    )
    mapper.apply(data, detectors=detectors)
    Delete(det_data=[temp_name,]).apply(data)

    if debug_dir is not None:
        if data.comm.group_rank == 0:
            from ..tests._helpers import plot_wcs_maps

            hitfile = os.path.join(debug_dir, f"{mapper.name}_hits.fits")
            mapfile = os.path.join(debug_dir, f"{mapper.name}_map.fits")
            plot_wcs_maps(
                hitfile=hitfile,
                mapfile=mapfile,
                # range_I=(-5.0, 5.0),
            )

    # Extract the 2D image and the WCS that was used
    wcs = WCS(header=pixels_cal.wcs.to_header())
    img_data = None
    img_data = np.zeros(
        (pixels_cal.wcs_shape[1], pixels_cal.wcs_shape[0]),
        dtype=np.float64,
    )
    img_data[:, :] = np.transpose(
        data[f"{mapper.name}_map"][:, :, 0].reshape(
            pixels_cal.wcs_shape
        )
    )

    # Clean up temporary data products
    for prod in ["hits", "map", "cov", "invcov", "rcond"]:
        prod_name = f"{mapper.name}_{prod}"
        data[prod_name].clear()
        del data[prod_name]

    return wcs, img_data


# Functions to fit a 2D gaussian given the previously-determined
# approximate peak location.

def evaluate_gaussian(
    n_lon,
    n_lat, 
    min_lon, 
    max_lon, 
    min_lat, 
    max_lat, 
    center_lon, 
    center_lat,
    sigma_major,
    sigma_minor,
    angle,
    amplitude,
):
    range_lon = max_lon - min_lon
    range_lat = max_lat - min_lat
    pix_incr_lon = range_lon / (n_lon - 1)
    pix_incr_lat = range_lat / (n_lat - 1)
    pix_lon = np.tile(
        min_lon + pix_incr_lon * (0.5 + np.arange(n_lon)) - center_lon,
        n_lat,
    )
    pix_lat = np.repeat(
        min_lat + pix_incr_lat * (0.5 + np.arange(n_lat)) - center_lat,
        n_lon,
    )

    # Compute coefficients
    cossq = np.cos(angle)**2
    sinsq = np.sin(angle)**2
    sintwo = np.sin(2 * angle)
    a = cossq / (2 * sigma_major**2) + sinsq / (2 * sigma_minor**2)
    b = - sintwo / (4 * sigma_major**2) + sintwo / (4 * sigma_minor**2)
    c = sinsq / (2 * sigma_major**2) + cossq / (2 * sigma_minor**2)

    return amplitude * np.exp(
        -(a * pix_lon**2 + 2 * b * pix_lon * pix_lat + c * pix_lat**2)
    ).reshape(
        (n_lon, n_lat)
    )

def fit_gaussian_func(x, *args, **kwargs):
    center_lon = x[0]
    center_lat = x[1]
    sigma_major = x[2]
    sigma_minor = x[3]
    angle = x[4]
    amp = x[5]
    n_lon = kwargs["n_lon"]
    n_lat = kwargs["n_lat"]
    min_lon = kwargs["min_lon"]
    max_lon = kwargs["max_lon"]
    min_lat = kwargs["min_lat"]
    max_lat = kwargs["max_lat"]
    data = kwargs["data"]

    current = evaluate_gaussian(
        n_lon,
        n_lat, 
        min_lon, 
        max_lon, 
        min_lat, 
        max_lat, 
        center_lon, 
        center_lat,
        sigma_major,
        sigma_minor,
        angle,
        amp,
    )
    return (current - data).reshape((-1))


def fit_gaussian(
    data, 
    min_lon, 
    max_lon, 
    min_lat, 
    max_lat,
    peak_lon,
    peak_lat,
    nominal_fwhm,
):
    n_lon = data.shape[0]
    n_lat = data.shape[1]
    bounds = (
        np.array(
            [
                peak_lon - 5 * nominal_fwhm, 
                peak_lat - 5 * nominal_fwhm,
                0.2 * nominal_fwhm,
                0.2 * nominal_fwhm,
                - np.pi,
                np.mean(data),
            ]
        ),
        np.array(
            [
                peak_lon + 5 * nominal_fwhm,
                peak_lat + 5 * nominal_fwhm,
                5.0 * nominal_fwhm,
                5.0 * nominal_fwhm,
                np.pi,
                2.0 * np.amax(data)
            ]
        ),
    )
    x_0 = np.array(
        [
            peak_lon,
            peak_lat,
            nominal_fwhm,
            nominal_fwhm,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )

    result = least_squares(
        fit_gaussian_func,
        x_0,
        jac="2-point",
        bounds=bounds,
        xtol=1.0e-10,
        gtol=1.0e-10,
        ftol=1.0e-10,
        max_nfev=500,
        method="trf",
        verbose=0,
        kwargs={
            "n_lon": n_lon,
            "n_lat": n_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "min_lat": min_lat,
            "max_lat": max_lat,
            "data": data,
        },
    )
    ret = dict()
    ret["fit_result"] = result
    if result.success:
        ret["center_lon"] = result.x[0]
        ret["center_lat"] = result.x[1]
        ret["sigma_major"] = result.x[2]
        ret["sigma_minor"] = result.x[3]
        ret["angle"] = result.x[4]
        ret["amplitude"] = result.x[5]
    else:
        ret["center_lon"] = peak_lon
        ret["center_lat"] = peak_lat
        ret["sigma_major"] = nominal_fwhm
        ret["sigma_minor"] = nominal_fwhm
        ret["angle"] = 0.0
        ret["amplitude"] = np.amax(data)
    return ret
