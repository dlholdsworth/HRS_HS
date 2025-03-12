from pyreduce import extract
#import clipnflip

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy import signal
import scipy
from tqdm import tqdm
import dlh_utils

# TODO allow other line shapes
def gaussian(x, A, mu, sig):
    """
    A: height
    mu: offset from central line
    sig: standard deviation
    """
    return A * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def lorentzian(x, A, x0, mu):
    """
    A: height
    x0: offset from central line
    mu: width of lorentzian
    """
    return A * mu / ((x - x0) ** 2 + 0.25 * mu ** 2)


def slitfunc_curved(
    img, im_flt, ycen, tilt, shear, lambda_sp, lambda_sf, osample, yrange, maxiter=20, gain=1
):
    """Decompose an image into a spectrum and a slitfunction, image may be curved

    Parameters
    ----------
    img : array[n, m]
        input image
    ycen : array[n]
        traces the center of the order
    tilt : array[n]
        tilt (1st order curvature) of the order along the image, set to 0 if order straight
    shear : array[n]
        shear (2nd order curvature) of the order along the image, set to 0 if order straight
    osample : int
        Subpixel ovsersampling factor (the default is 1, no oversampling)
    lambda_sp : float
        smoothing factor spectrum (the default is 0, no smoothing)
    lambda_sl : float
        smoothing factor slitfunction (the default is 0.1, small smoothing)
    yrange : array[2]
        number of pixels below and above the central line that have been cut out
    maxiter : int, optional
        maximumim number of iterations, by default 20
    gain : float, optional
        gain of the image, by default 1

    Returns
    -------
    sp, sl, model, unc
        spectrum, slitfunction, model, spectrum uncertainties
    """

    # Convert datatypes to expected values
    lambda_sf = float(lambda_sf)
    lambda_sp = float(lambda_sp)
    osample = int(osample)
    maxiter = int(maxiter)
    img = np.asanyarray(img, dtype=c_double)
    im_flt = np.asanyarray(im_flt, dtype=c_double)
    ycen = np.asarray(ycen, dtype=c_double)
    yrange = np.asarray(yrange, dtype=int)

    assert img.ndim == 2, "Image must be 2 dimensional"
    assert ycen.ndim == 1, "Ycen must be 1 dimensional"
    assert maxiter > 0, "Maximum iterations must be positive"

    if np.isscalar(tilt):
        tilt = np.full(img.shape[1], tilt, dtype=c_double)
    else:
        tilt = np.asarray(tilt, dtype=c_double)
    if np.isscalar(shear):
        shear = np.full(img.shape[1], shear, dtype=c_double)
    else:
        shear = np.asarray(shear, dtype=c_double)

    assert (
        img.shape[1] == ycen.size
    ), "Image and Ycen shapes are incompatible, got {} and {}".format(
        img.shape, ycen.shape
    )
    assert (
        img.shape[1] == tilt.size
    ), "Image and Tilt shapes are incompatible, got {} and {}".format(
        img.shape, tilt.shape
    )
    assert (
        img.shape[1] == shear.size
    ), "Image and Shear shapes are incompatible, got {} and {}".format(
        img.shape,
        shear.shape,
    )

    assert osample > 0, f"Oversample rate must be positive, but got {osample}"
    assert (
        lambda_sf >= 0
    ), f"Slitfunction smoothing must be positive, but got {lambda_sf}"
    assert lambda_sp >= 0, f"Spectrum smoothing must be positive, but got {lambda_sp}"

    # assert np.ma.all(np.isfinite(img)), "All values in the image must be finite"
    assert np.all(np.isfinite(ycen)), "All values in ycen must be finite"
    assert np.all(np.isfinite(tilt)), "All values in tilt must be finite"
    assert np.all(np.isfinite(shear)), "All values in shear must be finite"

    assert yrange.ndim == 1, "Yrange must be 1 dimensional"
    assert yrange.size == 2, "Yrange must have 2 elements"
    assert (
        yrange[0] + yrange[1] + 1 == img.shape[0]
    ), "Yrange must cover the whole image"
    assert yrange[0] >= 0, "Yrange must be positive"
    assert yrange[1] >= 0, "Yrange must be positive"

    # Retrieve some derived values
    nrows, ncols = img.shape
    ny = osample * (nrows + 1) + 1

    ycen_offset = ycen.astype(c_int)
    ycen_int = ycen - ycen_offset
    y_lower_lim = int(yrange[0])

    mask = np.ma.getmaskarray(img)
    img = np.ma.getdata(img)
    im_flt=np.ma.getdata(im_flt)
    mask2 = ~np.isfinite(img)
    img[mask2] = 0
    mask |= ~np.isfinite(img)

    # sp should never be all zero (thats a horrible guess) and leads to all nans
    # This is a simplified run of the algorithm without oversampling or curvature
    # But strong smoothing
    # To remove the most egregious outliers, which would ruin the fit
    sp = np.sum(img, axis=0)
    median_filter(sp, 5, output=sp)
    sl = np.median(im_flt, axis=1)
    #sl = np.median(img, axis=1)
    sl /= np.sum(sl)

    model = sl[:, None] * sp[None, :]
    diff = model - img
    mask[np.abs(diff) > 10 * diff.std()] = True

    sp = np.sum(img, axis=0)

    mask = np.where(mask, c_int(0), c_int(1))
    # Determine the shot noise
    # by converting electrons to photonsm via the gain
    pix_unc = np.nan_to_num(np.abs(img), copy=False)
    pix_unc *= gain
    np.sqrt(pix_unc, out=pix_unc)
    pix_unc[pix_unc < 1] = 1

    psf_curve = np.zeros((ncols, 3), dtype=c_double)
    psf_curve[:, 1] = tilt
    psf_curve[:, 2] = shear

    # Initialize arrays and ensure the correct datatype for C
    requirements = ["C", "A", "W", "O"]
    sp = np.require(sp, dtype=c_double, requirements=requirements)
    mask = np.require(mask, dtype=c_mask, requirements=requirements)
    img = np.require(img, dtype=c_double, requirements=requirements)
    im_flt = np.require(im_flt, dtype=c_double, requirements=requirements)
    pix_unc = np.require(pix_unc, dtype=c_double, requirements=requirements)
    ycen_int = np.require(ycen_int, dtype=c_double, requirements=requirements)
    ycen_offset = np.require(ycen_offset, dtype=c_int, requirements=requirements)

    # This memory could be reused between swaths
    sl = np.zeros(ny, dtype=c_double)
    model = np.zeros((nrows, ncols), dtype=c_double)
    unc = np.zeros(ncols, dtype=c_double)

    # Info contains the folowing: sucess, cost, status, iteration, delta_x
    info = np.zeros(5, dtype=c_double)

    col = np.sum(mask, axis=0) == 0
    if np.any(col):
        mask[mask.shape[0] // 2, col] = 1
    # assert not np.any(np.sum(mask, axis=0) == 0), "At least one mask column is all 0."

    # Call the C function
    slitfunc_2dlib.slit_func_curved(
        ffi.cast("int", ncols),ffi.cast("int", nrows),ffi.cast("int", ny),ffi.cast("double *", img.ctypes.data),
        ffi.cast("double *", im_flt.ctypes.data),
        ffi.cast("double *", pix_unc.ctypes.data),
        ffi.cast("unsigned char *", mask.ctypes.data),
        ffi.cast("double *", ycen_int.ctypes.data),
        ffi.cast("int *", ycen_offset.ctypes.data),
        ffi.cast("int", y_lower_lim),
        ffi.cast("int", osample),
        ffi.cast("double", lambda_sp),
        ffi.cast("double", lambda_sf),
        ffi.cast("int", maxiter),
        ffi.cast("double *", psf_curve.ctypes.data),
        ffi.cast("double *", sp.ctypes.data),
        ffi.cast("double *", sl.ctypes.data),
        ffi.cast("double *", model.ctypes.data),
        ffi.cast("double *", unc.ctypes.data),
        ffi.cast("double *", info.ctypes.data),
    )

    if np.any(np.isnan(sp)):
        logger.error("NaNs in the spectrum")

    # The decomposition failed
    if info[0] == 0:
        status = info[2]
        if status == 0:
            msg = "I dont't know what happened"
        elif status == -1:
            msg = f"Did not finish convergence after maxiter ({maxiter}) iterations"
        elif status == -2:
            msg = "Curvature is larger than the swath. Check the curvature!"
        else:
            msg = f"Check the C code, for status = {status}"
        logger.error(msg)
        # raise RuntimeError(msg)

    mask = mask == 0

    return sp, sl, model, unc, mask, info


def make_index(ymin, ymax, xmin, xmax, zero=0):
    """Create an index (numpy style) that will select part of an image with changing position but fixed height

    The user is responsible for making sure the height is constant, otherwise it will still work, but the subsection will not have the desired format

    Parameters
    ----------
    ymin : array[ncol](int)
        lower y border
    ymax : array[ncol](int)
        upper y border
    xmin : int
        leftmost column
    xmax : int
        rightmost colum
    zero : bool, optional
        if True count y array from 0 instead of xmin (default: False)

    Returns
    -------
    index : tuple(array[height, width], array[height, width])
        numpy index for the selection of a subsection of an image
    """

    # TODO
    # Define the indices for the pixels between two y arrays, e.g. pixels in an order
    # in x: the rows between ymin and ymax
    # in y: the column, but n times to match the x index
    ymin = np.asarray(ymin, dtype=int)
    ymax = np.asarray(ymax, dtype=int)
    xmin = int(xmin)
    xmax = int(xmax)

    if zero:
        zero = xmin

    index_x = np.array(
        [np.arange(ymin[col], ymax[col] + 1) for col in range(xmin - zero, xmax - zero)]
    )
    index_y = np.array(
        [
            np.full(ymax[col] - ymin[col] + 1, col)
            for col in range(xmin - zero, xmax - zero)
        ]
    )
    index = index_x.T, index_y.T + zero

    return index


def make_bins(swath_width, xlow, xhigh, ycen):
    """Create bins for the swathes
    Bins are roughly equally sized, have roughly length swath width (if given)
    and overlap roughly half-half with each other

    Parameters
    ----------
    swath_width : {int, None}
        initial value for the swath_width, bins will have roughly that size, but exact value may change
        if swath_width is None, determine a good value, from the data
    xlow : int
        lower bound for x values
    xhigh : int
        upper bound for x values
    ycen : array[ncol]
        center of the order trace

    Returns
    -------
    nbin : int
        number of bins
    bins_start : array[nbin]
        left(beginning) side of the bins
    bins_end : array[nbin]
        right(ending) side of the bins
    """

    if swath_width is None:
        ncol = len(ycen)
        i = np.unique(ycen.astype(int))  # Points of row crossing
        # ni = len(i)  # This is how many times this order crosses to the next row
        if len(i) > 1:  # Curved order crosses rows
            i = np.sum(i[1:] - i[:-1]) / (len(i) - 1)
            nbin = np.clip(
                int(np.round(ncol / i)) // 3, 3, 20
            )  # number of swaths along the order
        else:  # Perfectly aligned orders
            nbin = np.clip(ncol // 400, 3, None)  # Still follow the changes in PSF
        nbin = nbin * (xhigh - xlow) // ncol  # Adjust for the true order length
    else:
        nbin = np.clip(int(np.round((xhigh - xlow) / swath_width)), 1, None)

    bins = np.linspace(xlow, xhigh, 2 * nbin + 1)  # boundaries of bins
    bins_start = np.ceil(bins[:-2]).astype(int)  # beginning of each bin
    bins_end = np.floor(bins[2:]).astype(int)  # end of each bin

    return nbin, bins_start, bins_end

class Swath:
    def __init__(self, nswath):
        self.nswath = nswath
        self.spec = [None] * nswath
        self.slitf = [None] * nswath
        self.model = [None] * nswath
        self.unc = [None] * nswath
        self.mask = [None] * nswath
        self.info = [None] * nswath

    def __len__(self):
        return self.nswath

    def __getitem__(self, key):
        return (
            self.spec[key],
            self.slitf[key],
            self.model[key],
            self.unc[key],
            self.mask[key],
            self.info[key],
        )

    def __setitem__(self, key, value):
        self.spec[key] = value[0]
        self.slitf[key] = value[1]
        self.model[key] = value[2]
        self.unc[key] = value[3]
        self.mask[key] = value[4]
        self.info[key] = value[5]

def get_y_scale(ycen, xrange, extraction_width, nrow):
    """Calculate the y limits of the order
    This is especially important at the edges

    Parameters
    ----------
    ycen : array[ncol]
        order trace
    xrange : tuple(int, int)
        column range
    extraction_width : tuple(int, int)
        extraction width in pixels below and above the order
    nrow : int
        number of rows in the image, defines upper edge

    Returns
    -------
    y_low, y_high : int, int
        lower and upper y bound for extraction
    """
    ycen = ycen[xrange[0] : xrange[1]]

    ymin = ycen - extraction_width[0]
    ymin = np.floor(ymin)
    if min(ymin) < 0:
        ymin = ymin - min(ymin)  # help for orders at edge
    if max(ymin) >= nrow:
        ymin = ymin - max(ymin) + nrow - 1  # helps at edge

    ymax = ycen + extraction_width[1]
    ymax = np.ceil(ymax)
    if max(ymax) >= nrow:
        ymax = ymax - max(ymax) + nrow - 1  # helps at edge

    # Define a fixed height area containing one spectral order
    y_lower_lim = int(np.min(ycen - ymin))  # Pixels below center line
    y_upper_lim = int(np.min(ymax - ycen))  # Pixels above center line

    return y_lower_lim, y_upper_lim

def extract_spectrum(
    img,
    ycen,
    yrange,
    xrange,
    gain=1,
    readnoise=0,
    lambda_sf=0.1,
    lambda_sp=0,
    osample=1,
    swath_width=None,
    maxiter=20,
    telluric=None,
    scatter=None,
    normalize=False,
    threshold=0,
    tilt=None,
    shear=None,
    plot=False,
    plot_title=None,
    im_norm=None,
    im_ordr=None,
    out_spec=None,
    out_sunc=None,
    out_slitf=None,
    out_mask=None,
    progress=None,
    ord_num=0,
    **kwargs,
):
    """
    Extract the spectrum of a single order from an image
    The order is split into several swathes of roughly swath_width length, which overlap half-half
    For each swath a spectrum and slitfunction are extracted
    overlapping sections are combined using linear weights (centrum is strongest, falling off to the edges)
    Here is the layout for the bins:

    ::

           1st swath    3rd swath    5th swath      ...
        /============|============|============|============|============|

                  2nd swath    4th swath    6th swath
               |------------|------------|------------|------------|
               |.....|
               overlap

               +     ******* 1
                +   *
                 + *
                  *            weights (+) previous swath, (*) current swath
                 * +
                *   +
               *     +++++++ 0

    Parameters
    ----------
    img : array[nrow, ncol]
        observation (or similar)
    ycen : array[ncol]
        order trace of the current order
    yrange : tuple(int, int)
        extraction width in pixles, below and above
    xrange : tuple(int, int)
        columns range to extract (low, high)
    gain : float, optional
        adu to electron, amplifier gain (default: 1)
    readnoise : float, optional
        read out noise factor (default: 0)
    lambda_sf : float, optional
        slit function smoothing parameter, usually very small (default: 0.1)
    lambda_sp : int, optional
        spectrum smoothing parameter, usually very small (default: 0)
    osample : int, optional
        oversampling factor, i.e. how many subpixels to create per pixel (default: 1, i.e. no oversampling)
    swath_width : int, optional
        swath width suggestion, actual width depends also on ncol, see make_bins (default: None, which will determine the width based on the order tracing)
    telluric : {float, None}, optional
        telluric correction factor (default: None, i.e. no telluric correction)
    scatter : {array, None}, optional
        background scatter as 2d polynomial coefficients (default: None, no correction)
    normalize : bool, optional
        whether to create a normalized image. If true, im_norm and im_ordr are used as output (default: False)
    threshold : int, optional
        threshold for normalization (default: 0)
    tilt : array[ncol], optional
        The tilt (1st order curvature) of the slit in this order for the curved extraction (default: None, i.e. tilt = 0)
    shear : array[ncol], optional
        The shear (2nd order curvature) of the slit in this order for the curved extraction (default: None, i.e. shear = 0)
    plot : bool, optional
        wether to plot the progress, plotting will slow down the procedure significantly (default: False)
    ord_num : int, optional
        current order number, just for plotting (default: 0)
    im_norm : array[nrow, ncol], optional
        normalized image, only output if normalize is True (default: None)
    im_ordr : array[nrow, ncol], optional
        image of the order blaze, only output if normalize is True (default: None)

    Returns
    -------
    spec : array[ncol]
        extracted spectrum
    slitf : array[nslitf]
        extracted slitfunction
    mask : array[ncol]
        mask of the column range to use in the spectrum
    unc : array[ncol]
        uncertainty on the spectrum
    """
    
    _, ncol = img.shape
    ncol=ncol
    ylow, yhigh = yrange
    xlow, xhigh = xrange
    nslitf = osample * (ylow + yhigh + 2) + 1
    height = yhigh + ylow + 1

    ycen_int = np.floor(ycen).astype(int)

    spec = np.zeros(ncol) if out_spec is None else out_spec
    sunc = np.zeros(ncol) if out_sunc is None else out_sunc
    mask = np.full(ncol, False) if out_mask is None else out_mask
    slitf = np.zeros(nslitf) if out_slitf is None else out_slitf

    nbin, bins_start, bins_end = make_bins(swath_width, xlow, xhigh, ycen)
    nswath = 2 * nbin - 1
    swath = Swath(nswath)
    margin = np.zeros((nswath, 2), int)

    if normalize:
        norm_img = [None] * nswath
        norm_model = [None] * nswath

    # Perform slit decomposition within each swath stepping through the order with
    # half swath width. Spectra for each decomposition are combined with linear weights.
    with tqdm(
        enumerate(zip(bins_start, bins_end)),
        total=len(bins_start),
        leave=False,
        desc="Swath",
    ) as t:
        for ihalf, (ibeg, iend) in t:
            #logger.debug("Extracting Swath %i, Columns: %i - %i", ihalf, ibeg, iend)

            # Cut out swath from image
            index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
            swath_img = img[index]
            swath_ycen = ycen[ibeg:iend]

            # Corrections
            # TODO: what is it even supposed to do?
            if telluric is not None:  # pragma: no cover
                telluric_correction = calc_telluric_correction(telluric, swath_img)
            else:
                telluric_correction = 0

            if scatter is not None:
                scatter_correction = calc_scatter_correction(scatter, index)
            else:
                scatter_correction = 0

            swath_img -= scatter_correction + telluric_correction

            # Do Slitfunction extraction
            swath_tilt = tilt[ibeg:iend] if tilt is not None else 0
            swath_shear = shear[ibeg:iend] if shear is not None else 0
            swath[ihalf] = slitfunc_curved(
                swath_img,
                swath_ycen,
                swath_tilt,
                swath_shear,
                lambda_sp=lambda_sp,
                lambda_sf=lambda_sf,
                osample=osample,
                yrange=yrange,
                maxiter=maxiter,
                gain=gain,
            )
            t.set_postfix(chi=f"{swath[ihalf][5][1]:1.2f}")

            if normalize:
                # Save image and model for later
                # Use np.divide to avoid divisions by zero
                where = swath.model[ihalf] > threshold / gain
                norm_img[ihalf] = np.ones_like(swath.model[ihalf])
                np.divide(
                    np.abs(swath_img),
                    swath.model[ihalf],
                    where=where,
                    out=norm_img[ihalf],
                )
                norm_model[ihalf] = swath.model[ihalf]

            if plot >= 2 and not np.all(np.isnan(swath_img)):  # pragma: no cover
                if progress is None:
                    progress = ProgressPlot(
                        swath_img.shape[0], swath_img.shape[1], nslitf, title=plot_title
                    )
                progress.plot(
                    swath_img,
                    swath.spec[ihalf],
                    swath.slitf[ihalf],
                    swath.model[ihalf],
                    swath_ycen,
                    swath.mask[ihalf],
                    ord_num,
                    ibeg,
                    iend,
                )

    # Remove points at the border of the each swath, if order has tilt
    # as those pixels have bad information
    #for i in range(nswath):
    #    margin[i, :] = int(swath.info[i][4]) + 1

    # Weight for combining swaths
    weight = [np.ones(bins_end[i] - bins_start[i]) for i in range(nswath)]
    weight[0][: margin[0, 0]] = 0
    weight[-1][len(weight[-1]) - margin[-1, 1] :] = 0
    for i, j in zip(range(0, nswath - 1), range(1, nswath)):
        width = bins_end[i] - bins_start[i]
        overlap = bins_end[i] - bins_start[j]

        # Start and end indices for the two swaths
        start_i = width - overlap + margin[j, 0]
        end_i = width - margin[i, 1]

        start_j = margin[j, 0]
        end_j = overlap - margin[i, 1]

        # Weights for one overlap from 0 to 1, but do not include those values (whats the point?)
        triangle = np.linspace(0, 1, overlap + 1, endpoint=False)[1:]
        # Cut away the margins at the corners
        triangle = triangle[margin[j, 0] : len(triangle) - margin[i, 1]]

        # Set values
        weight[i][start_i:end_i] = 1 - triangle
        weight[j][start_j:end_j] = triangle

        # Don't use the pixels at the egdes (due to curvature)
        weight[i][end_i:] = 0
        weight[j][:start_j] = 0

    # Update column range
    xrange[0] += margin[0, 0]
    xrange[1] -= margin[-1, 1]
    mask[: xrange[0]] = True
    mask[xrange[1] :] = True

    # Apply weights
    for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
        spec[ibeg:iend] += swath.spec[i] * weight[i]
        sunc[ibeg:iend] += swath.unc[i] * weight[i]

    if normalize:
        for i, (ibeg, iend) in enumerate(zip(bins_start, bins_end)):
            index = make_index(ycen_int - ylow, ycen_int + yhigh, ibeg, iend)
            im_norm[index] += norm_img[i] * weight[i]
            im_ordr[index] += norm_model[i] * weight[i]

    slitf[:] = np.mean(swath.slitf, axis=0)
    sunc[:] = np.sqrt(sunc ** 2 + (readnoise / gain) ** 2)
    return spec, slitf, mask, sunc

def fix_parameters(xwd, cr, orders, nrow, ncol, nord, ignore_column_range=False):
    """Fix extraction width and column range, so that all pixels used are within the image.
    I.e. the column range is cut so that the everything is within the image

    Parameters
    ----------
    xwd : float, array
        Extraction width, either one value for all orders, or the whole array
    cr : 2-tuple(int), array
        Column range, either one value for all orders, or the whole array
    orders : array
        polynomial coefficients that describe each order
    nrow : int
        Number of rows in the image
    ncol : int
        Number of columns in the image
    nord : int
        Number of orders in the image
    ignore_column_range : bool, optional
        if true does not change the column range, however this may lead to problems with the extraction, by default False

    Returns
    -------
    xwd : array
        fixed extraction width
    cr : array
        fixed column range
    orders : array
        the same orders as before
    """

    if xwd is None:
        xwd = 0.5
    if np.isscalar(xwd):
        xwd = np.tile([xwd, xwd], (nord, 1))
    else:
        xwd = np.asarray(xwd)
        if xwd.ndim == 1:
            xwd = np.tile(xwd, (nord, 1))

    if cr is None:
        cr = np.tile([0, ncol], (nord, 1))
    else:
        cr = np.asarray(cr)
        if cr.ndim == 1:
            cr = np.tile(cr, (nord, 1))

    orders = np.asarray(orders)

    xwd = np.array([xwd[0], *xwd, xwd[-1]])
    cr = np.array([cr[0], *cr, cr[-1]])
    orders = extend_orders(orders, nrow)

    xwd = fix_extraction_width(xwd, orders, cr, ncol)
    if not ignore_column_range:
        cr = fix_column_range(cr, orders, xwd, nrow, ncol)

    orders = orders[1:-1]
    xwd = xwd[1:-1]
    cr = cr[1:-1]

    return xwd, cr, orders

###########################

def extend_orders(orders, nrow):
    """Extrapolate extra orders above and below the existing ones

    Parameters
    ----------
    orders : array[nord, degree]
        order tracing coefficients
    nrow : int
        number of rows in the image

    Returns
    -------
    orders : array[nord + 2, degree]
        extended orders
    """

    nord, ncoef = orders.shape

    if nord > 1:
        order_low = 2 * orders[0] - orders[1]
        order_high = 2 * orders[-1] - orders[-2]
    else:
        order_low = [0 for _ in range(ncoef)]
        order_high = [0 for _ in range(ncoef - 1)] + [nrow]

    return np.array([order_low, *orders, order_high])


#############################

def fix_extraction_width(xwd, orders, cr, ncol):
    """Convert fractional extraction width to pixel range

    Parameters
    ----------
    extraction_width : array[nord, 2]
        current extraction width, in pixels or fractions (for values below 1.5)
    orders : array[nord, degree]
        order tracing coefficients
    column_range : array[nord, 2]
        column range to use
    ncol : int
        number of columns in image

    Returns
    -------
    extraction_width : array[nord, 2]
        updated extraction width in pixels
    """

    if not np.all(xwd > 1.5):
        # if extraction width is in relative scale transform to pixel scale
        x = np.arange(ncol)
        for i in range(1, len(xwd) - 1):
            for j in [0, 1]:
                if xwd[i, j] < 1.5:
                    k = i - 1 if j == 0 else i + 1
                    left = max(cr[[i, k], 0])
                    right = min(cr[[i, k], 1])

                    if right < left:
                        raise ValueError(
                            f"Check your column ranges. Orders {i} and {k} are weird"
                        )

                    current = np.polyval(orders[i], x[left:right])
                    below = np.polyval(orders[k], x[left:right])
                    xwd[i, j] *= np.min(np.abs(current - below))

        xwd[0] = xwd[1]
        xwd[-1] = xwd[-2]

    xwd = np.ceil(xwd).astype(int)

    return xwd

##################

def fix_column_range(column_range, orders, extraction_width, nrow, ncol):
    """Fix the column range, so that no pixels outside the image will be accessed (Thus avoiding errors)

    Parameters
    ----------
    img : array[nrow, ncol]
        image
    orders : array[nord, degree]
        order tracing coefficients
    extraction_width : array[nord, 2]
        extraction width in pixels, (below, above)
    column_range : array[nord, 2]
        current column range
    no_clip : bool, optional
        if False, new column range will be smaller or equal to current column range, otherwise it can also be larger (default: False)

    Returns
    -------
    column_range : array[nord, 2]
        updated column range
    """

    ix = np.arange(ncol)
    # Loop over non extension orders
    for i, order in zip(range(1, len(orders) - 1), orders[1:-1]):
        # Shift order trace up/down by extraction_width
        coeff_bot, coeff_top = np.copy(order), np.copy(order)
        coeff_bot[-1] -= extraction_width[i, 0]
        coeff_top[-1] += extraction_width[i, 1]

        y_bot = np.polyval(coeff_bot, ix)  # low edge of arc
        y_top = np.polyval(coeff_top, ix)  # high edge of arc

        # find regions of pixels inside the image
        # then use the region that most closely resembles the existing column range (from order tracing)
        # but clip it to the existing column range (order tracing polynomials are not well defined outside the original range)
        points_in_image = np.where((y_bot >= 0) & (y_top < nrow))[0]

        if len(points_in_image) == 0:
            raise ValueError(
                f"No pixels are completely within the extraction width for order {i}"
            )

        regions = np.where(np.diff(points_in_image) != 1)[0]
        regions = [(r, r + 1) for r in regions]
        regions = [
            points_in_image[0],
            *points_in_image[(regions,)].ravel(),
            points_in_image[-1],
        ]
        regions = [[regions[i], regions[i + 1] + 1] for i in range(0, len(regions), 2)]
        overlap = [
            min(reg[1], column_range[i, 1]) - max(reg[0], column_range[i, 0])
            for reg in regions
        ]
        iregion = np.argmax(overlap)
        column_range[i] = np.clip(
            regions[iregion], column_range[i, 0], column_range[i, 1]
        )

    column_range[0] = column_range[1]
    column_range[-1] = column_range[-2]

    return column_range
    
    ###############################
    
def find_peaks(vec, cr,oversample):
    # This should probably be the same as in the wavelength calibration
    max_vec=np.max(vec)
    threshold = 2.
    peak_width = 2.
    window_width = 15.*oversample
    #vec -= np.ma.median(vec)
    height = np.median(vec)
    height=0.001
    #vec = np.ma.filled(vec, 0)

    #height = np.percentile(vec, 90) * threshold
    peaks, _ = signal.find_peaks(
        vec/max_vec, height=height, width=peak_width, distance=window_width
    )

    # Remove peaks at the edge
    peaks = peaks[
        (peaks >= window_width + 1)
        & (peaks < len(vec) - window_width - 1)
    ]
    # Remove the offset, due to vec being a subset of extracted
    peaks += cr[0]
    return vec, peaks

def determine_curvature_all_lines(original, extracted,column_range,extraction_width,orders,oversample):
    ncol = original.shape[1]
    nord =extracted.shape[0]
    # Store data from all orders
    all_peaks = []
    all_tilt = []
    all_shear = []
    plot_vec = []

    for j in tqdm(range(nord), desc="Order",leave=False):
        cr = column_range[j]
        xwd = extraction_width[j]
        ycen = np.polyval(orders[j], np.arange(ncol))
        ycen_int = ycen.astype(int)
        ycen -= ycen_int

        # Find peaks
        vec = extracted[j,cr[0] : cr[1]].astype('float64')
        vec, peaks = find_peaks(vec, cr,oversample)

        npeaks = len(peaks)
        

        # Determine curvature for each line seperately
        tilt = np.zeros(npeaks)
        shear = np.zeros(npeaks)
        mask = np.full(npeaks, True)
        for ipeak, peak in tqdm(
            enumerate(peaks), total=len(peaks), desc="Peak", leave=False
        ):
            try:
                tilt[ipeak], shear[ipeak] = determine_curvature_single_line(
                    original, peak, ycen, ycen_int, xwd
                )
            except RuntimeError:  # pragma: no cover
                mask[ipeak] = False

        # Store results
        all_peaks += [peaks[mask]]
        all_tilt += [tilt[mask]]
        all_shear += [shear[mask]]
        plot_vec += [vec]
    return all_peaks, all_tilt, all_shear, plot_vec


def determine_curvature_single_line(original, peak, ycen, ycen_int, xwd):
    """
    Fit the curvature of a single peak in the spectrum
    This is achieved by fitting a model, that consists of gaussians
    in spectrum direction, that are shifted by the curvature in each row.
    Parameters
    ----------
    original : array of shape (nrows, ncols)
        whole input image
    peak : int
        column position of the peak
    ycen : array of shape (ncols,)
        row center of the order of the peak
    xwd : 2 tuple
        extraction width above and below the order center to use
    Returns
    -------
    tilt : float
        first order curvature
    shear : float
        second order curvature
    """
    _, ncol = original.shape

    window_width = 9.
    curv_degree = 1
    # look at +- width pixels around the line
    # Extract short horizontal strip for each row in extraction width
    # Then fit a gaussian to each row, to find the center of the line
    x = peak + np.arange(-window_width, window_width + 1)
    x = x[(x >= 0) & (x < ncol)]
    xmin, xmax = int(x[0]), int(x[-1] + 1)

    # Look above and below the line center
    y = np.arange(-xwd[0], xwd[1] + 1)[:, None] - ycen[xmin:xmax][None, :]

    x = x[None, :]
    idx = make_index(ycen_int - xwd[0], ycen_int + xwd[1], xmin, xmax)
    img = original[idx].astype('float64')
    img_compressed = np.ma.compressed(img)

    img -= np.percentile(img_compressed, 1)
    img /= np.percentile(img_compressed, 99)
    img = np.ma.clip(img, 0, 1)

    sl = np.ma.mean(img, axis=1)
    sl = sl[:, None]

    peak_func = {"gaussian": gaussian, "lorentzian": lorentzian}
    peak_func = peak_func["gaussian"]

    def model(coef):
        A, middle, sig, *curv = coef
        mu = middle + shift(curv)
        mod = peak_func(x, A, mu, sig)
        mod *= sl
        return (mod - img).ravel()

    def model_compressed(coef):
        return np.ma.compressed(model(coef))

    A = np.nanpercentile(img_compressed, 95)
    sig = (xmax - xmin) / 4  # TODO
    if curv_degree == 1:
        shift = lambda curv: curv[0] * y
    elif curv_degree == 2:
        shift = lambda curv: (curv[0] + curv[1] * y) * y
    else:
        raise ValueError("Only curvature degrees 1 and 2 are supported")
    # res = least_squares(model, x0=[A, middle, sig, 0], loss="soft_l1", bounds=([0, xmin, 1, -10],[np.inf, xmax, xmax, 10]))
    x0 = [A, peak, sig] + [0] * curv_degree
#        res = least_squares(
#            model_compressed, x0=x0, method="trf", loss="soft_l1", f_scale=0.1
#        )
    #DLH mod
    res = least_squares(
        model_compressed, x0=x0, method="lm", loss="linear", f_scale=0.1
    )


    if curv_degree == 1:
        tilt, shear = res.x[3], 0
    elif curv_degree == 2:
        tilt, shear = res.x[3], res.x[4]
    else:
        tilt, shear = 0, 0

    model = model(res.x).reshape(img.shape) + img
    vmin = 0
    vmax = np.max(model)

    y = y.ravel()
    x = res.x[1] - xmin + (tilt + shear * y) * y
    y += xwd[0]
    

#    plt.subplot(121)
#    plt.imshow(img, vmin=vmin, vmax=vmax, origin="lower")
#    plt.plot(xwd[0] + ycen[xmin:xmax], "r")
#    plt.title("Input Image")
#    plt.xlabel("x [pixel]")
#    plt.ylabel("y [pixel]")
#
#    plt.subplot(122)
#    plt.imshow(model, vmin=vmin, vmax=vmax, origin="lower")
#    plt.plot(x, y, "r", label="curvature")
#    plt.ylim((-0.5, model.shape[0] - 0.5))
#    plt.title("Model")
#    plt.xlabel("x [pixel]")
#    plt.ylabel("y [pixel]")
#
#    plt.show()


    return tilt, shear

def fit_curvature_single_order(peaks, tilt, shear):
    try:
        middle = np.median(tilt)
        sigma = np.percentile(tilt, (32, 68))
        sigma = middle - sigma[0], sigma[1] - middle
        mask = (tilt >= middle - 5 * sigma[0]) & (tilt <= middle + 5 * sigma[1])
        peaks, tilt, shear = peaks[mask], tilt[mask], shear[mask]

        coef_tilt = np.zeros(1 + 1)
        res = least_squares(
            lambda coef: np.polyval(coef, peaks) - tilt,
            x0=coef_tilt,
            loss="arctan",
        )
        coef_tilt = res.x

        coef_shear = np.zeros(1 + 1)
        res = least_squares(
            lambda coef: np.polyval(coef, peaks) - shear,
            x0=coef_shear,
            loss="arctan",
        )
        coef_shear = res.x

    except:
        print(
            "Could not fit the curvature of this order. Using no curvature instead"
        )
        coef_tilt = np.zeros(1 + 1)
        coef_shear = np.zeros(1 + 1)

    return coef_tilt, coef_shear, peaks


def fit(peaks, tilt, shear,nord):
    mode="1D"
    if mode == "1D":
        coef_tilt = np.zeros((nord, 1 + 1))
        coef_shear = np.zeros((nord, 1 + 1))
        for j in range(nord):
            coef_tilt[j], coef_shear[j], _ = fit_curvature_single_order(
                peaks[j], tilt[j], shear[j]
            )
    elif self.mode == "2D":
        x = np.concatenate(peaks)
        y = [np.full(len(p), i) for i, p in enumerate(peaks)]
        y = np.concatenate(y)
        z = np.concatenate(tilt)
        coef_tilt = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

        z = np.concatenate(shear)
        coef_shear = polyfit2d(x, y, z, degree=self.fit_degree, loss="arctan")

    return coef_tilt, coef_shear

def eval(peaks, order, coef_tilt, coef_shear):
    mode="1D"
    if mode == "1D":
        tilt = np.zeros(peaks.shape)
        shear = np.zeros(peaks.shape)
        for i in np.unique(order):
            idx = order == i
            tilt[idx] = np.polyval(coef_tilt[i], peaks[idx])
            shear[idx] = np.polyval(coef_shear[i], peaks[idx])
    elif mode == "2D":
        tilt = polyval2d(peaks, order, coef_tilt)
        shear = polyval2d(peaks, order, coef_shear)
    return tilt, shear

def plot_comparison(original, tilt, shear, peaks,extraction_width,nord,orders,column_range):  # pragma: no cover
    _, ncol = original.shape
    output = np.zeros((np.sum(extraction_width) + nord, ncol))
    pos = [0]
    x = np.arange(ncol)
    for i in range(nord):
        ycen = np.polyval(orders[i], x)
        yb = ycen - extraction_width[i, 0]
        yt = ycen + extraction_width[i, 1]
        xl, xr = column_range[i]
        index = make_index(yb, yt, xl, xr)
        yl = pos[i]
        yr = pos[i] + index[0].shape[0]
        output[yl:yr, xl:xr] = original[index]
        pos += [yr]

    vmin, vmax = np.percentile(output[output != 0], (5, 95))
    plt.imshow(output, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")

    for i in range(nord):
        for p in peaks[i]:
            ew = extraction_width[i]
            x = np.zeros(ew[0] + ew[1] + 1)
            y = np.arange(-ew[0], ew[1] + 1)
            for j, yt in enumerate(y):
                x[j] = p + yt * tilt[i, p] + yt ** 2 * shear[i, p]
            y += pos[i] + ew[0]
            plt.plot(x, y, "r")

    locs = np.sum(extraction_width, axis=1) + 1
    locs = np.array([0, *np.cumsum(locs)[:-1]])
    locs[:-1] += (np.diff(locs) * 0.5).astype(int)
    locs[-1] += ((output.shape[0] - locs[-1]) * 0.5).astype(int)

    plt.yticks(locs, range(len(locs)))

    plt.xlabel("x [pixel]")
    plt.ylabel("order")
    plt.show()


def execute(fname,order_file,master_flat,Plot,night,oversample,arm,data_dir,out_location):

    hdu = fits.open(fname)
    h_prime = hdu[0].header

    img = hdu[0].data
    hdu.close()
    if arm =="R":
        img = img[::-1,::]
    if oversample != 1:
        img = scipy.ndimage.zoom(img, oversample, order=0)
    
    nrow,ncol = img.shape

    #Order file
    #order_file="ord_default_new.npz"
    file_dic=np.load(order_file, allow_pickle=True)
    column_range = file_dic["column_range"]
    c_all=file_dic["orders"]
    nord=len(c_all)
    n_ord=nord
    order_range = (0, nord)
    n = order_range[1] - order_range[0]
    mode = "1D"
    
    
    #Define the apertures
    if oversample !=1:
        master_flat_zoom = scipy.ndimage.zoom(master_flat, oversample, order=0)
    else:
        master_flat_zoom = master_flat
    if arm =='H':
        ext_aperture,c_all = dlh_utils.order_width_B(n_ord,master_flat_zoom,oversample,c_all)
        ext_aperture,_ = dlh_utils.order_width_B(n_ord,master_flat_zoom,oversample,c_all)

    #Fix the parameters so that they are all within in the chip
    extraction_width, column_range, c_all = dlh_utils.fix_parameters(ext_aperture, column_range, c_all, nrow, ncol, n_ord)
    ext_aperture,_,_=dlh_utils.fix_parameters(ext_aperture,column_range,c_all,nrow,ncol,n_ord)
    
    extracted = np.zeros((nord,ncol))
    x = np.arange(ncol)
    mask = np.full(img.shape, True)
        
    # Add mask as defined by column ranges
    mask1 = np.full((nord, ncol), True)
    for i in range(nord):
        mask1[i, column_range[i, 0] : column_range[i, 1]] = False
    extracted = np.ma.array(extracted, mask=mask1)
    
    for i in range(n_ord):
        x_left_lim = column_range[i, 0]
        x_right_lim = column_range[i, 1]

        # Rectify the image, i.e. remove the shape of the order
        # Then the center of the order is within one pixel variations
        ycen = np.polyval(c_all[i], x).astype(int)
        yb, yt = ycen - extraction_width[i,0], ycen + extraction_width[i,1]
        index = make_index(yb, yt, x_left_lim, x_right_lim)
        mask[index] = False
    
        extracted_ord = np.sum(img[index],axis=0)

        extracted[i,column_range[i, 0] : column_range[i, 1]] = extracted_ord
        
    
    peaks, tilt, shear, vec = determine_curvature_all_lines(
                img, extracted, column_range,extraction_width,c_all,oversample)


        
    coef_tilt, coef_shear = fit(peaks, tilt, shear,n_ord)

    iorder,ipeaks = np.indices(extracted.shape)
    
    tilt, shear = eval(ipeaks, iorder, coef_tilt, coef_shear)
    #Copy the solution from the P fibre (odd order number) to the O fibre
#    for i in range(1,nord,2):
#        tilt[i-1][:]=tilt[i][:]
    
    if (Plot == "True"):
        plot_comparison(img, tilt, shear, peaks,ext_aperture,n_ord,c_all,column_range)

    savefile = (out_location+"Curvature_"+arm+night+"_OS_"+str(oversample)+".npz")
    np.savez(savefile, tilt=tilt, shear=shear)
    return tilt, shear


