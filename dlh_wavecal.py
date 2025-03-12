import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate,correlation_lags
from scipy.linalg import lstsq
from scipy.special import binom
from scipy.constants import speed_of_light
from os.path import dirname, join
from tqdm import tqdm
from lmfit import  Model,Parameter
from scipy.optimize import curve_fit, least_squares
import dlh_sim_correction
from scipy.interpolate import interp1d
from numpy.polynomial.legendre import Legendre

from astropy.time import Time
import barycorrpy
import scipy.constants as conts

##################################################################################
# FUNCTION:defining a gaussian fit
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    #return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))
    #return (1./(wid*np.sqrt(2*np.pi))) * np.exp(-(x-cen)**2 / (2*wid**2))
    return amp*np.exp(-(x-cen)**2/(2*wid**2))
    
def air_to_vac(wl_air):
    """
    Convert wavelengths in air to vacuum wavelength
    Author: Nikolai Piskunov
    """
    wl_vac = np.copy(wl_air)
    ii = np.where(wl_air > 1999.352)

    sigma2 = (1e4 / wl_air[ii]) ** 2  # Compute wavenumbers squared
    fact = (
        1e0
        + 8.336624212083e-5
        + 2.408926869968e-2 / (1.301065924522e2 - sigma2)
        + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
    )
    wl_vac[ii] = wl_air[ii] * fact  # Convert to vacuum wavelength
    return wl_vac
   

##################################################################################
def gaussfit3(x, y):
    """A very simple (and relatively fast) gaussian fit
    gauss = A * exp(-(x-mu)**2/(2*sig**2)) + offset

    Parameters
    ----------
    x : array of shape (n,)
        x data
    y : array of shape (n,)
        y data

    Returns
    -------
    popt : list of shape (4,)
        Parameters A, mu, sigma**2, offset
    """
    mask = np.ma.getmaskarray(x) | np.ma.getmaskarray(y)
    x, y = x[~mask], y[~mask]

    gauss = gaussval2
    i = np.argmax(y[len(y) // 4 : len(y) * 3 // 4]) + len(y) // 4
    p0 = [y[i], x[i], 1, np.min(y)]

    with np.warnings.catch_warnings():
        np.warnings.simplefilter("ignore")
        popt, _ = curve_fit(gauss, x, y, p0=p0)

    return popt
    
def plot2d(x, y, z, coeff, title=None):
    # regular grid covering the domain of the data
    if x.size > 500:
        choice = np.random.choice(x.size, size=500, replace=False)
    else:
        choice = slice(None, None, None)
    x, y, z = x[choice], y[choice], z[choice]
    X, Y = np.meshgrid(
        np.linspace(np.min(x), np.max(x), 20), np.linspace(np.min(y), np.max(y), 20)
    )
    Z = np.polynomial.polynomial.polyval2d(X, Y, coeff)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(x, y, z, c="r", s=50)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")
    if title is not None:
        plt.title(title)
    # ax.axis("equal")
    # ax.axis("tight")
    plt.show()

##################################################################################

def _unscale(x, y, norm, offset):
    x = x * norm[0] + offset[0]
    y = y * norm[1] + offset[1]
    return x, y

##################################################################################

def gaussval2(x, a, mu, sig, const):
    return a * np.exp(-((x - mu) ** 2) / (2 * sig)) + const
    
##################################################################################

def gaussfit2(x, y):
    """Fit a gaussian(normal) curve to data x, y

    gauss = A * exp(-(x-mu)**2/(2*sig**2)) + offset

    Parameters
    ----------
    x : array[n]
        x values
    y : array[n]
        y values

    Returns
    -------
    popt : array[4]
        coefficients of the gaussian: A, mu, sigma**2, offset
    """

    gauss = gaussval2

    x = np.ma.compressed(x)
    y = np.ma.compressed(y)

    if len(x) == 0 or len(y) == 0:
        raise ValueError("All values masked")

    if len(x) != len(y):
        raise ValueError("The masks of x and y are different")

    # Find the peak in the center of the image
    weights = np.ones(len(y), dtype=y.dtype)
    midpoint = len(y) // 2
    weights[:midpoint] = np.linspace(0, 1, midpoint, dtype=weights.dtype)
    weights[midpoint:] = np.linspace(1, 0, len(y) - midpoint, dtype=weights.dtype)

    i = np.argmax(y * weights)
    p0 = [y[i], x[i], 1]
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")

    res = least_squares(
        lambda c: gauss(x, *c, np.ma.min(y)) - y,
        p0,
        loss="soft_l1",
        bounds=(
            [min(np.ma.mean(y), y[i]), np.ma.min(x), 0],
            [np.ma.max(y) * 1.5, np.ma.max(x), len(x) / 2],
        ),
    )
    popt = list(res.x) + [np.min(y)]
    return popt
    
####################################################################
    
def fit_single_line(obs, center, width, plot=False):
    low = int(center - width * 3)
    low = max(low, 0)
    high = int(center + width * 3)
    high = min(high, len(obs))

    section = obs[low:high]-np.min(obs[low:high])
    x = np.arange(low, high, 1)
    x = np.ma.masked_array(x, mask=np.ma.getmaskarray(section))

    if (np.isnan(np.sum(x)) or np.isnan(np.sum(section))):
        return
    else:
        coef = gaussfit2(x, section)
        
    if plot == "True":
        x2 = np.linspace(x.min(), x.max(), len(x) * 100)
        plt.plot(x, section, label="Observation")
        plt.plot(x2, gaussval2(x2, *coef), label="Fit")
        title = "Gaussian Fit to spectral line"
#        if self.plot_title is not None:
#            title = f"{self.plot_title}\n{title}"
        plt.title(title)
        plt.xlabel("x [pixel]")
        plt.ylabel("Intensity [a.u.]")
        plt.legend()
        plt.show()
    return coef
       
####################################################################
       
def fit_lines(obs, lines):
    """
    Determine exact position of each line on the detector based on initial guess

    This fits a Gaussian to each line, and uses the peak position as a new solution

    Parameters
    ----------
    obs : array of shape (nord, ncol)
        observed wavelength calibration image
    lines : recarray of shape (nlines,)
        reference line data

    Returns
    -------
    lines : recarray of shape (nlines,)
        Updated line information (posm is changed)
    """
    # For each line fit a gaussian to the observation
    for i, line in tqdm(
        enumerate(lines), total=len(lines), leave=False, desc="Lines"
    ):
        if(np.logical_and(line["width"] > 1.2,line["width"]<6)):

            coef = fit_single_line(
                obs[int(line["order"])],
                line["posm"],
                2.3548*line["width"],
                plot=line["flag"],
            )

            if coef is not None:

                lines[i]["posm"] = coef[1]
                lines[i]["height"] = coef[0]
                lines[i]["width"] = 2.3548*np.sqrt(coef[2])

                # Gaussian fit failed, dont use line
                if coef[2] > 8.35:
                    lines[i]["flag"] = False
                if coef[0] < 300:
                    lines[i]["flag"] = False
            else:
                lines[i]["flag"] = False
        else:
            lines[i]["flag"] = False

    return lines

################################################################
    
def make_wave(wave_solution, nord, ncol, plot=False):
    """Expand polynomial wavelength solution into full image

    Parameters
    ----------
    wave_solution : array of shape(degree,)
        polynomial coefficients of wavelength solution
    plot : bool, optional
        wether to plot the solution (default: False)

    Returns
    -------
    wave_img : array of shape (nord, ncol)
        wavelength solution for each point in the spectrum
    """

    y, x = np.indices((nord, ncol))
    wave_img = evaluate_solution(x, y, wave_solution)

    return wave_img

##################################################################################

def calculate_AIC(lines, wave_solution):

#    print("DLH3", wave_solution.item())
#
#    if dimensionality == "1D":
#        k = 1
#        for _, v in wave_solution.item():
#            k += np.size(v[0])
#            k += np.size(v[1])
#    elif dimensionality == "2D":
#        k = 1
#        poly_coef, steps_coef = wave_solution
#        for _, v in steps_coef.items():
#            k += np.size(v)
#        k += np.size(poly_coef)
##    else:
    k = np.size(wave_solution) + 1

    # We get the residuals in velocity space
    # but need to remove the speed of light component, to get dimensionless parameters
    x = lines["posm"]
    y = lines["order"]
    mask = ~lines["flag"]
    solution = evaluate_solution(x, y, wave_solution)
    rss = (solution - lines["wll"]) / lines["wll"]

    # rss = self.calculate_residual(wave_solution, lines)
    # rss /= speed_of_light
    n = rss.size
    rss = np.ma.sum(rss ** 2)

    # As per Wikipedia https://en.wikipedia.org/wiki/Akaike_information_criterion
    logl = np.log(rss)
    aic = 2 * k + n * logl
    logl = logl
    aicc = aic + (2 * k ** 2 + 2 * k) / (n - k - 1)
    aic = aic
    return aic

##################################################################################

def plot_results(wave_img, obs,nord,ncol):
    plt.subplot(211)
    plot_title = "Results"
    title = "Wavelength solution with Wavelength calibration spectrum\nOrders are in different colours"
    if plot_title is not None:
        title = f"{plot_title}\n{title}"
    plt.title(title)
    plt.xlabel("Wavelength")
    plt.ylabel("Observed spectrum")
    for i in range(nord):
        plt.plot(wave_img[i], obs[i], label="Order %i" % i)

    plt.subplot(212)
    plt.title("2D Wavelength solution")
    plt.imshow(
        wave_img, aspect="auto", origin="lower", extent=(0, ncol, 0, nord)
    )
    cbar = plt.colorbar()
    plt.xlabel("Column")
    plt.ylabel("Order")
    cbar.set_label("Wavelength [Å]")
    plt.show()

    
##################################################################################

def auto_id(obs, wave_img, lines,atlas,nord,threshold):
    """Automatically identify peaks that are close to known lines

    Parameters
    ----------
    obs : array of shape (nord, ncol)
        observed spectrum
    wave_img : array of shape (nord, ncol)
        wavelength solution image
    lines : struc_array
        line data
    threshold : int, optional
        difference threshold between line positions in m/s, until which a line is considered identified (default: 1)
    plot : bool, optional
        wether to plot the new lines

    Returns
    -------
    lines : struct_array
        line data with new flags
    """

    new_lines = []
    if atlas is not None:
        # For each order, find the corresponding section in the Atlas
        # Look for strong lines in the atlas and the spectrum that match in position
        # Add new lines to the linelist
        width_of_atlas_peaks = 3
        for order in range(obs.shape[0]):
            mask = ~np.ma.getmaskarray(obs[order])
            index_mask = np.arange(len(mask))[mask]
            data_obs = obs[order, mask]
            data_obs[np.isnan(data_obs)] = 0
            data_obs /= np.max(data_obs)
            wave_obs = wave_img[order, mask]

            threshold_of_peak_closeness = (
                np.diff(wave_obs) / wave_obs[:-1] * speed_of_light
            )
            threshold_of_peak_closeness = np.max(threshold_of_peak_closeness)
            wmin, wmax = wave_obs[0], wave_obs[-1]
            imin, imax = np.searchsorted(atlas.wave, (wmin, wmax))
            wave_atlas = atlas.wave[imin:imax]
            data_atlas = atlas.flux[imin:imax]
            if len(data_atlas) == 0:
                continue
            data_atlas = data_atlas/ data_atlas.max()

            line = lines[
                (lines["order"] == order)
                & (lines["wll"] > wmin)
                & (lines["wll"] < wmax)
            ]

#            peaks_atlas, peak_info_atlas = find_peaks(
#                data_atlas, height=0.001, width=width_of_atlas_peaks
#            )
            
            peaks_atlas = np.loadtxt('thar_list_new_2.txt',usecols=1,unpack=True)
            ii=np.where(np.logical_and(peaks_atlas>=wmin, peaks_atlas<=wmax))
            peaks_atlas = peaks_atlas[ii]
            peaks_obs, peak_info_obs = find_peaks(
                data_obs, height=0.00005, width=1
            )

            for i, p in enumerate(peaks_atlas):
                # Look for an existing line in the vicinity
                #wpeak = wave_atlas[p]
                wpeak = p
                diff = np.abs(line["wll"] - wpeak) / wpeak * speed_of_light
                if np.any(diff < threshold_of_peak_closeness):
                    # Line already in the linelist, ignore
                    continue
                else:
                    # Look for matching peak in observation
                    diff = (
                        np.abs(wpeak - wave_obs[peaks_obs]) / wpeak * speed_of_light
                    )
                    imin = np.argmin(diff)

                    if diff[imin] < threshold_of_peak_closeness:
                        # Add line to linelist
                        # Location on the detector
                        # Include the masked areas!!!
                        ipeak = peaks_obs[imin]
                        ipeak = index_mask[ipeak]

                        # relative height of the peak
                        hpeak = data_obs[peaks_obs[imin]]
                        wipeak = peak_info_obs["widths"][imin]
                        # wave, order, pos, width, height, flag
                        new_lines.append([wpeak, order, ipeak, wipeak, hpeak, True])

        # Add new lines to the linelist
        if len(new_lines) != 0:
            new_lines = np.array(new_lines).T
            new_lines = LineList.from_list(*new_lines)
            new_lines = fit_lines(obs, new_lines)
            lines = np.append(lines,new_lines)

    # Option 1:
    # Step 1: Loop over unused lines in lines
    # Step 2: find peaks in neighbourhood
    # Step 3: Toggle flag on if close
    counter = 0
    for i, line in enumerate(lines):
        if line["flag"]:
            # Line is already in use
            continue
        if line["order"] < 0 or line["order"] >= nord:
            # Line outside order range
            continue
        iord = int(line["order"])
        if line["wll"] < wave_img[iord][0] or line["wll"] >= wave_img[iord][-1]:
            # Line outside pixel range
            continue

        wl = line["wll"]
        width = line["width"] * 5
        wave = wave_img[iord]
        order_obs = obs[iord]
        # Find where the line should be
        try:
            idx = np.digitize(wl, wave)
        except ValueError:
            # Wavelength solution is not monotonic
            idx = np.where(wave >= wl)[0][0]

        low = int(idx - width)
        low = max(low, 0)
        high = int(idx + width)
        high = min(high, len(order_obs))

        vec = order_obs[low:high]
        if np.all(np.ma.getmaskarray(vec)):
            continue
        # Find the best fitting peak
        # TODO use gaussian fit?
        peak_idx, _ = find_peaks(vec, height=0.00005, width=3)
        if len(peak_idx) > 0:
            peak_pos = np.copy(peak_idx).astype(float)
            for j in range(len(peak_idx)):
                try:
                    coef = fit_single_line(vec, peak_idx[j], line["width"])
                    peak_pos[j] = coef[1]
                except:
                    peak_pos[j] = np.nan
                    pass

            pos_wave = np.interp(peak_pos, np.arange(high - low), wave[low:high])
            residual = np.abs(wl - pos_wave) / wl * speed_of_light
            idx = np.argmin(residual)
            if residual[idx] < threshold:
                counter += 1
                lines["flag"][i] = True
                lines["posm"][i] = low + peak_pos[idx]

    print("AutoID identified {} new lines".format(str(counter + len(new_lines))))
    return lines
    
    ##################################################################################


def reject_outlier(residual, lines):
    """
    Reject the strongest outlier

    Parameters
    ----------
    residual : array of shape (nlines,)
        residuals of all lines
    lines : recarray of shape (nlines,)
        line data

    Returns
    -------
    lines : struct_array
        line data with one more flagged line
    residual : array of shape (nlines,)
        residuals of each line, with outliers masked (including the new one)
    """

    # Strongest outlier
    ibad = np.ma.argmax(np.abs(residual))
    lines["flag"][ibad] = False

    return lines

##################################################################################

def calculate_residual(wave_solution, lines):
    """
    Calculate all residuals of all given lines

    Residual = (Wavelength Solution - Expected Wavelength) / Expected Wavelength * speed of light

    Parameters
    ----------
    wave_solution : array of shape (degree_x, degree_y)
        polynomial coefficients of the wavelength solution (in numpy format)
    lines : recarray of shape (nlines,)
        contains the position of the line on the detector (posm), the order (order), and the expected wavelength (wll)

    Returns
    -------
    residual : array of shape (nlines,)
        Residual of each line in m/s
    """
    x = lines["posm"]
    y = lines["order"]
    mask = ~lines["flag"]

    solution = evaluate_solution(x, y, wave_solution)

    residual = (solution - lines["wll"]) / lines["wll"] * speed_of_light
    residual = np.ma.masked_array(residual, mask=mask)
    return residual
    
    ##################################################################################

def build_2d_solution(lines, nord, degree, ncol, plot=False):
    """
    Create a 2D polynomial fit to flagged lines
    degree : tuple(int, int), optional
        polynomial degree of the fit in (column, order) dimension (default: (6, 6))

    Parameters
    ----------
    lines : struc_array
        line data
    plot : bool, optional
        wether to plot the solution (default: False)

    Returns
    -------
    coef : array[degree_x, degree_y]
        2d polynomial coefficients
    """

    dimensionality = "2D"

    # Only use flagged data
    mask = lines["flag"]  # True: use line, False: dont use line
    m_wave = lines["wll"][mask]
    m_pix = lines["posm"][mask]
    m_ord = lines["order"][mask]

    if dimensionality == "1D":
        coef = np.zeros((nord, degree + 1))
        for i in range(nord):
            select = m_ord == i
            if np.count_nonzero(select) < 2:
                # Not enough lines for wavelength solution
                logger.warning(
                    "Not enough valid lines found wavelength calibration in order % i",
                    i,
                )
                coef[i] = np.nan
                continue

            deg = max(min(degree, np.count_nonzero(select) - 2), 0)
            coef[i, -(deg + 1) :] = np.polyfit(
                m_pix[select], m_wave[select], deg=deg
            )
    elif dimensionality == "2D":
        # 2d polynomial fit with: x = column, y = order, z = wavelength
        coef = polyfit2d(m_pix, m_ord, m_wave, degree=degree, plot=False)
    else:
        raise ValueError(
            f"Parameter 'mode' not understood. Expected '1D' or '2D' but got {dimensionality}"
        )

#DLH Diagnostic plot commented out
#    if plot or plot >= 2:  # pragma: no cover
#        plot_residuals(lines, coef, ncol, title="Residuals")

    return coef
    
#####################################################################################

def polyfit2d(
    x, y, z, degree=1, max_degree=None, scale=True, plot=False, plot_title=None
):
    """A simple 2D plynomial fit to data x, y, z
    The polynomial can be evaluated with numpy.polynomial.polynomial.polyval2d

    Parameters
    ----------
    x : array[n]
        x coordinates
    y : array[n]
        y coordinates
    z : array[n]
        data values
    degree : int, optional
        degree of the polynomial fit (default: 1)
    max_degree : {int, None}, optional
        if given the maximum combined degree of the coefficients is limited to this value
    scale : bool, optional
        Wether to scale the input arrays x and y to mean 0 and variance 1, to avoid numerical overflows.
        Especially useful at higher degrees. (default: True)
    plot : bool, optional
        wether to plot the fitted surface and data (slow) (default: False)

    Returns
    -------
    coeff : array[degree+1, degree+1]
        the polynomial coefficients in numpy 2d format, i.e. coeff[i, j] for x**i * y**j
    """
    # Flatten input
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    z = np.asarray(z).ravel()

    # Removed masked values
    mask = ~(np.ma.getmask(z) | np.ma.getmask(x) | np.ma.getmask(y))
    x, y, z = x[mask].ravel(), y[mask].ravel(), z[mask].ravel()

    if scale:
        x, y, norm, offset = _scale(x, y)

    # Create combinations of degree of x and y
    # usually: [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), ....]
    if np.isscalar(degree):
        degree = (int(degree), int(degree))
    assert len(degree) == 2, "Only 2D polynomials can be fitted"
    degree = [int(degree[0]), int(degree[1])]
    # idx = [[i, j] for i, j in product(range(degree[0] + 1), range(degree[1] + 1))]
    coeff = np.zeros((degree[0] + 1, degree[1] + 1))
    idx = _get_coeff_idx(coeff)

    # Calculate elements 1, x, y, x*y, x**2, y**2, ...
    A = polyvander2d(x, y, degree)

    # We only want the combinations with maximum order COMBINED power
    if max_degree is not None:
        mask = idx[:, 0] + idx[:, 1] <= int(max_degree)
        idx = idx[mask]
        A = A[:, mask]

    # Do least squares fit
    C, *_ = lstsq(A, z)

    # Reorder coefficients into numpy compatible 2d array
    for k, (i, j) in enumerate(idx):
        coeff[i, j] = C[k]

    # # Backup copy of coeff
    if scale:
        coeff = polyscale2d(coeff, *norm, copy=False)
        coeff = polyshift2d(coeff, *offset, copy=False)

    if plot:  # pragma: no cover
        if scale:
            x, y = _unscale(x, y, norm, offset)
        plot2d(x, y, z, coeff, title=plot_title)

    return coeff

#####################################################################################

def reject_lines(lines, nord, ncol, degree,threshold ,plot=False):
    """
    Reject the largest outlier one by one until all residuals are lower than the threshold

    Parameters
    ----------
    lines : recarray of shape (nlines,)
        Line data with pixel position, and expected wavelength
    threshold : float, optional
        upper limit for the residual, by default 100
    degree : tuple, optional
        polynomial degree of the wavelength solution (pixel, column) (default: (6, 6))
    plot : bool, optional
        Wether to plot the results (default: False)

    Returns
    -------
    lines : recarray of shape (nlines,)
        Line data with updated flags
    """

    wave_solution = build_2d_solution(lines, nord, degree,ncol)
    residual = calculate_residual(wave_solution, lines)
    nbad = 0
    while np.ma.any(np.abs(residual) > threshold):
        lines = reject_outlier(residual, lines)
        wave_solution = build_2d_solution(lines, nord, degree,ncol)
        residual = calculate_residual(wave_solution, lines)
        nbad += 1
    print("Discarding {} lines".format(nbad))

    if plot or plot >= 2:  # pragma: no cover
        mask = lines["flag"]
        _, axis = plt.subplots()
        axis.plot(lines["order"][mask], residual[mask], "X", label="Accepted Lines")
        axis.plot(
            lines["order"][~mask], residual[~mask], "D", label="Rejected Lines"
        )
        axis.set_xlabel("Order")
        axis.set_ylabel("Residual [m/s]")
        axis.set_title("Residuals versus order")
        axis.legend()

        fig, ax = plt.subplots(
            nrows=nord // 2, ncols=2, sharex=True, squeeze=False
        )
        plt.subplots_adjust(hspace=0)
        fig.suptitle("Residuals of each order versus image columns")

        for iord in range(nord):
            order_lines = lines[lines["order"] == iord]
            solution = evaluate_solution(
                order_lines["posm"], order_lines["order"], wave_solution
            )
            # Residual in m/s
            residual = (
                (solution - order_lines["wll"])
                / order_lines["wll"]
                * speed_of_light
            )
            mask = order_lines["flag"]
            ax[iord // 2, iord % 2].plot(
                order_lines["posm"][mask],
                residual[mask],
                "X",
                label="Accepted Lines",
            )
            ax[iord // 2, iord % 2].plot(
                order_lines["posm"][~mask],
                residual[~mask],
                "D",
                label="Rejected Lines",
            )
            # ax[iord // 2, iord % 2].tick_params(labelleft=False)
            ax[iord // 2, iord % 2].set_ylim(
                -threshold * 1.5, +threshold * 1.5
            )

        ax[-1, 0].set_xlabel("x [pixel]")
        ax[-1, 1].set_xlabel("x [pixel]")

        ax[0, 0].legend()

        plt.show()
    return lines

#####################################################################################


def _scale(x, y):
    # Normalize x and y to avoid huge numbers
    # Mean 0, Variation 1
    offset_x, offset_y = np.mean(x), np.mean(y)
    norm_x, norm_y = np.std(x), np.std(y)
    if norm_x == 0:
        norm_x = 1
    if norm_y == 0:
        norm_y = 1
    x = (x - offset_x) / norm_x
    y = (y - offset_y) / norm_y
    return x, y, (norm_x, norm_y), (offset_x, offset_y)

#####################################################################################

def _get_coeff_idx(coeff):
    idx = np.indices(coeff.shape)
    idx = idx.T.swapaxes(0, 1).reshape((-1, 2))
    # degree = coeff.shape
    # idx = [[i, j] for i, j in product(range(degree[0]), range(degree[1]))]
    # idx = np.asarray(idx)
    return idx


#####################################################################################

def polyvander2d(x, y, degree):
    # A = np.array([x ** i * y ** j for i, j in idx], dtype=float).T
    A = np.polynomial.polynomial.polyvander2d(x, y, degree)
    return A

#####################################################################################

def find_peaks_py(comb):
    # Find peaks in the comb spectrum
    # Run find_peak twice
    # once to find the average distance between peaks
    # once for real (disregarding close peaks)
    c = comb - np.ma.min(comb)
    width = 3.
    height = np.ma.median(c)
    #DLH Mod -- my mods currently commented.
    height = 0.00005
    peaks, _ = find_peaks(c, height=height, width=width)
    #peaks, _ = signal.find_peaks(c)
    width = 1
    distance = np.median(np.diff(peaks)) // 16
    distance=5
    peaks, _ = find_peaks(c, height=height, distance=distance, width=width)
    #peaks, _ = signal.find_peaks(c,distance=6)

    # Fit peaks with gaussian to get accurate position
    new_peaks = peaks.astype(float)
    peaks_fwhm = peaks.astype(float)
    width = np.mean(np.diff(peaks)) // 2
    for j, p in enumerate(peaks):
        idx = p + np.arange(-width, width + 1, 1)
        idx = np.clip(idx, 0, len(c) - 1).astype(int)
        try:
            coef = gaussfit3(np.arange(len(idx)), c[idx])
            new_peaks[j] = coef[1] + p - width
            peaks_fwhm[j] = 2.355*np.sqrt(coef[2])
        except RuntimeError:
            new_peaks[j] = p

    n = np.arange(len(peaks))

    # keep peaks within the range
    mask = (new_peaks > 0) & (new_peaks < len(c))
    n, new_peaks,peaks_fwhm = n[mask], new_peaks[mask],peaks_fwhm[mask]

    return n, new_peaks
    
def gaussval2(x, a, mu, sig, const):
    return a * np.exp(-((x - mu) ** 2) / (2 * sig)) + const
    
def polyshift2d(coeff, offset_x, offset_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    # Copy coeff because it changes during the loop
    coeff2 = np.copy(coeff)
    for k, m in idx:
        not_the_same = ~((idx[:, 0] == k) & (idx[:, 1] == m))
        above = (idx[:, 0] >= k) & (idx[:, 1] >= m) & not_the_same
        for i, j in idx[above]:
            b = binom(i, k) * binom(j, m)
            sign = (-1) ** ((i - k) + (j - m))
            offset = offset_x ** (i - k) * offset_y ** (j - m)
            coeff[k, m] += sign * b * coeff2[i, j] * offset
    return coeff

#####################################################################################


def polyscale2d(coeff, scale_x, scale_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    for k, (i, j) in enumerate(idx):
        coeff[i, j] /= scale_x ** i * scale_y ** j
    return coeff
    
#####################################################################################
def evaluate_solution(pos, order, solution):
    """
    Evaluate the 1d or 2d wavelength solution at the given pixel positions and orders

    Parameters
    ----------
    pos : array
        pixel position on the detector (i.e. x axis)
    order : array
        order of each point
    solution : array of shape (nord, ndegree) or (degree_x, degree_y)
        polynomial coefficients. For mode=1D, one set of coefficients per order.
        For mode=2D, the first dimension is for the positions and the second for the orders
    mode : str, optional
        Wether to interpret the solution as 1D or 2D polynomials, by default "1D"

    Returns
    -------
    result: array
        Evaluated polynomial

    Raises
    ------
    ValueError
        If pos and order have different shapes, or mode is of the wrong value
    """
    if not np.array_equal(np.shape(pos), np.shape(order)):
        raise ValueError("pos and order must have the same shape")
        
    dimensionality = "2D"

#    if step_mode:
#        return evaluate_step_solution(pos, order, solution)

    if dimensionality == "1D":
        result = np.zeros(pos.shape)
        for i in np.unique(order):
            select = order == i
            result[select] = np.polyval(solution[int(i)], pos[select])
    elif dimensionality == "2D":
        result = np.polynomial.polynomial.polyval2d(pos, order, solution)
    else:
        raise ValueError(
            f"Parameter 'mode' not understood, expected '1D' or '2D' but got {self.dimensionality}"
        )
    return result

#####################################################################################

def evaluate_step_solution(self, pos, order, solution):
    if not np.array_equal(np.shape(pos), np.shape(order)):
        raise ValueError("pos and order must have the same shape")
    if self.dimensionality == "1D":
        result = np.zeros(pos.shape)
        for i in np.unique(order):
            select = order == i
            result[select] = self.f(
                pos[select],
                solution[i][0],
                solution[i][1][:, 0],
                solution[i][1][:, 1],
            )
    elif self.dimensionality == "2D":
        poly_coef, step_coef = solution
        pos = np.copy(pos)
        for i in np.unique(order):
            pos[order == i] = self.g(
                pos[order == i], step_coef[i][:, 0], step_coef[i][:, 1]
            )
        result = polyval2d(pos, order, poly_coef)
    else:
        raise ValueError(
            f"Parameter 'mode' not understood, expected '1D' or '2D' but got {self.dimensionality}"
        )
    return result

#####################################################################################
    
class LineList:
    dtype = np.dtype(
        (
            np.record,
            [
                (("wlc", "WLC"), ">f8"),  # Wavelength (before fit)
                (("wll", "WLL"), ">f8"),  # Wavelength (after fit)
                (("posc", "POSC"), ">f8"),  # Pixel Position (before fit)
                (("posm", "POSM"), ">f8"),  # Pixel Position (after fit)
                (("xfirst", "XFIRST"), ">i2"),  # first pixel of the line
                (("xlast", "XLAST"), ">i2"),  # last pixel of the line
                (
                    ("approx", "APPROX"),
                    "O",
                ),  # Not used. Describes the shape used to approximate the line. "G" for Gaussian
                (("width", "WIDTH"), ">f8"),  # width of the line in pixels
                (("height", "HEIGHT"), ">f8"),  # relative strength of the line
                (("order", "ORDER"), ">i2"),  # echelle order the line is found in
                ("flag", "?"),  # flag that tells us if we should use the line or not
            ],
        )
    )

    def __init__(self, lines=None):
        if lines is None:
            lines = np.array([], dtype=self.dtype)
        self.data = lines
        self.dtype = self.data.dtype

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    @classmethod
    def load(cls, filename):
        data = np.load(filename, allow_pickle=True)
        linelist = cls(data["cs_lines"])
        return linelist

    def save(self, filename):
        np.savez(filename, cs_lines=self.data)

    def append(self, linelist):
        if isinstance(linelist, LineList):
            linelist = linelist.data
        self.data = np.append(self.data, linelist)

    def add_line(self, wave, order, pos, width, height, flag):
        lines = self.from_list([wave], [order], [pos], [width], [height], [flag])
        self.data = np.append(self.data, lines)

    @classmethod
    def from_list(cls, wave, order, pos, width, height, flag):
        lines = [
            (w, w, p, p, p - wi / 2, p + wi / 2, b"G", wi, h, o, f)
            for w, o, p, wi, h, f in zip(wave, order, pos, width, height, flag)
        ]
        lines = np.array(lines, dtype=cls.dtype)
        return cls(lines)
        
####################################################################

class LineAtlas:
    def __init__(self, element, medium="vac"):
        self.element = element
        self.medium = medium
        #fname = element.lower() + "_utextas_spec.dat"
        fname = element.lower() + "_best.fits"
        folder = "./"#dirname(__file__)
        self.fname = join(folder, fname)
        self.wave, self.flux = self.load_fits(self.fname)

        
        try:
            # If a specific linelist file is provided
            fname_list = element.lower() + "_list_new_2.txt"
            fname_list = "Thorium_mask_031921.mas"
            self.fname_list = join(folder, fname_list)
            #linelist = np.genfromtxt(self.fname_list, dtype="U8,f8,f8")
            linelist = np.genfromtxt(self.fname_list, dtype="f8,f8")
            #element, wpos, heights = linelist["f0"], linelist["f1"], linelist["f2"]
            wpos, heights = linelist["f0"], linelist["f1"]
            #DLH commented below and added f2 in above line
            indices = self.wave.searchsorted(wpos)
 

#            self.linelist = np.rec.fromarrays(
#                [wpos, heights, element], names=["wave", "heights", "element"]
#            )
            self.linelist = np.rec.fromarrays(
                [wpos, heights], names=["wave", "heights"]
            )
            
            
            #DLH MOD to force using only identified lines from file.
            #self.wave, self.flux = self.linelist.wave, self.linelist.heights
            
        except (FileNotFoundError, IOError):
            # Otherwise fit the line positions from the spectrum
#            logger.warning(
#                "No dedicated linelist found for %s, determining peaks based on the reference spectrum instead.",
#                element,
#            )
            #module = WavelengthCalibration(plot=False)
            n, peaks = find_peaks_py(self.flux)
            wpos = np.interp(peaks, np.arange(len(self.wave)), self.wave)
            element = np.full(len(wpos), element)
            indices = self.wave.searchsorted(wpos)
            heights = self.flux[indices]
            
            #Convert to vaccuum
            wl_air = wpos
            wl_vac=wpos
            ii = np.where(wl_vac > 2e3)

            sigma2 = (1e4 / wl_vac[ii]) ** 2  # Compute wavenumbers squared
            fact = (
                1e0
                + 8.336624212083e-5
                + 2.408926869968e-2 / (1.301065924522e2 - sigma2)
                + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
                )
            wl_air[ii] = wl_vac[ii] * fact  # Convert to air wavelength
            wpos = wl_air[ii]
            
            self.linelist = np.rec.fromarrays(
                [wpos, heights, element], names=["wave", "heights", "element"]
            )
        # The data files are in vaccuum, if the instrument is in air, we need to convert
        if medium == "air":
            self.wave = util.vac2air(self.wave)
            self.linelist["wave"] = util.vac2air(self.linelist["wave"])

    def load_fits(self, fname):
        hdu = fits.open(fname)
  
        if len(hdu) == 1:
            # Its just the spectrum
            # with the wavelength defined via the header keywords
            header = hdu[0].header
            spec = hdu[0].data.ravel()
            wmin = header["CRVAL1"]
            wdel = header["CDELT1"]
            wave = np.arange(spec.size) * wdel + wmin

        else:
            # Its a binary Table, with two columns for the wavelength and the
            # spectrum
            data = hdu[1].data
            wave = data["wave"]
            spec = data["spec"]

        spec /= np.nanmax(spec)
        spec = np.clip(spec, 0, None)
        return wave, spec



def execute(file,arm,sci_frame,Plot,degree,offset):
#This takes the processed science frame, calcualtes the wavelength solution and returns the BC wavelength
      
    #Start with a pre determined guess and some known quantities
    if arm =="H":
        line_list = np.load("/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/DLH_Codes_combined/TEST_Wave_Sol/hrs_hs.H.linelist.npz",allow_pickle=True)
        nord=42
        #degree=[4,2]
        threshold = 100

    if arm =="R":
        line_list = np.load("/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/datasets/HRS/reduced/hrs_hs.R.linelist.npz",allow_pickle=True)
        nord= 32
        degree=[4,6]
        threshold = 400
        
    #Read the reference arc file DLH to move location
    hdu=fits.open('thar_best.fits')
    thar= hdu[0].data.ravel()
    wmin = hdu[0].header["CRVAL1"]
    wdel = hdu[0].header["CDELT1"]
    thar_x=np.arange(thar.size) * wdel + wmin
    hdu.close

    #Open the data file
    hdu=fits.open(file)
    data=hdu[0].data
    hdr=hdu[0].header
    hdu.close
    
    ncol=data.shape[2]
    #X axis for the observed file
    x=np.arange(len(data[1][1]))

    linelist=LineList()
    lines=line_list['cs_lines']
    
    #Correct the wavelengths to vaccuum
#    lines['wll'] = air_to_vac(lines['wll'])
#    lines['wlc'] = air_to_vac(lines['wlc'])

    #Apply the pixel offset
    lines['posm'] += offset
    lines['posc'] += offset
        
    atlas = LineAtlas('thar', 'vac')

    P_Fibre = np.zeros((nord,ncol))
    O_Fibre = np.zeros((nord,ncol))

    if arm == "R":
        old_data=fits.getdata("/Users/daniel/Desktop/SALT_HRS_DATA/Red/2024/1102/reduced/HRS_E_bogR202411029999.fits")
#    print(old_data.shape)
    for w in range(0,nord):
        O_Fibre[w] = data[w*2][1]
        P_Fibre[w] = data[(w*2)+1][1]


    #If it's the ARC reference file, we want to use this to determine the wavelength solution for the night.
    if sci_frame == "Arc":
        lines=fit_lines(O_Fibre,lines)
        
        
        #Select the initial positions
        m_pix = lines["posm"]
        m_ord = lines["order"]
        m_wave = lines["wll"]
        m_height= lines["height"]
        
#        '''
#        Residual = (Wavelength Solution - Expected Wavelength) / Expected Wavelength * speed of light
#        '''
#
#        for o in range(0,nord):
#            ii=np.where(m_ord == o)
#            ii=ii[0]
#            #plt.plot(m_pix[ii],m_wave[ii],'.')
#            P=np.polyfit(m_pix[ii],m_wave[ii],3)
#            y_fit=np.polyval(P,m_pix[ii])
#            res = ((y_fit -m_wave[ii])/m_wave[ii]) * speed_of_light
#            plt.plot(m_wave[ii],res,'.')
#            jj=np.where(res > 10000)
#            jj=jj[0]
#            lines['flag'][ii[jj]] = False
#        plt.show()


#        offsets = dlh_sim_correction.execute()
                
        wave_solution = polyfit2d(m_pix, m_ord, m_wave, degree=degree, plot=False)
        wave_img = make_wave(wave_solution,nord,ncol)

        lines = auto_id(O_Fibre, wave_img, lines,atlas,nord,threshold)
        lines = reject_lines(lines,nord,ncol,degree,threshold, plot=False)
        
        
#        for ord in range(nord):
#            ii=np.where(lines['order'] == ord)[0]
#            print("DLH_in",lines['posm'][ii][0])
#            lines['posm'][ii] += offsets[ord]
#            print("DLH_out",lines['posm'][ii][0])
        
        # Step 6: build final 2d solution
        wave_solution = build_2d_solution(lines, nord, degree, ncol,plot=False)
        wave_img = make_wave(wave_solution,nord,ncol)


        if Plot == "True":
            plot_results(wave_img, O_Fibre,nord,ncol)
        
        aic = calculate_AIC(lines, wave_solution)
        print("AIC of wavelength fit:", aic)
        print("Number of lines used for wavelength calibration:",np.count_nonzero(lines["flag"]))

        if Plot == "True":
            peaks_atlas = np.loadtxt('thar_list_new_2.txt',usecols=1,unpack=True)
            #plt.plot(thar_x,thar)
            plt.xlim(3600,5600)
            for ord in range(nord):
                plt.plot(wave_img[ord],O_Fibre[ord])
            jj=np.where(lines["flag"] == True)
            plt.plot(lines["wll"][jj[0]],lines["height"][jj[0]], 'go')
            jj=np.where(lines["flag"] == False)
            plt.plot(lines["wll"][jj[0]],lines["height"][jj[0]], 'ro')
            plt.vlines(peaks_atlas,ymin=0,ymax=200000,ls='--')

            plt.show()
            
        return wave_img, O_Fibre, wave_img, P_Fibre

    
    else:
        lines= fit_lines(P_Fibre,lines)
    
        #Select the initial positions
        m_pix = lines["posm"]
        m_ord = lines["order"]
        m_wave = lines["wll"]
        m_height= lines["height"]

        wave_solution = polyfit2d(m_pix, m_ord, m_wave, degree=degree, plot=False)
        wave_img = make_wave(wave_solution,nord,ncol)

        lines = auto_id(obs, wave_img, lines,atlas,nord,threshold)
        lines = reject_lines(lines,nord,ncol,degree,threshold, plot=False)

        # Step 6: build final 2d solution
        wave_solution = build_2d_solution(lines, nord, degree, ncol,plot=False)
        wave_img = make_wave(wave_solution,nord,ncol)
        

        #Plot = "True"

        if Plot == "True":
            plot_results(wave_img, obs,nord,ncol)
        
        aic = calculate_AIC(lines, wave_solution)
        print("AIC of wavelength fit:", aic)
        print("Number of lines used for wavelength calibration:",np.count_nonzero(lines["flag"]))

        Plot = "True"

        if Plot == "True":
            plt.plot(thar_x,thar)
            for ord in range(nord):
                plt.plot(wave_img[ord],obs[ord])

            plt.show()


        header=hdu[0].header

        obs_date = header["DATE-OBS"]
        ut = header["TIME-OBS"]

        if obs_date is not None and ut is not None:
            obs_date = f"{obs_date}T{ut}"
        fwmt = header["EXP-MID"]
        et = header["EXPTIME"]

        if fwmt > 0.:
            mid = float(fwmt)/86400.
        else:
            mid =  float(float(et)/2./86400.)

        jd = Time(obs_date,scale='utc',format='isot').jd + mid

        lat = -32.3722685109
        lon = 20.806403441
        alt = header["SITEELEV"]
    
    if sci_frame == "Science":
        object = header["OBJECT"]
        BCV =(barycorrpy.get_BC_vel(JDUTC=jd,starname = object, lat=lat, longi=lon, alt=alt, leap_update=True))
        BJD = barycorrpy.JDUTC_to_BJDTDB(jd, starname = object, lat=lat, longi=lon, alt=alt)

        #Apply the BC
        wave_corr = ((wave_img)*(1.0+(BCV[0]/conts.c)))
    else:
        wave_corr = wave_img
    
#    for i in range(0,nord):
#        plt.plot(wave_corr[i],sci[i],'--')
#    plt.show()
    
    return wave_corr, O_Fibre, wave_img, P_Fibre
#    return wave_img, obs

