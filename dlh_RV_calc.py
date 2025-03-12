import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import constants as const
import scipy.constants as conts
from astropy.modeling import models, fitting
import math

import CCF_3d_cpython

from lmfit import  Model,Parameter
from lmfit.models import LinearModel, LorentzianModel, GaussianModel,VoigtModel
from astropy.time import Time
import barycorrpy
import glob

LIGHT_SPEED = const.c.to('km/s').value  # light speed in km/s
LIGHT_SPEED_M = const.c.value  # light speed in m/s
FIT_G = fitting.LevMarLSQFitter()

##################################################################################
# FUNCTION:defining a gaussian fit
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    #return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))
    #return (1./(wid*np.sqrt(2*np.pi))) * np.exp(-(x-cen)**2 / (2*wid**2))
    return amp*np.exp(-(x-cen)**2/(2*wid**2))
    
def calc_ccf(v_steps, new_line_start, new_line_end, x_pixel_wave, spectrum, new_line_weight, sn, zb, velocity_loop):
    """ Cross correlation by the shifted mask line and the spectrum data of one order for each velocity step.

    Args:
        v_steps (int): Total velocity steps.
        new_line_start (numpy.ndarray): Start of the mask line.
        new_line_end (numpy.ndarray): End of the mask line.
        x_pixel_wave (numpy.ndarray): Wavelength calibration of the pixels.
        spectrum (numpy.ndarray): 1D Spectrum data.
        new_line_weight (numpy.ndarray): Mask weight
        sn (numpy.ndarray): Additional SNR scaling factor (comply with the implementation of CCF of C version)
        zb (float): Redshift at the observation time.

    Returns:
        numpy.ndarray: ccf at velocity steps.
        numpy.ndarray: Intermediate CCF numbers at pixels.
    """

    ccf = np.zeros(v_steps)
    shift_lines_by = (1.0 + (velocity_loop / LIGHT_SPEED)) / (1.0 + zb)

    n_pixel = np.shape(x_pixel_wave)[0] - 1                # total size in  x_pixel_wave_start
    n_line_index = np.shape(new_line_start)[0]

    # pix1, pix2 = 10, n_pixel - 11
    pix1, pix2 = 0, n_pixel-1
    x_pixel_wave_end = x_pixel_wave[1: n_pixel+1]            # total size: n_pixel
    x_pixel_wave_start = x_pixel_wave[0: n_pixel]
    ccf_pixels = np.zeros([v_steps, n_pixel])

    for c in range(v_steps):
        line_doppler_shifted_start = new_line_start * shift_lines_by[c]
        line_doppler_shifted_end = new_line_end * shift_lines_by[c]
        # line_doppler_shifted_center =  new_line_center * shift_lines_by[c]

        # from the original:
        # closest_match = np.sum((x_pixel_wave_start - line_doppler_shifted_center[:,np.newaxis] <= 0.), axis=1)
        closest_match = np.sum((x_pixel_wave_end - line_doppler_shifted_start[:, np.newaxis] < 0.), axis=1)
        closest_match_next = np.sum((x_pixel_wave_start - line_doppler_shifted_end[:, np.newaxis] <= 0.), axis=1)
        mask_spectra_doppler_shifted = np.zeros(n_pixel)

        # this is from the original code, it may miss some pixels at the ends or work on more pixels than needed
        """
        for k in range(n_line_index):
            closest_x_pixel = closest_match[k] - 1    # fix: closest index before line_doppler_shifted_center
            # closest_x_pixel = closest_match[k]      # before fix

            line_start_wave = line_doppler_shifted_start[k]
            line_end_wave = line_doppler_shifted_end[k]
            line_weight = new_line_weight[k]

            if pix1 < closest_x_pixel < pix2:
                for n in range(closest_x_pixel - 5, closest_x_pixel + 5):
                    # if there is overlap
                    if x_pixel_wave_start[n] <= line_end_wave and x_pixel_wave_end[n] >= line_start_wave:
                        wave_start = max(x_pixel_wave_start[n], line_start_wave)
                        wave_end = min(x_pixel_wave_end[n], line_end_wave)
                        mask_spectra_doppler_shifted[n] = line_weight * (wave_end - wave_start) / \
                                                          (x_pixel_wave_end[n] - x_pixel_wave_start[n])

        """
        idx_collection = list()
        for k in range(n_line_index):
            closest_x_pixel = closest_match[k]  # closest index starting before line_dopplershifted_start
            closest_x_pixel_next = closest_match_next[k]  # closest index starting after line_dopplershifted_end
            line_start_wave = line_doppler_shifted_start[k]
            line_end_wave = line_doppler_shifted_end[k]
            line_weight = new_line_weight[k]

            if closest_x_pixel_next <= pix1 or closest_x_pixel >= pix2:
                continue
            else:
                for n in range(closest_x_pixel, closest_x_pixel_next):
                    if n > pix2:
                        break
                    if n < pix1:
                        continue
                    # if there is overlap
                    if x_pixel_wave_start[n] <= line_end_wave and x_pixel_wave_end[n] >= line_start_wave:
                        wave_start = max(x_pixel_wave_start[n], line_start_wave)
                        wave_end = min(x_pixel_wave_end[n], line_end_wave)
                        mask_spectra_doppler_shifted[n] = line_weight * (wave_end - wave_start) / \
                            (x_pixel_wave_end[n] - x_pixel_wave_start[n])

                        if n in idx_collection:
                            pass
                            # print(str(n), ' already taken')
                        else:
                            idx_collection.append(n)
        ccf_pixels[c, :] = spectrum * mask_spectra_doppler_shifted * sn
        ccf[c] = np.nansum(ccf_pixels[c, :])
    return ccf, ccf_pixels


def compute_mask_line(mask_path, v_steps, zb_min, zb_max, mask_width=0.5, air_to_vacuum=False):
        """ Calculate mask coverage based on the mask center, mask width, and redshift range over a period of time.

        The calculation includes the following steps,

            * collect the mask centers and weights from the mask file.
            * convert mask wavelength from air to vacuum environment if needed.
            * compute the start and end points around each center.
            * compute adjust start and end points around each center per velocity range and  maximum and minimum
              redshift from barycentric velocity correction over a period of time.

        Args:
            mask_path (str): Mask file path.
            v_steps (numpy.ndarray): Velocity steps at even interval.
            zb_min (float): Minimum redshift.
            zb_max (float): Maximum redshift.
            mask_width (float, optional): Mask width. Defaults to 0.25 (Angstrom)
            air_to_vacuum (bool, optional): A flag indicating if converting mask from air to vacuum environment.
                Defaults to False.

        Returns:
            dict: Information of mask. Please refer to `mask_line` in `Attributes` of this class.

        """

        line_center, line_weight = np.loadtxt(mask_path, dtype=float, unpack=True)  # load mask file
        #50 and 10000 work well generally
        line_weight = 1-line_weight
        ii=np.where(np.logical_and(line_weight > 0.1, line_weight< 1))
#        line_weight=1-line_weight[ii]
        line_center=line_center[ii]
#        ii=np.where(line_weight < 5000)
        line_weight=line_weight[ii]
#        line_center=line_center[ii]

        def air2vac(wl_air):
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
        
        def vac2air(wl_air):
            """
            Convert wavelengths vacuum to air wavelength
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
            wl_vac[ii] = wl_air[ii] / fact  # Convert to vacuum wavelength
            return wl_vac
        
        vacuum_to_air = False
        if vacuum_to_air:
            line_center = vac2air(line_center)
        
        air_to_vacuum = True
        if air_to_vacuum:
            line_center = air2vac(line_center)
            
        line_mask_width = line_center * (mask_width / LIGHT_SPEED)

        mask_line = {'start': line_center - line_mask_width,
                     'end': line_center + line_mask_width,
                     'center': line_center,
                     'weight': line_weight}
        
        dummy_start = mask_line['start'] * ((1.0 + (v_steps[0] / LIGHT_SPEED)) / (zb_max + 1.0))
        dummy_end = mask_line['end'] * ((1.0 + (v_steps[-1] / LIGHT_SPEED)) / (zb_min + 1.0))
        
        mask_line.update({'bc_corr_start': dummy_start, 'bc_corr_end': dummy_end})

        return mask_line


def get_mask_line(mask_path, v_steps, zb_range,  mask_width=0.5, air_to_vacuum=False):
    """ Get mask information.

    Args:
        mask_path (str): Mask file path.
        v_steps (numpy.ndarray): Velocity steps at even interval.
        zb_range (numpy.ndarray): Array containing Barycentric velocity correction minimum and maximum.
        mask_width (float, optional): Mask width (km/s). Defaults to 0.25.
        air_to_vacuum (bool, optional): A flag indicating if converting mask from air to vacuum environment.
            Defaults to False.

    Returns:
        dict: Information of mask. Please refer to `mask_line` in `Attributes` of this class.

    """
    mask_line = compute_mask_line(mask_path, v_steps, zb_range[0], zb_range[1], mask_width,air_to_vacuum)
    return mask_line


def get_velocity_steps(start_vel,step):
    """ Total velocity steps.

    Returns:
        int: Total velocity steps based on attribute `velocity_steps` of the class. Attribute `velocity_steps` is
        updated.

    """
    vel_loop = get_velocity_loop(start_vel,step)
    velocity_steps = len(vel_loop)
    return velocity_steps,vel_loop
    
def get_velocity_loop(start_vel,step):
    """ Get array of velocities based on step range, step interval, and estimated star radial velocity.

    Returns:
        numpy.ndarray: Array of evenly spaced velocities. Attribute `velocity_loop` is updated.

    """

    v_range = get_step_range()
    
    if start_vel is not None:
        velocity_loop = np.arange(0, v_range[1]-v_range[0], step) + \
                             start_vel
    return velocity_loop

def get_step_range(default=[-100, 101]):
    """ Get the step range for the velocity.

    Args:
        default (list): Default step range in string format. Defaults to '[-80, 81]'.

    Returns:
        list: Step range. `step_range` in attribute `rv_config` is updated.

    """
    default=[-50, 50]

    return default
    
def get_rv_config_value(prop, star_config=None, default=None):
    """ Get value of specific parameter from the config file or star config file.

    Check the value from the configuration file first, then from the star configuration file if it is available.
    The default is set if it is not defined in any configuration file.

    Args:
        prop (str): Name of the parameter to be searched.
        star_config (ConfigHandler): Section of designated star in star configuration file.
        default (Union[int, float, str, bool], optional): Default value for the searched parameter.
            Defaults to None.

    Returns:
        Union[int, float, str, bool]: Value for the searched parameter.

    """

    val = self.get_value_from_config(prop, default=None)

    # not exist in module config, check star config if there is or return default
    if val is None:
        if star_config is not None:
            return self.get_value_from_config(prop, config=star_config, default=default)
        else:
            return default

    if type(val) != str:
        return val

    # check if getting the value further from star config
    tag = 'star/'     # to find value from star config file
    if val.startswith(tag):
        if star_config is not None:
            attr = val[len(tag):]
            return self.get_value_from_config(attr, config=star_config, default=default)
        else:
            return default

    return val

        
def cross_correlate_by_mask_shift(wave_cal, spectrum, vb,start_vel, step):
        """Cross correlation by the shifted mask line and the spectrum data of one order for each velocity step.

        Args:
            wave_cal (numpy.ndarray): Wavelength calibration associated with `spectrum`.
            spectrum (numpy.ndarray): Reduced 1D spectrum data of one order from optimal extraction computation.
            vb (float): BC velocity (m/sec) at the observation time.

        Returns:
            numpy.ndarray: Cross correlation result of one order at all velocity steps. Please refer to `Returns` of
            function :func:`~alg.RadialVelocityAlg.get_rv_on_spectrum()` for cross correlation results of
            all orders.

        """
    
        v_steps,velocity_loop = get_velocity_steps(start_vel,step)
        ccf = np.zeros(v_steps)
        zb = vb/LIGHT_SPEED_M
        z_b = ((1.0/(1+zb)) - 1.0)
        v_b = z_b * LIGHT_SPEED # km/s
 
        mask_path = "G8_espresso.txt"
        mask_path = "F9_espresso.txt"
        #mask_path, air_to_vacuum = "test_VALD_new.list", False
        #mask_path = "thar_list.list"
        zb_range=[0,0]
      
        air_to_vacuum = False
        line = get_mask_line(mask_path,velocity_loop,zb_range,air_to_vacuum=air_to_vacuum)
        if line is None:
            return ccf
        # made some fix on line_index. the original calculation may miss some pixels at the edges while
        # finding the overlap between the wavelength range of the pixels and the maximum wavelength range of
        # the mask line
        # from the original

        line_index = np.where((line.get('bc_corr_start') > np.min(wave_cal)) &
                              (line.get('bc_corr_end') < np.max(wave_cal)))[0]
#
#        plt.vlines(line['center'],-1,line['weight'],'g')
#        plt.vlines(line['center'][line_index],0,line['weight'][line_index],'r')

#        plt.plot(wave_cal,spectrum)
#        plt.show()

        # line_index = np.where((line.get('bc_corr_end') > np.min(wave_cal)) &
        #                       (line.get('bc_corr_start') < np.max(wave_cal)))[0]
        n_line_index = len(line_index)
        if n_line_index == 0 or wave_cal.size <= 2:
            print("return early")
            return ccf

        n_pixel = np.shape(wave_cal)[0]

        new_line_start = line['start'][line_index]
        new_line_end = line['end'][line_index]
        new_line_center = line['center'][line_index]
        new_line_weight = line['weight'][line_index]

        x_pixel_wave_start = (wave_cal + np.roll(wave_cal, 1)) / 2.0  # w[0]-(w[1]-w[0])/2, (w[0]+w[1])/2.....
        x_pixel_wave_end = np.roll(x_pixel_wave_start, -1)            # (w[0]+w[1])/2,      (w[1]+w[2])/2....

        # pixel_wave_end = (wave_cal + np.roll(wave_cal,-1))/2.0      # from the original
        # pixel_wave_start[0] = wave_cal[0]
        # pixel_wave_end[-1] = wave_cal[-1]

        # fix
        x_pixel_wave_start[0] = wave_cal[0] - (wave_cal[1] - wave_cal[0]) / 2.0
        x_pixel_wave_end[-1] = wave_cal[-1] + (wave_cal[-1] - wave_cal[-2]) / 2.0

        x_pixel_wave = np.zeros(n_pixel + 1)
        x_pixel_wave[1:n_pixel] = x_pixel_wave_start[1:n_pixel]
        x_pixel_wave[n_pixel] = x_pixel_wave_end[-1]
        x_pixel_wave[0] = x_pixel_wave_start[0]

        ccf_code = 'c'

        # shift_lines_by = (1.0 + (self.velocity_loop / LIGHT_SPEED)) / (1.0 + zb)  # Shifting mask in redshift space
        if ccf_code == 'c':
            # ccf_pixels_c = np.zeros([v_steps, n_pixel])

            for c in range(v_steps):
                # add one pixel before and after the original array in order to uniform the calculation between c code
                # and python code
                new_wave_cal = np.pad(wave_cal, (1, 1), 'constant')
                new_wave_cal[0] = 2 * wave_cal[0] - wave_cal[1]     # w[0] - (w[1]-w[0])
                new_wave_cal[-1] = 2 * wave_cal[-1] - wave_cal[-2]  # w[n-1] + (w[n-1] - w[n-2])

                new_spec = np.pad(spectrum, (1, 1), 'constant')
                new_spec[1:n_pixel+1] = spectrum
                sn = np.ones(n_pixel+2)

                ccf[c] = CCF_3d_cpython.calc_ccf(new_line_start.astype('float64'), new_line_end.astype('float64'),
                                                 new_wave_cal.astype('float64'), new_spec.astype('float64'),
                                                 new_line_weight.astype('float64'), sn.astype('float64'),
                                                 velocity_loop[c], -v_b)
        else:
            sn_p = np.ones(n_pixel)
            ccf, ccf_pixels_python = calc_ccf(v_steps, new_line_start.astype('float64'),
                                                   new_line_end.astype('float64'),
                                                   x_pixel_wave.astype('float64'),
                                                   spectrum.astype('float64'),
                                                   new_line_weight.astype('float64'),
                                                   sn_p, -z_b,velocity_loop)



        return ccf,velocity_loop


def resample_spec(waveobs, fluxes, err, resampled_waveobs, bessel=False, zero_edges=True, frame=None):
    """
    Interpolate flux for a given wavelength by using Bessel's Central-Difference Interpolation.
    It considers:

    - 4 points in general
    - 2 when there are not more (i.e. at the beginning of the array or outside)

    * It does not interpolate if any of the fluxes used for interpolation is zero or negative
    this way it can respect gaps in the spectrum
    """
    last_reported_progress = -1
    current_work_progress = 10.0

    total_points = len(waveobs)
    new_total_points = len(resampled_waveobs)
    resampled_flux = np.zeros(new_total_points)
    resampled_err = np.zeros(new_total_points)
    from_index = 0 # Optimization: discard regions already processed
    for i in np.arange(new_total_points):
        # Target wavelength
        objective_wavelength = resampled_waveobs[i]

        # Find the index position of the first wave length equal or higher than the objective
        index = waveobs[from_index:].searchsorted(objective_wavelength)
        index += from_index

        if index == total_points:
            # DISCARD: Linear extrapolation using index-1 and index-2
            # flux = fluxes[index-1] + (objective_wavelength - waveobs[index-1]) * ((fluxes[index-1]-fluxes[index-2])/(waveobs[index-1]-waveobs[index-2]))
            if zero_edges:
                # JUST ZERO:
                resampled_flux[i] = 0.0
                resampled_err[i] = 0.0
            else:
                # JUST DUPLICATE:
                resampled_flux[i] = fluxes[index-1]
                resampled_err[i] = err[index-1]
        #elif index == 0 and waveobs[index] != objective_wavelength:
        elif index == 0:
            # DISCARD: Linear extrapolation using index+1 and index
            # flux = fluxes[index] + (objective_wavelength - waveobs[index]) * ((fluxes[index+1]-fluxes[index])/(waveobs[index+1]-waveobs[index]))
            if zero_edges:
                # JUST ZERO:
                resampled_flux[i] = 0.0
                resampled_err[i] = 0.0
            else:
                # JUST DUPLICATE:
                resampled_flux[i] = fluxes[index]
                resampled_err[i] = err[index]
        # Do not do this optimization because it can produce a value surounded
        # by zeros because of the condition "Do not interpolate if any of the
        # fluxes is zero or negative" implemented in the rest of the cases
        #elif waveobs[index] == objective_wavelength:
            #resampled_flux[i] = fluxes[index]
            #resampled_err[i] = err[index]
        else:
            if not bessel or index == 1 or index == total_points-1:
                # Do not interpolate if any of the fluxes is zero or negative
                if fluxes[index-1] <= 1e-10 or fluxes[index] <= 1e-10:
                    resampled_flux[i] = 0.0
                    resampled_err[i] = 0.0
                else:
                    # Linear interpolation between index and index-1
                    # http://en.wikipedia.org/wiki/Linear_interpolation#Linear_interpolation_between_two_known_points
                    d1 = (objective_wavelength - waveobs[index-1])
                    d2 = (waveobs[index]-waveobs[index-1])
                    resampled_flux[i] = fluxes[index-1] + d1 * ((fluxes[index]-fluxes[index-1])/d2)
                    # Same formula as for interpolation but I have re-arranged the terms to make
                    # clear that it is valid for error propagation (sum of errors multiplied by constant values)
                    resampled_err[i] = (err[index-1] * (d2 - d1)  + (err[index] * d1)) / d2
                    # Do not allow negative fluxes or errors
                    if resampled_err[i] < 0:
                        resampled_err[i] = 1e-10
                    if resampled_flux[i] < 0:
                        resampled_flux[i] = 0
                        resampled_err[i] = 0
            else:
                # Bessel's Central-Difference Interpolation with 4 points
                #   p = [(x - x0) / (x1 - x0)]
                #   f(x) = f(x0) + p ( f(x1) - f(x0) ) + [ p ( p - 1 ) / 4 ] ( f(x2) - f(x1) - f(x0) + f(x-1) )
                # where x-1 < x0 < objective_wavelength = x < x1 < x2 and f() is the flux
                #   http://physics.gmu.edu/~amin/phys251/Topics/NumAnalysis/Approximation/polynomialInterp.html

                #  x-1= index - 2
                #  x0 = index - 1
                #  x  = objective_wavelength
                #  x1 = index
                #  x2 = index + 1

                ## Array access optimization
                flux_x_1 = fluxes[index - 2]
                wave_x0 = waveobs[index-1]
                flux_x0 = fluxes[index - 1]
                wave_x1 = waveobs[index]
                flux_x1 = fluxes[index]
                flux_x2 = fluxes[index + 1]

                err_x_1 = err[index - 2]
                err_x0 = err[index - 1]
                err_x1 = err[index]
                err_x2 = err[index + 1]

                # Do not interpolate if any of the fluxes is zero or negative
                if flux_x_1 <= 1e-10 or flux_x0 <= 1e-10 or flux_x1 <= 1e-10 or flux_x2 <= 1e-10:
                    resampled_flux[i] = 0.0
                    resampled_err[i] = 0.0
                else:
                    p = (objective_wavelength - wave_x0)/ (wave_x1 - wave_x0)
                    factor = (p * (p - 1)/ 4)
                    resampled_flux[i] = flux_x0 + p * (flux_x1 - flux_x0) + factor * (flux_x2 - flux_x1 - flux_x0 + flux_x_1)
                    # Same formula as for interpolation but I have re-arranged the terms to make
                    # clear that it is valid for error propagation (sum of errors multiplied by constant values)
                    resampled_err[i] = err_x_1 * factor + err_x0 * (1 - p - factor) + err_x1 * (p - factor) + err_x2 * factor
                    # Do not allow negative fluxes or errors
                    if resampled_err[i] < 0:
                        resampled_err[i] = 1e-10
                    if resampled_flux[i] < 0:
                        resampled_flux[i] = 0
                        resampled_err[i] = 0

        if index > 4:
            from_index = index - 4

        current_work_progress = np.min([(i*1.0/ new_total_points) * 100, 90.0])
#        if report_progress(current_work_progress, last_reported_progress):
#            last_reported_progress = current_work_progress
#            #logging.info("%.2f%%" % current_work_progress)
#            if frame is not None:
#                frame.update_progress(current_work_progress)

    return resampled_waveobs, resampled_flux, resampled_err

def ccf_error_calc(velocities, ccfs, fit_wid, vel_span_pixel, rv_guess = 0.0):
        """Estimate photon-limited RV uncertainty of computed CCF.

           Calculate weighted slope information of CCF and convert to approximate RV uncertainty based on
           photon noise alone.

        Args:
            velocities (np.ndarray): velocity steps for CCF computation
            ccfs (np.ndarray): cross correlation results on velocities
            fit_wid (float): velocity width of the CCF.
            vel_span_pixel (float): approximate velocity span per CCD pixel.

        Returns:
            float: Estimated photon-limited uncertainty of RV measurement using specified ccf

        """
        vel_step = np.mean(np.diff(velocities))    # km/s,  velocity coverage per step
        n_scale_pix = vel_step / vel_span_pixel    # number of spectral pixels per ccf velocity step

        inds_fit = np.where((velocities >= (rv_guess - fit_wid / 2.)) & (velocities <= (rv_guess + fit_wid / 2.)))
        vels_fit = velocities[inds_fit]
        ccfs_fit  = ccfs[inds_fit]

        # the cases causing crashes
        if not ccfs_fit.any() or np.size(np.where(ccfs_fit < 0)[0]) > 0:
            return 0.0

        noise_ccf = (ccfs_fit) ** 0.5
        deriv_ccf = np.gradient(ccfs_fit, vels_fit)

        weighted_slopes = (deriv_ccf) ** 2. / (noise_ccf) ** 2.

        top = (np.sum(weighted_slopes)) ** 0.5
        bottom = (np.sum(ccfs_fit)) ** 0.5
        qccf = (top / bottom) * (n_scale_pix ** 0.5)
        sigma_ccf = 1. / (qccf * ((np.sum(ccfs_fit)) ** 0.5))  # km/s

        return sigma_ccf


def fit_ccf(result_ccf, rv_guess, velocities, mask_method=None, velocity_cut=50.0, rv_guess_on_ccf=False,
            vel_span_pixel=None):
    """Gaussian fitting to the values of cross correlation vs. velocity steps.

    Find the radial velocity from the summation of cross correlation values over orders by the use of
    Gaussian fitting and starting from a guessed value.

    Args:
        result_ccf (numpy.ndarray): 1D array containing summation the summation of cross correlation data over
            orders. Please refer to `Returns` of :func:`~alg.RadialVelocityAlg.get_rv_on_spectrum()`.
        rv_guess (float): Approximation of radial velocity.
        velocities (np.array): An array of velocity steps.
        mask_method (str): mask method for ccf, default to None.
        velocity_cut (float, optional): Range limit around the guessed radial velocity. Defaults to 100.0 (km/s).
        rv_guss_on_ccf (bool, optional): If doing rv guess per ccf values and mask method.
        vel_span_pixel (float, optional) Velocity width per pixel for rv error calculation.
    Returns:
        tuple: Gaussian fitting mean and values for the fitting,

            * **gaussian_fit** (*fitting.LevMarLSQFitter*): Instance for doing Gussian fitting based on
              Levenberg-Marquardt algorithm and least squares statistics.
            * **mean** (*float*): Mean value from Gaussian fitting.
            * **g_x** (*numpy.ndarray*): Collection of velocity steps for Gaussian fitting.
            * **g_y** (*numpy.ndarray*): Collection of cross correlation summation offset to the mean of
              cross correlation summation values along *g_x*.

    """
    if mask_method is not None:
        mask_method = mask_method.lower()
    if rv_guess_on_ccf:  # kpf get rv_guess from the ccf values
        # print('first guess: ', rv_guess)
        rv_guess, ccf_guess, ccf_dir = RadialVelocityAlg.rv_estimation_from_ccf_order(result_ccf, velocities,
                    rv_guess, mask_method)
        # print('second guess: ', rv_guess, ' mask: ', mask_method)
    else:
        ccf_dir = -1

    rv_error = 0.0
    if ccf_dir == 0 and rv_guess == 0.0:
        return None, rv_guess, None, None, rv_error

    def gaussian_rv(v_cut, rv_mean, sd):
        # amp = -1e7 if ccf_dir < 0 else 1e7
        ccf = result_ccf
        i_cut = (velocities >= rv_mean - v_cut) & (velocities <= rv_mean + v_cut)
        if not i_cut.any():
            return None, None, None
        g_x = velocities[i_cut]
        g_y = ccf[i_cut] - np.nanmedian(ccf[i_cut])
        y_dist = abs(np.nanmax(g_y) - np.nanmin(g_y)) * 100
        amp = max(-1e7, np.nanmin(g_y) - y_dist) if ccf_dir < 0 else min(1e7, np.nanmax(g_y) + y_dist)

        if sd is None:
            g_init = models.Gaussian1D(amplitude=amp, mean=rv_mean)
        else:
            g_init = models.Gaussian1D(amplitude=amp, mean=rv_mean, stddev=sd)

        gaussian_fit = FIT_G(g_init, g_x, g_y)
        return gaussian_fit, g_x, g_y


    two_fitting = True

#    # first gaussian fitting
#    if mask_method in RadialVelocityAlg.vel_range_per_mask.keys():
#        velocity_cut = RadialVelocityAlg.vel_range_per_mask[mask_method]
#        sd = 0.5            # for narrower velocity range
#        two_fitting = False
#    else:
    sd = 5.0
    g_fit, g_x, g_y = gaussian_rv(velocity_cut, rv_guess, sd)
    
#    plt.plot(g_x,g_y)
#    plt.plot(g_x,g_fit(g_x))
#    plt.show()

    if g_fit is not None \
            and g_x[0] <= g_fit.mean.value <= g_x[-1] \
            and two_fitting:
        v_cut = 25.0
        #print('mean before 2nd fitting: ', g_fit.mean.value)

        g_fit2, g_x2, g_y2 = gaussian_rv(v_cut, g_fit.mean.value, sd)
        if g_fit2 is not None and \
                not (g_x2[0] <= g_fit2.mean.value <= g_x2[-1]):
            # print('mean after 2nd fitting (out of range): ', g_fit2.mean.value)
            g_fit2 = None
    else:
        g_fit2 = None

    if vel_span_pixel is not None and vel_span_pixel != 0.0 and g_fit is not None:
        if g_fit2 is None or math.isnan(g_fit.mean.value):
            g_mean = rv_guess               # use the 1st guess if the 2nd fitting fails
            f_wid = velocity_cut            # use the 1st vel range if the 2nd fitting fails
        else:
            g_mean = g_fit.mean.value
            f_wid = v_cut

        rv_error = ccf_error_calc(velocities, result_ccf, f_wid*2, vel_span_pixel, g_mean)

    if g_fit2 is not None and not math.isnan(g_fit2.mean.value):
        return g_fit2, g_fit2.mean.value, g_x2, g_y2, rv_error
    elif g_fit is not None:
        return g_fit, (0.0 if math.isnan(g_fit.mean.value) else g_fit.mean.value), g_x, g_y, rv_error
    else:
        return None, 0.0, None, None, 0.0


#files = sorted(glob.glob("/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2024/0108/reduced/HRS_E_W_bogH20240108004?.fits"))
#This is the file with the wavelength solution e.g. Reference fibre frame
#file1 ="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2024/0108/reduced/HRS_E_W_bogH202401080021.fits"

#hdu = fits.open(file1)
#data1=hdu[0].data
#hdu.close
 
def execute(header,data,known_val):

    lat = -32.3722685109
    lon = 20.806403441
    alt = header["SITEELEV"]
    object = header["OBJECT"]

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


    BJD = barycorrpy.JDUTC_to_BJDTDB(jd, starname = object, lat=lat, longi=lon, alt=alt)
    BCV =(barycorrpy.get_BC_vel(JDUTC=jd,starname = object, lat=lat, longi=lon, alt=alt, leap_update=True))
        
    good_ord=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    good_ord=[6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
        
    rv = []
    rv_err=[]
    counter = 0
    vb=BCV[0]
    
    step=0.1
    #known_val = -16597 #m/s FOR TAU CETI
    #known_val = 72416 #m/s
    start_vel=-50+(known_val/1000.) #this should be -80+known val to get the trough of the CCF roughly in the centre of the fit.

    for order in good_ord:

        #print("Calculating CCF for order", order)
        order_wave = data[0][order][100:-100]
        order_spec = data[1][order][100:-100]
        #order_spec /= np.max(order_spec)
        #wavezzz=fits.getdata("./test.fits")
        #order_wave = wavezzz[0][order][100:-100]

        
        fit=np.polyfit(order_wave,order_spec,1)
        line=fit[0]*order_wave + fit[1]
#        plt.plot(order_wave,order_spec/line)
#        plt.plot(order_wave,order_spec)
#        plt.show()
        #order_spec /= line

        result_ccf, velocities = cross_correlate_by_mask_shift(order_wave, order_spec,vb, start_vel, step)
#        plt.plot(order_wave,order_spec)
#        plt.show()
        vels = np.asarray(velocities)
        ccf = np.asarray(result_ccf)
        peak_pos= np.where(ccf == np.min(ccf))

#        plt.plot(vels,ccf)
#        plt.show()

        vel_span_pixel = 3.3 #km/s
        
        _,rv_ord,_,_,rv_err_ord = fit_ccf(ccf,known_val/1000.,vels,vel_span_pixel=vel_span_pixel)

        if rv_err_ord > 0.:
        #print("RV ORD",rv_ord*1000., rv_err_ord*1000.)
            rv.append(rv_ord*1000.)
            rv_err.append(rv_err_ord*1000.)

    rv = np.array(rv)
    rv_err = np.array(rv_err)

    rv_mean = np.average(rv,weights=(1/rv_err**2),returned=True)
    
    #plt.xlabel("Time (BJD)")
    #plt.ylabel("RV (m/s)")
    #plt.show()
     
    #plt.plot(good_ord,rv,'bo')
#    plt.errorbar(good_ord , rv,yerr=rv_err,fmt='o')
#    plt.axhline(y = known_val, color = 'r', linestyle = '-')
#    plt.axhline(y = rv_mean[0], color = 'b', linestyle = '--')
#
#    y=rv
#    rms = np.sqrt(np.mean((rv_mean[0]-y)**2))
#    plt.axhline(y = rv_mean[0]+rms, color = 'b', linestyle = ':')
#    plt.axhline(y = rv_mean[0]-rms, color = 'b', linestyle = ':')
#    plt.show()
    
    return rv_mean[0],np.sqrt(1/rv_mean[1])


