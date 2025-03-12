import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit, least_squares
from scipy.stats import chisquare

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]
    
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
    
def gaussval2(x, a, mu, sig, const):
    return a * np.exp(-((x - mu) ** 2) / (2 * sig)) + const
    
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

def execute():
    #Test this out with data for HD 126053 CAL_RVST from 2023 03 13

    #Reference frame file
    arc = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0313/reduced/HRS_E_bogH202303130013.fits"

    target_file1 = "/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0313/reduced/HRS_E_bogH202303130032.fits"

    arc_data = fits.getdata(arc)
    target_data1 = fits.getdata(target_file1)
    #Even orders are O Fibre
    #Odd orders are P Fibre (simultaneous)
    
    nord=42
    
    out_put=np.zeros((nord))
    
    for ord in range(0,nord):
        O_arc = arc_data[ord*2][1]
        P_arc = arc_data[(ord*2)+1][1]
        
        x=np.arange(len(O_arc))
        
        O_arc /= np.max(O_arc)
        P_arc /= np.max(P_arc)
        
        O_targ = target_data1[ord*2][1]
        P_targ = target_data1[(ord*2)+1][1] /np.max(target_data1[ord+1][1])
        
        O_peaks = find_peaks(O_arc,height=0.001, distance=10)[0]
        P_peaks = find_peaks(P_arc,height=0.001, distance=10)[0]
        
        offsets = []
        #Peak width pixels
        width = 5
        
        #Fit gaussian to peaks for precise centre
        for peak in range(len(O_peaks)):
            coefs_O = fit_single_line(O_arc,O_peaks[peak],width)
     
            if np.logical_and(np.abs(coefs_O[1]-O_peaks[peak]) < 0.51,O_peaks[peak]>width):
                
                #Fit the corresponding peak in the P fibre, with knowledge that the offset is ~2.5 pixels.
                coefs_P = fit_single_line(P_arc,coefs_O[1]-2.5,width)
                
                #Chi-squared test of the peak fit, often bad given the lower counts in the science P fibre
                low = int(coefs_P[1] - width * 3)
                low = max(low, 0)
                high = int(coefs_P[1] + width * 3)
                high = min(high, len(P_arc))

                section = P_arc[low:high]-np.min(P_arc[low:high])
                x2 = np.arange(low, high, 1)
                x2 = np.ma.masked_array(x2, mask=np.ma.getmaskarray(section))
                fit=gaussval2(x2, *coefs_P)

                chi = sum(((section-fit)**2)/fit)
                
                if chi > 1e20 and np.logical_and(coefs_O[1]-coefs_P[1]>1.8,(coefs_O[1]-coefs_P[1])<4.):
                    #Now find the corresponding peak in the target frame, and fit its position
                    P_targ_peak = fit_single_line(P_targ,coefs_P[1],width)

                    #Chi-squared test of the peak fit, often bad given the lower counts in the science P fibre
                    low = int(coefs_P[1] - width * 3)
                    low = max(low, 0)
                    high = int(coefs_P[1] + width * 3)
                    high = min(high, len(P_targ))

                    section = P_targ[low:high]-np.min(P_targ[low:high])
                    x2 = np.arange(low, high, 1)
                    x2 = np.ma.masked_array(x2, mask=np.ma.getmaskarray(section))
                    fit=gaussval2(x2, *P_targ_peak)

                    chi = sum((section-fit)**2/fit)
    #                plt.plot(x2,section)
    #                plt.plot(x2,fit)
     
                    if chi > 1e20:
                        #plt.plot(P_targ_peak[1],P_targ_peak[1]-coefs_P[1],'o')
                        offsets.append(P_targ_peak[1]-coefs_P[1])
#        plt.plot(offset,'ro')
        offsets = np.array(offsets)
        offsets = reject_outliers(offsets)

        if ord ==41:
            print(offsets)
            plt.plot(P_arc)
            plt.plot(P_targ)
            plt.show()
            out_put[ord]= 0
        else:

            print("Order", ord, " Median offset (pixel)",np.median(offsets), "mean offset (pixel)",np.mean(offsets))
            out_put[ord]= np.median(offsets)
        
    return out_put
        #plt.plot(P_peak_cen,O_peak_cen-P_peak_cen,'o')
    #    plt.title("Order "+str(ord))
    #    plt.show()

    #    plt.plot(O_peak_cen,O_peak_height,'gx')
    #    plt.show()
