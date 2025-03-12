import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime
import pytz
import scipy
from scipy import interpolate
from scipy.signal import correlate, find_peaks
from multiprocessing import Pool
from scipy.signal import savgol_filter

import os

import Marsh

import glob

"""
===================================================================================================================
"""

def gaussbroad(x, y, hwhm):
    """
    Apply gaussian broadening to x, y data with half width half maximum hwhm

    Parameters
    ----------
    x : array(float)
        x values
    y : array(float)
        y values
    hwhm : float > 0
        half width half maximum
    Returns
    -------
    array(float)
        broadened y values
    """

    # alternatively use:
    # from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
    # but that doesn't have an x coordinate

    nw = len(x)
    dw = (x[-1] - x[0]) / (len(x) - 1)

    if hwhm > 5 * (x[-1] - x[0]):
        return np.full(len(x), sum(y) / len(x))

    nhalf = int(3.3972872 * hwhm / dw)
    ng = 2 * nhalf + 1  # points in gaussian (odd!)
    # wavelength scale of gaussian
    wg = dw * (np.arange(0, ng, 1, dtype=float) - (ng - 1) / 2)
    xg = (0.83255461 / hwhm) * wg  # convenient absisca
    gpro = (0.46974832 * dw / hwhm) * np.exp(-xg * xg)  # unit area gaussian w/ FWHM
    gpro = gpro / np.sum(gpro)

    # Pad spectrum ends to minimize impact of Fourier ringing.
    npad = nhalf + 2  # pad pixels on each end
    spad = np.concatenate((np.full(npad, y[0]), y, np.full(npad, y[-1])))

    # Convolve and trim.
    sout = np.convolve(spad, gpro)  # convolve with gaussian
    sout = sout[npad : npad + nw]  # trim to original data / length
    return sout  # return broadened spectrum.


"""
===================================================================================================================
"""

def gaussfit(x, y):
    """
    Fit a simple gaussian to data

    gauss(x, a, mu, sigma) = a * exp(-z**2/2)
    with z = (x - mu) / sigma

    Parameters
    ----------
    x : array(float)
        x values
    y : array(float)
        y values
    Returns
    -------
    gauss(x), parameters
        fitted values for x, fit paramters (a, mu, sigma)
    """

    gauss = lambda x, A0, A1, A2: A0 * np.exp(-(((x - A1) / A2) ** 2) / 2)
    popt, _ = curve_fit(gauss, x, y, p0=[max(y), 0, 1])
    return gauss(x, *popt), popt
    
"""
===================================================================================================================
"""

def create_masterbias(all_files,path,arm,night,out_location,Plot):
    #Heavily based on PyReduce combine_bias in combine_frames.py

    #Test if Master Bias exists
    master_file = glob.glob(out_location+"Master_Bias_"+arm+night+".fits")

    if len(master_file) == 0:

        #Open files to check if they are the correct OBSTYPE (Bias)
        Bias_files = []
        if len(all_files) > 0:
            for file in all_files:
                file_night = file.removeprefix(out_location)[3:11]
                if(file_night == night):
                    hdu=fits.open(file)
                    if (hdu[0].header["OBSTYPE"] == "Bias"):
                        Bias_files.append(file)
                    hdu.close
        else:
            print ("\n   !!! No files found in {}. Check the arm. Exiting.\n".format(path))
            
        if len(Bias_files) <1:
            print ("\n   !!! No Bias files found in {}. Check night and arm. Exiting.\n".format(path))
            exit()
            
        n = len(Bias_files)
        if n == 0:
            print ("\n   !!! No Bias files found in {}. Check night and arm. Exiting.\n".format(path))
            exit()
        elif n == 1:
            #if there is just one element compare it with itself, not really useful, but it works
            list1 = list2 = Bias_files
            n = 2
        else:
            list1, list2 = Bias_files[: n // 2], Bias_files[n // 2 :]

        # Lists of images.
        n1 = len(list1)
        n2 = len(list2)

        bias_concat1 = []
        bias_concat2 = []
        Bias_files_short = []
        # Separate images in two groups.
        for file in list1:
            Bias_files_short.append(file.lstrip(path))
            hdu=fits.open(file)
            bias_concat1.append(hdu[0].data.astype(np.float64))
            hdu.close
        bias1 = np.median(bias_concat1, axis=0)
        
        for file in list2:
            Bias_files_short.append(file.lstrip(path))
            hdu=fits.open(file)
            bias_concat2.append(hdu[0].data.astype(np.float64))
            hdu.close
        bias2 = np.median(bias_concat2, axis=0)
        
        #Plot images side by side
        if (Plot == "True"):
            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(bias1, vmin=np.min(bias1), vmax=np.max(bias1), origin="lower")
            axarr[1].imshow(bias2, vmin=np.min(bias1), vmax=np.max(bias1), origin="lower")
            plt.show()
            
    #    # Make sure we know the gain. Since we can have multiple amplifies, take the mean of the possible values
        hdu = fits.open(list1[0])
        gain = hdu[0].header["GAIN"]
        if arm =="H":
            gain1 = float(gain.split()[0])
            gain2 = float(gain.split()[1])
            gain = gain1
        if arm == "R":
            gain1 = float(gain.split()[0])
            gain2 = float(gain.split()[1])
            gain3 = float(gain.split()[2])
            gain4 = float(gain.split()[3])
            gain = gain1 #np.mean([gain1,gain2,gain3,gain4])
        
        # Construct normalized sum.
#        bias1 *= gain
#        bias2 *= gain
        bias = (((bias1 * n1 + bias2 * n2) / n))

        # Compute noise in difference image by fitting Gaussian to distribution.
        diff = 0.5 * (bias1 - bias2)
        if np.min(diff) != np.max(diff):
            crude = np.ma.median(np.abs(diff))  # estimate of noise
            hmin = -5.0 * crude
            hmax = +5.0 * crude
            bin_size = np.clip(2 / n, 0.5, None)
            nbins = int((hmax - hmin) / bin_size)

            h, _ = np.histogram(diff, range=(hmin, hmax), bins=nbins)
            xh = hmin + bin_size * (np.arange(0.0, nbins) + 0.5)

            hfit, par = gaussfit(xh, h)
            noise = abs(par[2])  # noise in diff, bias

            # Determine where wings of distribution become significantly non-Gaussian.
            contam = (h - hfit) / np.sqrt(np.clip(hfit, 1, None))
            imid = np.where(abs(xh) < 2 * noise)
            consig = np.std(contam[imid])

            smcontam = gaussbroad(xh, contam, 0.1 * noise)
            igood = np.where(smcontam < 3 * consig)
            gmin = np.min(xh[igood])
            gmax = np.max(xh[igood])

            # Find and fix bad pixels.
            ibad = np.where((diff <= gmin) | (diff >= gmax))
            nbad = len(ibad[0])

            bias[ibad] = np.clip(bias1[ibad], None, bias2[ibad])
            bias = np.int16(bias)
            
            # Compute read noise.
            biasnoise = noise
            bgnoise = biasnoise * np.sqrt(n)

            # Print diagnostics.
            if Plot =="True":
                print("change in bias between image sets= ",  np.abs(par[1])," electons")
                print("measured background noise per image=", bgnoise)
                print("background noise in combined image=", biasnoise)
                print("Number of bad pixels fixed %i", nbad)

        else:
            diff = 0
            biasnoise = 1.0
            nbad = 0

        #Write the master bias to file with approraite header info
        new_hdu = fits.PrimaryHDU(data=bias)
        new_hdu.header.insert(5,('COMMENT',"  FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
        new_hdu.header.insert(6,('COMMENT',"  and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"))
        new_hdu.header['FIFPORT'] = (str(hdu[0].header['FIFPORT']), "FIF port selection")
        new_hdu.header['OBJECT'] = (str('Master_Bias'), "Object name")
        new_hdu.header['OBSERVAT'] = (str('SALT'), "South African Large Telescope")
        new_hdu.header['SITEELEV'] = (str(1798.), "Site elevation")
        new_hdu.header['SITELAT'] = (str(-32.3795), "Geographic latitude of the observation")
        new_hdu.header['SITELONG'] = (str(20.812), "Geographic longitude of the observation")
        new_hdu.header['AMPSEC'] = (str(hdu[0].header['AMPSEC']),"Amplifier Section")
        new_hdu.header['BIASSEC'] = (str(hdu[0].header['BIASSEC']),"Bias Section")
        new_hdu.header['CCDNAMPS'] = (str(hdu[0].header['CCDNAMPS']), "No. of amplifiers used")
        new_hdu.header['CCDSEC'] = (str(hdu[0].header['CCDSEC']), "CCD Section")
        new_hdu.header['CCDSUM'] = (str(hdu[0].header['CCDSUM']), "On-chip binning")
        new_hdu.header['CCDTYPE'] = (str('Bias'),"Observation type")
        new_hdu.header['DATASEC'] = (str(hdu[0].header['DATASEC']),"Data Section")
        new_hdu.header['DATE-OBS'] = (str(hdu[0].header['DATE-OBS']),"Date of observation")
        new_hdu.header['DETMODE'] = (str(hdu[0].header['DETMODE']),"Detector Mode")
        new_hdu.header['DETNAM'] = (str(hdu[0].header['DETNAM']),"Detector Name")
        new_hdu.header['DETSEC'] = (str(hdu[0].header['DATASEC']), "Detector Section")
        new_hdu.header['DETSER'] = (str(hdu[0].header['DETSER']), "Detector serial number")
        new_hdu.header['DETSIZE'] = (str(hdu[0].header['DETSIZE']), "Detector Size")
        new_hdu.header['DETSWV'] = (str(hdu[0].header['DETSWV']),"Detector software version")
        new_hdu.header['EXPTIME'] = (str(hdu[0].header['EXPTIME']),"Exposure time (s)")
        new_hdu.header['GAIN'] = (str(gain),"CCD gain (photons/ADU)")
        new_hdu.header['GAINSET'] = (str(hdu[0].header['GAINSET']),"Gain Setting")
        new_hdu.header['INSTRUME'] = (str('HRS'),"Instrument name: High Resolution Spectrograph")
        new_hdu.header['NCCDS'] = (str(hdu[0].header['NCCDS']),"Number of CCDs")
        new_hdu.header['NODCOUNT'] = (str(hdu[0].header['NODCOUNT']), "No. of Nod/Shuffles")
        new_hdu.header['NODPER'] = (str(hdu[0].header['NODPER']), "Nod & Shuffle Period (s)")
        new_hdu.header['NODSHUFF'] = (str(hdu[0].header['NODSHUFF']), "Nod & Shuffle enabled?")
        new_hdu.header['OBSMODE'] = (str(hdu[0].header['OBSMODE']),"Observation mode")
        new_hdu.header['OBSTYPE'] = (str('Bias'), "Observation type")
        new_hdu.header['PRESCAN'] = (str(hdu[0].header['PRESCAN']), "Prescan pixels at start/end of line")
        new_hdu.header['ROSPEED'] = (str(hdu[0].header['ROSPEED']),"CCD readout speed (Hz)")
        new_hdu.header['GAIN_p'] = (str(gain),"Gain that has been applied")
        new_hdu.header['BGNOIS'] = (str(bgnoise), "Background noise per image")
        new_hdu.header['BNOISE'] = (str(biasnoise),"Background noise in combined image")
        new_hdu.header['BIAVAR'] = (str(np.abs(par[1])), "Change in bias between image sets")
        DATE_EXT=str(datetime.now(tz=pytz.UTC).strftime("%Y-%m-%d"))
        UTC_EXT = str(datetime.now(tz=pytz.UTC).strftime("%H:%M:%S.%f"))
        new_hdu.header['DATE-EXT'] = (DATE_EXT,'Date file created')
        new_hdu.header['UTC-EXT'] = (UTC_EXT,'Time file created')
        new_hdu.header['N_FILE'] = (str(n),"Number of files combined")
        new_hdu.header['HISTORY'] = ("Files used for Master: "+str(Bias_files_short))
                
        new_hdu.writeto(str(out_location)+"Master_Bias_"+str(arm)+str(night)+".fits",overwrite=True)
        hdu.close
    
    else:
        print("\n      +++ Reading Master Bias frame "+master_file[0]+"\n")
        bias_hdu=fits.open(master_file[0])
        bias=bias_hdu[0].data
        bias_hdu.close
    
    if Plot == "True":  # pragma: no cover
        title = "Master Bias"
        plt.title(title)
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        bot, top = np.percentile(bias, (1, 99))
        plt.imshow(bias, vmin=bot, vmax=top, origin="lower")
        plt.show()
        
    
    return bias

"""
===================================================================================================================
"""


def create_masterflat(all_files,path,mode,arm,night,master_bias,out_location,Plot):

    #Test if Master Bias exists
    master_file = glob.glob(out_location+"Master_Flat_"+arm+night+".fits")
    if len(master_file) == 0:

        #Open files to check if they are the correct OBSTYPE (Flat Field)
        Flat_files = []
        if len(all_files) > 0:
            for file in all_files:
                file_night = file.removeprefix(out_location)[3:11]
                if(file_night == night):
                    hdu=fits.open(file)
                    if (hdu[0].header["OBSTYPE"] == "Flat field" and hdu[0].header["FIFPORT"] == mode):
                        Flat_files.append(file)
                    hdu.close
        else:
            print ("\n   !!! No files found in {}. Check the arm and night. Exiting.\n".format(path))
        n=len(Flat_files)
        if n <1:
            print ("\n   !!! No Flat files found in {}. Check arm ({}) and night ({}). Exiting.\n".format(path,arm,night))
            exit()
            
        Flat_concat = []
        gain = hdu[0].header["GAIN"]
        
        if arm =="H":
            gain1 = float(gain.split()[0])
            gain2 = float(gain.split()[1])
            gain = gain1
            
        if arm == "R":
            gain1 = float(gain.split()[0])
            gain2 = float(gain.split()[1])
            gain3 = float(gain.split()[2])
            gain4 = float(gain.split()[3])
            gain = gain1
        Flat_files_short = []
        
        jd_mean = []
        PRE_DEW =[]
        PRE_VAC =[]
        TEM_AIR =[]
        TEM_BCAM=[]
        TEM_COLL=[]
        TEM_ECH =[]
        TEM_IOD =[]
        TEM_OB  =[]
        TEM_RCAM=[]
        TEM_RMIR=[]
        TEM_VAC =[]
        
        for file in Flat_files:
            Flat_files_short.append(file.lstrip(out_location))
            hdu=fits.open(file)
            #Perform a test to reject bad files
            if np.std(hdu[0].data) > 200:
                #Gain correct and subtract master bias
                flat_data = (hdu[0].data) - master_bias
                hdu.close
                #Overwrite the data in the file with the corrected information and write to a new file prefixed with b (for bais corrected) and g (for gain corrected)
                hdu[0].data = flat_data
                file_out=str(out_location+"b"+file.removeprefix(out_location))
                hdu.writeto(file_out,overwrite=True)
                Flat_concat.append(flat_data)#/flat_mean)
                jd_mean.append(float(hdu[0].header['JD']))
                PRE_DEW.append(float(hdu[0].header['PRE-DEW']))
                PRE_VAC.append(float(hdu[0].header['PRE-VAC']))
                TEM_AIR.append(float(hdu[0].header['TEM-AIR']))
                TEM_BCAM.append(float(hdu[0].header['TEM-BCAM']))
                TEM_COLL.append(float(hdu[0].header['TEM-COLL']))
                TEM_ECH.append(float(hdu[0].header['TEM-ECH']))
                TEM_IOD.append(float(hdu[0].header['TEM-IOD']))
                TEM_OB.append(float(hdu[0].header['TEM-OB']))
                TEM_RCAM.append(float(hdu[0].header['TEM-RCAM']))
                TEM_RMIR.append(float(hdu[0].header['TEM-RMIR']))
                TEM_VAC.append(float(hdu[0].header['TEM-VAC']))
            else:
            #Change the file name to show it is bad
                os.rename(file, str(path+"Bad_Flat_"+file.removeprefix(out_location)[2:]))
                hdu.close
        jd_mean = np.array(jd_mean)
        jd_mean = np.mean(jd_mean)
        
        PRE_DEW =np.array(PRE_DEW)
        PRE_VAC =np.array(PRE_VAC)
        TEM_AIR =np.array(TEM_AIR)
        TEM_BCAM=np.array(TEM_BCAM)
        TEM_COLL=np.array(TEM_COLL)
        TEM_ECH =np.array(TEM_ECH)
        TEM_IOD =np.array(TEM_IOD)
        TEM_OB  =np.array(TEM_OB)
        TEM_RCAM=np.array(TEM_RCAM)
        TEM_RMIR=np.array(TEM_RMIR)
        TEM_VAC =np.array(TEM_VAC)
        
        #Create the master flat and write to new FITS file.
        master_flat = np.median(Flat_concat,axis=0)
        master_flat = np.float32(master_flat)
        new_hdu = fits.PrimaryHDU(data=master_flat)
        new_hdu.header.insert(5,('COMMENT',"  FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
        new_hdu.header.insert(6,('COMMENT',"  and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"))
        new_hdu.header['FIFPORT'] = (str(hdu[0].header['FIFPORT']), "FIF port selection")
        new_hdu.header['OBJECT'] = (str('Master_Flat'), "Object name")
        new_hdu.header['OBSERVAT'] = (str('SALT'), "South African Large Telescope")
        new_hdu.header['SITEELEV'] = (str(1798.), "Site elevation")
        new_hdu.header['SITELAT'] = (str(-32.3795), "Geographic latitude of the observation")
        new_hdu.header['SITELONG'] = (str(20.812), "Geographic longitude of the observation")
        new_hdu.header['AMPSEC'] = (str(hdu[0].header['AMPSEC']),"Amplifier Section")
        new_hdu.header['BIASSEC'] = (str(hdu[0].header['BIASSEC']),"Bias Section")
        new_hdu.header['CCDNAMPS'] = (str(hdu[0].header['CCDNAMPS']), "No. of amplifiers used")
        new_hdu.header['CCDSEC'] = (str(hdu[0].header['CCDSEC']), "CCD Section")
        new_hdu.header['CCDSUM'] = (str(hdu[0].header['CCDSUM']), "On-chip binning")
        new_hdu.header['CCDTYPE'] = (str('Flat'),"Observation type")
        new_hdu.header['DATASEC'] = (str(hdu[0].header['DATASEC']),"Data Section")
        new_hdu.header['DATE-OBS'] = (str(hdu[0].header['DATE-OBS']),"Date of observation")
        new_hdu.header['DETMODE'] = (str(hdu[0].header['DETMODE']),"Detector Mode")
        new_hdu.header['DETNAM'] = (str(hdu[0].header['DETNAM']),"Detector Name")
        new_hdu.header['DETSEC'] = (str(hdu[0].header['DATASEC']), "Detector Section")
        new_hdu.header['DETSER'] = (str(hdu[0].header['DETSER']), "Detector serial number")
        new_hdu.header['DETSIZE'] = (str(hdu[0].header['DETSIZE']), "Detector Size")
        new_hdu.header['DETSWV'] = (str(hdu[0].header['DETSWV']),"Detector software version")
        new_hdu.header['EXPTIME'] = (str(hdu[0].header['EXPTIME']),"Exposure time (s)")
        new_hdu.header['GAIN'] = (str(gain),"CCD gain (photons/ADU)")
        new_hdu.header['GAINSET'] = (str(hdu[0].header['GAINSET']),"Gain Setting")
        new_hdu.header['INSTRUME'] = (str('HRS'),"Instrument name: High Resolution Spectrograph")
        new_hdu.header['NCCDS'] = (str(hdu[0].header['NCCDS']),"Number of CCDs")
        new_hdu.header['NODCOUNT'] = (str(hdu[0].header['NODCOUNT']), "No. of Nod/Shuffles")
        new_hdu.header['NODPER'] = (str(hdu[0].header['NODPER']), "Nod & Shuffle Period (s)")
        new_hdu.header['NODSHUFF'] = (str(hdu[0].header['NODSHUFF']), "Nod & Shuffle enabled?")
        new_hdu.header['OBSMODE'] = (str(hdu[0].header['OBSMODE']),"Observation mode")
        new_hdu.header['OBSTYPE'] = (str('Flat'), "Observation type")
        new_hdu.header['PRESCAN'] = (str(hdu[0].header['PRESCAN']), "Prescan pixels at start/end of line")
        new_hdu.header['ROSPEED'] = (str(hdu[0].header['ROSPEED']),"CCD readout speed (Hz)")
        new_hdu.header['JD_MEAN'] = (str(jd_mean),"Mean JD of input files")
        new_hdu.header['MN_PDEW']= (str(np.mean(PRE_DEW)),"Mean of input files")
        new_hdu.header['MN_PVAC']= (str(np.mean(PRE_VAC)),"Mean of input files")
        new_hdu.header['MN_TAIR']= (str(np.mean(TEM_AIR)),"Mean of input files")
        new_hdu.header['MN_TBCAM']= (str(np.mean(TEM_BCAM)),"Mean of input files")
        new_hdu.header['MN_TCOLL']= (str(np.mean(TEM_COLL)),"Mean of input files")
        new_hdu.header['MN_TECH']= (str(np.mean(TEM_ECH)),"Mean of input files")
        new_hdu.header['MN_TIOD']= (str(np.mean(TEM_IOD)),"Mean of input files")
        new_hdu.header['MN_TOB']= (str(np.mean(TEM_OB)),"Mean of input files")
        new_hdu.header['MN_TRCAM']= (str(np.mean(TEM_RCAM)),"Mean of input files")
        new_hdu.header['MN_TRMIR']= (str(np.mean(TEM_RMIR)),"Mean of input files")
        new_hdu.header['MN_TVAC']= (str(np.mean(TEM_VAC)),"Mean of input files")
        new_hdu.header['GAIN_p'] = (str(gain),"Gain that has been applied")
        DATE_EXT=str(datetime.now(tz=pytz.UTC).strftime("%Y-%m-%d"))
        UTC_EXT = str(datetime.now(tz=pytz.UTC).strftime("%H:%M:%S.%f"))
        new_hdu.header['DATE-EXT'] = (DATE_EXT,'Date file created')
        new_hdu.header['UTC-EXT'] = (UTC_EXT,'Time file created')
        new_hdu.header['N_FILE'] = (str(n),"Number of files combined")
        new_hdu.header['HISTORY'] = ("Files used for Master: "+str(Flat_files_short))
        
        new_hdu.writeto(str(out_location)+"Master_Flat_"+str(arm)+str(night)+".fits",overwrite=True)
    
    else:
        print("\n      +++ Reading Master Flat frame "+master_file[0]+"\n")
        flat_hdu=fits.open(master_file[0])
        master_flat=flat_hdu[0].data
        flat_hdu.close
    
    if Plot == "True":  # pragma: no cover
        title = "Master Flat"
        plt.title(title)
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        bot, top = np.percentile(master_flat, (1, 99))
        plt.imshow(master_flat, vmin=bot, vmax=top, origin="lower")
        plt.show()
        
    return master_flat

"""
===================================================================================================================
"""

def get_them_B(sc,exap,ncoef,oversample,maxords=-1,startfrom=0,nsigmas=10.,mode=1,endat=-1,nc2=2,Plot='false'):
    exap = int(exap)
    def fitfunc(p,x):
        ret = p[0] + p[1] * np.exp(-.5*((x-p[2])/p[3])**2)
        return ret
    errfunc = lambda p,y,x: np.ravel( (fitfunc(p,x)-y) )

    def gauss2(params,x):
        amp1 = params[0]
        amp2 = params[1]
        med1 = params[2]
        med2 = params[3]
        sig1 = params[4]
        sig2 = params[5]
        g1 = amp1 * np.exp(-0.5*((x-med1)/sig1)**2)
        g2 = amp2 * np.exp(-0.5*((x-med2)/sig2)**2)
        return g1 + g2

    def res_gauss2(params,g,x):
        return g-gauss2(params,x)
    
    sc_or = sc.copy()
    if endat == -1:
        sc = sc[startfrom:,:]
    else:
        sc = sc[startfrom:endat,:]
    
    medc = int(.5*sc.shape[1])
    d = np.median(sc[:,medc-exap:medc+exap+1],axis=1)

    ejx = np.arange(len(d))
#    ccf=[]
    refw = 1*exap
    sigc = 0.5*exap
#    i = 0
#    while i < sc.shape[0]:
#        if i-refw < 0:
#            refw2=int(refw/2*oversample)
#            if oversample == 5:
#                refw2=int(refw*3/oversample)
#            x = ejx[:i+refw2+1]
#            y = d[:i+refw2+1]
#        elif i + refw +1 > sc.shape[0]:
#            x = ejx[i-refw:]
#            y = d[i-refw:]
#        else:
#            x = ejx[i-refw:i+refw+1]
#            y = d[i-refw:i+refw+1]
#
#        g = gaussian(x, 1, np.mean(x), 5.*oversample)#np.exp(-0.5*((x-i)/sigc)**2)
#        ccf.append(np.add.reduce(y*g))
##        def tophat(x, base_level, hat_level, hat_mid, hat_width):
##            return np.where((hat_mid-hat_width/2. < x) & (x < hat_mid+hat_width/2.), hat_level, base_level)
##        g = tophat(x,0,100,20*oversample,16*oversample)
##        ccf.append(correlate(y,g,mode='full',method='fft')/len(y))
#        i+=1
#    i = 1
#
#    maxs = []
#    while i < len(ccf)-2:
#        if ccf[i]>ccf[i-1] and ccf[i]>ccf[i+1]:
#            if (i < 2000*oversample):
#                maxs.append(i)
#            elif (i > 2000*oversample and ccf[i] > 3000*oversample):
#                maxs.append(i)
#        i+=1
#
#    maxs = np.array(maxs)
#    ccf = np.array(ccf)
#
#    pos = np.arange(len(ccf))[maxs] #+ refw
#    dt = d.copy()
#    sp = pos[1] - pos[0] - 2*exap
#    vx,vy = [],[]
#    if (Plot == "True"):
#        plt.plot(d)
#        plt.plot(ccf)
#        plt.plot(pos,d[pos],'ro')
#        plt.title("Orders=")
#        plt.show()
#    exap2 = exap# + int(exap*2./3.)
#    tbase = np.array([])
#    for i in range(len(pos)):
#        exs,vec = [],[]
#        if i == 0:
#            if pos[i] - exap2 - sp < 0:
#                exs = np.arange(0, pos[i] - exap2+1,1)
#                vec = d[: pos[i] - exap2+1]
#            else:
#                exs = np.arange(pos[i] - exap2 - sp, pos[i] - exap2+1,1)
#                vec = d[pos[i] - exap2 - sp: pos[i] - exap2+1]
#        else:
#            print("DLH",i,pos[i-1] + exap2,pos[i] - exap2+1)
#            if pos[i-1] + exap2 < pos[i] - exap2+1:
#                #print pos[i-1] + exap2 , pos[i] - exap2+1
#                exs = np.arange(pos[i-1] + exap2 , pos[i] - exap2+1,1)
#                vec = d[pos[i-1] + exap2 : pos[i] - exap2 + 1]
#                print("DLH exs > 0",pos[i-1] + exap2 , pos[i] - exap2+1,1)
#
#        if len(exs)>0:
#            tbase = np.hstack((tbase,exs))
#            vx.append(np.median(exs))
#            vy.append(np.median(vec))
#
#    tbase = tbase.astype('int')
#    vx,vy = np.array(vx),np.array(vy)
#    tck = interpolate.splrep(vx,vy,k=1)
#    dtemp = d[tbase] - interpolate.splev(tbase,tck)
#    ddev = np.sqrt(np.var(dtemp[5:-5]))
#    dt = d-interpolate.splev(np.arange(len(d)),tck)
#    I = np.where(dt[pos]>dtemp.mean() + nsigmas*ddev)[0]
#    pos = pos[I]
#    I2 = np.where(pos>22*oversample)
#    pos=pos[I2]
#
#    print("DLH DEBUG")

    x=np.arange(0,len(d),1)
    t_hat = np.array(d[1370*oversample:1400*oversample])
    y=np.array(np.ones(len(x)-30*oversample))
    test=np.append(t_hat,y)
    ccf = correlate(d,test,mode='full',method='fft')/(len(x)*oversample)#-29*oversample
    tmp=ccf[len(x)-1*oversample:]
    tmp=tmp[20*oversample:]
    x=np.arange(0,len(tmp),1)+35*oversample
    peaks_2,_=find_peaks(tmp, distance=25*oversample)
    peaks_2=np.asarray(peaks_2)+35*oversample

    peaks_x=peaks_2
    peaks_y=tmp[peaks_2-35*oversample]
    for i in range(len(peaks_x)):
        if (peaks_x[i] > 1500*oversample and peaks_y[i] < 20):
            peaks_x[i] = 0
            peaks_y[i] = 0

    I=np.where(peaks_x == 0)
    peaks_x = np.delete(peaks_x,I)
    peaks_y = np.delete(peaks_y,I)
    pos= []
    
    #Pick the peaks that make sense
    for i in range(len(peaks_x)):
        if peaks_x[i] <= 1000*oversample:
            pos.append(peaks_x[i])
        if (1000*oversample < peaks_x[i] <=2000*oversample):
            if d[peaks_x[i]] > 80:
                pos.append(peaks_x[i])
        if (2000*oversample < peaks_x[i] < 4150*oversample) and d[peaks_x[i]] > 200:
            pos.append(peaks_x[i])
    pos=(np.array(pos))

    if Plot == "True":
        plt.plot(d)
        plt.plot(ccf)
        plt.plot(pos,d[pos],'bo')
        plt.title(str(len(pos)))
        plt.show()
    
    if len(pos) != 84:
        print("Found ",len(pos)," orders rather than 84. Exiting from get_them_B")
        exit()

    if maxords >0:
        pos = pos[::-1]
        pos = pos[:maxords]
        pos = pos[::-1]

    I = np.where(pos < exap)[0]
    pos = np.delete(pos,I)
    I = np.where(pos > sc.shape[0]-exap)[0]
    pos = np.delete(pos,I)


    ref = []
    if mode == 1 or mode == 2:
        if mode == 1:
                #exap2 = exap + .5*exap
                #dev = exap2/3.
                exap2 = 1. + .5*exap
                dev = exap2/4.
        else:
                exap2 = exap + .2*exap
                dev = exap2/4.
        exap2 = int(exap2)
        for i in range(len(pos)):
            if pos[i]-exap2 < 0:
                    x = ejx[:int(pos[i]+exap2+1)]
                    y = d[:int(pos[i]+exap2+1)]
            elif pos[i]+exap2+1 > len(d):
                    x = ejx[int(pos[i]-exap2):]
                    y = d[int(pos[i]-exap2):]
            else:
                    x = ejx[int(pos[i]-exap2):int(pos[i]+exap2+1)]
                    y = d[int(pos[i]-exap2):int(pos[i]+exap2+1)]
            tx1 = np.arange(x[0]-dev,x[0],1)
            tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
            ty1 = np.zeros(len(tx1))
            ty2 = np.zeros(len(tx2))
            x = np.hstack((tx1,x,tx2))
            y = np.hstack((ty1,y,ty2))
            y -= y.min()

            if mode == 1:
                if len(x) < 4:
                    tref.append(ref[j])
                else:
                    p, success =  scipy.optimize.leastsq(errfunc, [y.min(),y.max()-y.min(),x.mean(),dev], args=(y,x))
                ref.append(p[2])
            else:
                midi = int(0.5*len(x))
                if len(x) < 7:
                    tref.append(ref[j])
                else:
                    guess = [np.max(y[:midi]),np.max(y[midi:]),x[0]+np.argmax(y[:midi]),x[0]+midi+np.argmax(y[midi:]),dev,dev]
                    p, success =  scipy.optimize.leastsq(res_gauss2, guess, args=(y,x))

                ref.append(0.5*(p[2]+p[3]))

    ref = np.array(ref)

    mat = np.zeros((len(ref),sc.shape[1]))
    mat[:,medc] = ref
    i = medc -1
    
    while i >=0:
        #print (i)
        d = sc[:,i]
        j = 0
        pos = np.around(ref).astype('int')
        tref = []
        if mode == 1 or mode == 2:
                if mode == 1:
                    #exap2 = exap + .2*exap
                    #dev = exap2/3.
                    exap2 = 1 + .5*exap
                    dev = exap2/4.
                else:
                    exap2 = exap + .2*exap
                    dev = exap2/4.
                exap2 = int(exap2)
                while j < len(pos):
                    if pos[j]-exap2 < 0:
                            x = ejx[:int(pos[j]+exap2+1)]
                            y = d[:int(pos[j]+exap2+1)]
                    elif pos[j]+exap2+1 > len(d):
                            x = ejx[int(pos[j]-exap2):]
                            y = d[int(pos[j]-exap2):]
                    else:
                            x = ejx[int(pos[j]-exap2):int(pos[j]+exap2+1)]
                            y = d[int(pos[j]-exap2):int(pos[j]+exap2+1)]

                    if mode==1:
                        if len(x) < 4:
                                tref.append(ref[j])
                        else:
                                tx1 = np.arange(x[0]-dev,x[0],1)
                                tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
                                ty1 = np.zeros(len(tx1)) + y.min()
                                ty2 = np.zeros(len(tx2)) + y.min()
                                x = np.hstack((tx1,x,tx2))
                                y = np.hstack((ty1,y,ty2))
                                p, success =  scipy.optimize.leastsq(errfunc, [y.min(),y.max()-y.min(),x.mean(),dev], args=(y,x))

                                tref.append(p[2])
                    else:
                            if len(x) < 7:
                                tref.append(ref[j])
                            else:
                                tx1 = np.arange(x[0]-dev,x[0],1)
                                tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
                                ty1 = np.zeros(len(tx1)) + y.min()
                                ty2 = np.zeros(len(tx2)) + y.min()
                                x = np.hstack((tx1,x,tx2))
                                y = np.hstack((ty1,y,ty2))
                                y -= y.min()
                                midi = int(0.5*len(x))
                                guess = [np.max(y[:midi]),np.max(y[midi:]),x[0]+np.argmax(y[:midi]),x[0]+midi+np.argmax(y[midi:]),dev,dev]
                                p, success =  scipy.optimize.leastsq(res_gauss2, guess, args=(y,x))
                                tref.append(0.5*(p[2]+p[3]))
                    j+=1

        oref = ref.copy()
        tref = np.array(tref)
        dif = tref-ref
        coef = np.polyfit(ref,dif,nc2)

        coef2 = np.polyfit(np.arange(len(dif)),dif,1)

        residuals = dif - np.polyval(coef,ref)
        rms = np.sqrt(np.var(residuals))
        I = np.where(np.absolute(residuals)>3*rms)[0]
        cond = True
        if len(I)==0:
            cond = False
        while cond:
            im  = np.argmax(np.absolute(residuals))
            dif = np.delete(dif,im)
            ref = np.delete(ref,im)
            coef = np.polyfit(ref,dif,nc2)
            residuals = dif - np.polyval(coef,ref)
            rms = np.sqrt(np.var(residuals))
            I = np.where(np.absolute(residuals)>3*rms)[0]
            if len(I)==0:
                cond = False
        cdif = np.polyval(coef,oref)
        ref = oref + cdif

        mat[:,i] = ref
        i-=4
        
    i = medc+1
    ref = mat[:,medc]
    while i < sc.shape[1]:
        #print i
        d = sc[:,i]
        j = 0
        pos = np.around(ref).astype('int')
        tref = []
        if mode == 1 or mode == 2:
                if mode == 1:
                    #exap2 = exap + .5*exap
                    #dev = exap2/3.
                    exap2 = 1. + .5*exap
                    dev = exap2/4.
                else:
                    exap2 = exap + .2*exap
                    dev = exap2/4.
                while j < len(pos):
                    if pos[j]-exap2 < 0:
                            x = ejx[:int(pos[j]+exap2+1)]
                            y = d[:int(pos[j]+exap2+1)]
                    elif pos[j]+exap2+1 > len(d):
                            x = ejx[int(pos[j]-exap2):]
                            y = d[int(pos[j]-exap2):]
                    else:
                            x = ejx[int(pos[j]-exap2):int(pos[j]+exap2+1)]
                            y = d[int(pos[j]-exap2):int(pos[j]+exap2+1)]

                    if mode == 1:
                            if len(x) < 4:
                                tref.append(ref[j])
                            else:
                                tx1 = np.arange(x[0]-dev,x[0],1)
                                tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
                                ty1 = np.zeros(len(tx1)) + y.min()
                                ty2 = np.zeros(len(tx2)) + y.min()
                                x = np.hstack((tx1,x,tx2))
                                y = np.hstack((ty1,y,ty2))
                                p, success =  scipy.optimize.leastsq(errfunc, [y.min(),y.max()-y.min(),x.mean(),dev], args=(y,x))
                                tref.append(p[2])
                    else:
                            if len(x) < 7:
                                tref.append(ref[j])
                            else:
                                tx1 = np.arange(x[0]-dev,x[0],1)
                                tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
                                ty1 = np.zeros(len(tx1)) + y.min()
                                ty2 = np.zeros(len(tx2)) + y.min()
                                x = np.hstack((tx1,x,tx2))
                                y = np.hstack((ty1,y,ty2))
                                y -= y.min()
                                midi = int(0.5*len(x))
                                guess = [np.max(y[:midi]),np.max(y[midi:]),x[0]+np.argmax(y[:midi]),x[0]+midi+np.argmax(y[midi:]),dev,dev]
                                p, success =  scipy.optimize.leastsq(res_gauss2, guess, args=(y,x))
                                tref.append(0.5*(p[2]+p[3]))
                
                    j+=1

        oref = ref.copy()
        tref = np.array(tref)
        dif = tref-ref

        coef = np.polyfit(ref,dif,nc2)

        residuals = dif - np.polyval(coef,ref)
        rms = np.sqrt(np.var(residuals))
        I = np.where(np.absolute(residuals)>3*rms)[0]
        cond = True
        if len(I)==0:
            cond = False
        while cond:
            im  = np.argmax(np.absolute(residuals))
            dif = np.delete(dif,im)
            ref = np.delete(ref,im)
            coef = np.polyfit(ref,dif,nc2)
            residuals = dif - np.polyval(coef,ref)
            rms = np.sqrt(np.var(residuals))
            I = np.where(np.absolute(residuals)>3*rms)[0]
            if len(I)==0:
                cond = False
        
        cdif = np.polyval(coef,oref)
        ref = oref + cdif
        #plot(ref,np.polyval(coef,ref))
        #show()
        mat[:,i] = ref
        i+=4

    if (Plot == "True"):
        plt.imshow(sc_or,vmax=20,origin='lower')
    for i in range(mat.shape[0]):
        y = mat[i]
        x = np.arange(len(y))
        I = np.where(y!=0)[0]
        x,y = x[I],y[I]+startfrom
        coef = np.polyfit(x,y,ncoef)
        if (Plot == "True"):
            plt.plot(x,y,'bo')
        residuals = y - np.polyval(coef,x)
        rms = np.sqrt(np.var(residuals))
        I = np.where(np.absolute(residuals)>3*rms)[0]
        cond = True
        if len(I)==0:
            cond = False
        while cond:
            im = np.argmax(np.absolute(residuals))
            x = np.delete(x,im)
            y = np.delete(y,im)
            coef = np.polyfit(x,y,ncoef)
            residuals = y - np.polyval(coef,x)
            rms = np.sqrt(np.var(residuals))
            I = np.where(np.absolute(residuals)>3*rms)[0]
            if len(I)==0:
                cond = False
        #coef[-1] += startfrom
        if i == 0:
            acoefs = coef
        else:
            acoefs = np.vstack((acoefs,coef))
        if (Plot == "True"):
            plt.plot(np.polyval(coef,np.arange(len(mat[i]))),'k')


    if (Plot == "True"):
        plt.show()
    return acoefs, len(acoefs)


"""
===================================================================================================================
"""

def order_width_B(n_ord,master_flat_zoom,oversample,c_all):

    #Define the aperture for each order as the aperture expands and we go up the chip
    ext_aperture =np.zeros((n_ord, 2))
    x=int(master_flat_zoom.shape[1]/2.)
    for i in range(n_ord):
        y_mid =int(c_all[i][0]*x**4 + c_all[i][1]*x**3 + c_all[i][2]*x**2 + c_all[i][3]*x + c_all[i][4])
        xxx = []
        yxx = []
        for q in range(-20*oversample,20*oversample):
            xxx.append(q)
            yxx.append(master_flat_zoom[y_mid+q,x])
        xxx=np.asarray(xxx)
        yxx=np.asarray(yxx)
        #plt.plot(xxx,yxx,'-ro')
        #yxx = savgol_filter(yxx, 11, 9)
        #plt.plot(xxx,yxx,'-bo')
        #Working from the centre of the order, find the minimum that defines the order edge. Plot a verticle line to illustrate it.
        min_lower_y=1e8
        for q in range(-1,-20*oversample,-1):
            if yxx[20*oversample+q] < min_lower_y:
                min_lower=xxx[20*oversample+q]
                min_lower_y = yxx[20*oversample+q]
        #plt.axvline(x=min_lower)
        #Working from the centre of the order, find the minimum that defines the order edge. Plot a verticle line to illustrate it.
        min_upper_y=1e8
        for q in range(0,19*oversample):
            if yxx[20*oversample+q] < min_upper_y:
                min_upper=xxx[20*oversample+q]
                min_upper_y = yxx[20*oversample+q]
        #plt.axvline(x=min_upper)
        #m=(np.diff(yxx)/np.diff(xxx))
        #plt.plot(xxx[1:],m)
        #plt.show()
        ext_aperture[i][0] =np.abs(min_upper)
        ext_aperture[i][1] =np.abs(min_lower)

        #Using the width information, change the centre of the order trance so we are symetrical about the centre for the optimal extraction.

        c_all[i][4]=c_all[i][4]+0.5*(ext_aperture[i][0]-ext_aperture[i][1])
        ext_aperture[i][0],ext_aperture[i][1]=int((ext_aperture[i][0]+ext_aperture[i][1])*0.5) ,int((ext_aperture[i][0]+ext_aperture[i][1])*0.5)

    return ext_aperture,c_all

"""
===================================================================================================================
"""

def get_them_R(sc,exap,ncoef,oversample,maxords=-1,startfrom=0,nsigmas=10.,mode=1,endat=-1,nc2=2,Plot='false'):
    exap = int(exap)
    def fitfunc(p,x):
        ret = p[0] + p[1] * np.exp(-.5*((x-p[2])/p[3])**2)
        return ret
    errfunc = lambda p,y,x: np.ravel( (fitfunc(p,x)-y) )

    def gauss2(params,x):
        amp1 = params[0]
        amp2 = params[1]
        med1 = params[2]
        med2 = params[3]
        sig1 = params[4]
        sig2 = params[5]
        g1 = amp1 * np.exp(-0.5*((x-med1)/sig1)**2)
        g2 = amp2 * np.exp(-0.5*((x-med2)/sig2)**2)
        return g1 + g2

    def res_gauss2(params,g,x):
        return g-gauss2(params,x)
    
    sc_or = sc.copy()
    if endat == -1:
        sc = sc[startfrom:,:]
    else:
        sc = sc[startfrom:endat,:]
    
    medc = int(.5*sc.shape[1])
    d = np.median(sc[:,medc-exap:medc+exap+1],axis=1)

    ejx = np.arange(len(d))
#    ccf=[]
    refw = 1*exap
    sigc = 0.5*exap

    x=np.arange(0,len(d),1)
    t_hat = np.array(d[1370*oversample:1400*oversample])
    y=np.array(np.ones(len(x)-30*oversample))
    test=np.append(t_hat,y)
    ccf = correlate(d,test,mode='full',method='fft')/(len(x)*oversample)-29*oversample
    tmp=ccf[len(x)-1*oversample:]
    tmp=tmp[20*oversample:]
    x=np.arange(0,len(tmp),1)+35*oversample
    peaks_2,_=find_peaks(tmp, distance=25*oversample)
    peaks_2=np.asarray(peaks_2)+35*oversample

    peaks_x=peaks_2
    peaks_y=tmp[peaks_2-35*oversample]
    for i in range(len(peaks_x)):
        if (peaks_x[i] > 1500*oversample and peaks_y[i] < 20):
            peaks_x[i] = 0
            peaks_y[i] = 0

    I=np.where(peaks_x == 0)
    peaks_x = np.delete(peaks_x,I)
    peaks_y = np.delete(peaks_y,I)
    pos= []
    
    #Pick the peaks that make sense
    for i in range(len(peaks_x)):
#        if peaks_x[i] <= 1000*oversample:
#            pos.append(peaks_x[i])
        if (000*oversample < peaks_x[i] <=2000*oversample):
            if d[peaks_x[i]] > 200:
                pos.append(peaks_x[i])
        if (2000*oversample < peaks_x[i] < 4150*oversample) and d[peaks_x[i]] > 500:
            pos.append(peaks_x[i])
    pos=(np.array(pos))

    if Plot == "True":
        plt.plot(d)
        plt.plot(ccf)
        plt.plot(pos,d[pos],'bo')
        plt.title(str(len(pos)))
        plt.show()
    
    if len(pos) != 64:
        print("Found ",len(pos)," orders rather than 64. Exiting from get_them_R")
        exit()

    if maxords >0:
        pos = pos[::-1]
        pos = pos[:maxords]
        pos = pos[::-1]

    I = np.where(pos < exap)[0]
    pos = np.delete(pos,I)
    I = np.where(pos > sc.shape[0]-exap)[0]
    pos = np.delete(pos,I)


    ref = []
    if mode == 1 or mode == 2:
        if mode == 1:
                #exap2 = exap + .5*exap
                #dev = exap2/3.
                exap2 = 1. + .5*exap
                dev = exap2/4.
        else:
                exap2 = exap + .2*exap
                dev = exap2/4.
        exap2 = int(exap2)
        for i in range(len(pos)):
            if pos[i]-exap2 < 0:
                    x = ejx[:int(pos[i]+exap2+1)]
                    y = d[:int(pos[i]+exap2+1)]
            elif pos[i]+exap2+1 > len(d):
                    x = ejx[int(pos[i]-exap2):]
                    y = d[int(pos[i]-exap2):]
            else:
                    x = ejx[int(pos[i]-exap2):int(pos[i]+exap2+1)]
                    y = d[int(pos[i]-exap2):int(pos[i]+exap2+1)]
            tx1 = np.arange(x[0]-dev,x[0],1)
            tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
            ty1 = np.zeros(len(tx1))
            ty2 = np.zeros(len(tx2))
            x = np.hstack((tx1,x,tx2))
            y = np.hstack((ty1,y,ty2))
            y -= y.min()

            if mode == 1:
                if len(x) < 4:
                    tref.append(ref[j])
                else:
                    p, success =  scipy.optimize.leastsq(errfunc, [y.min(),y.max()-y.min(),x.mean(),dev], args=(y,x))
                ref.append(p[2])
            else:
                midi = int(0.5*len(x))
                if len(x) < 7:
                    tref.append(ref[j])
                else:
                    guess = [np.max(y[:midi]),np.max(y[midi:]),x[0]+np.argmax(y[:midi]),x[0]+midi+np.argmax(y[midi:]),dev,dev]
                    p, success =  scipy.optimize.leastsq(res_gauss2, guess, args=(y,x))

                ref.append(0.5*(p[2]+p[3]))

    ref = np.array(ref)

    mat = np.zeros((len(ref),sc.shape[1]))
    mat[:,medc] = ref
    i = medc -1
    
    while i >=0:
        #print (i)
        d = sc[:,i]
        j = 0
        pos = np.around(ref).astype('int')
        tref = []
        if mode == 1 or mode == 2:
                if mode == 1:
                    #exap2 = exap + .2*exap
                    #dev = exap2/3.
                    exap2 = 1 + .5*exap
                    dev = exap2/4.
                else:
                    exap2 = exap + .2*exap
                    dev = exap2/4.
                exap2 = int(exap2)
                while j < len(pos):
                    if pos[j]-exap2 < 0:
                            x = ejx[:int(pos[j]+exap2+1)]
                            y = d[:int(pos[j]+exap2+1)]
                    elif pos[j]+exap2+1 > len(d):
                            x = ejx[int(pos[j]-exap2):]
                            y = d[int(pos[j]-exap2):]
                    else:
                            x = ejx[int(pos[j]-exap2):int(pos[j]+exap2+1)]
                            y = d[int(pos[j]-exap2):int(pos[j]+exap2+1)]

                    if mode==1:
                        if len(x) < 4:
                                tref.append(ref[j])
                        else:
                                tx1 = np.arange(x[0]-dev,x[0],1)
                                tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
                                ty1 = np.zeros(len(tx1)) + y.min()
                                ty2 = np.zeros(len(tx2)) + y.min()
                                x = np.hstack((tx1,x,tx2))
                                y = np.hstack((ty1,y,ty2))
                                p, success =  scipy.optimize.leastsq(errfunc, [y.min(),y.max()-y.min(),x.mean(),dev], args=(y,x))

                                tref.append(p[2])
                    else:
                            if len(x) < 7:
                                tref.append(ref[j])
                            else:
                                tx1 = np.arange(x[0]-dev,x[0],1)
                                tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
                                ty1 = np.zeros(len(tx1)) + y.min()
                                ty2 = np.zeros(len(tx2)) + y.min()
                                x = np.hstack((tx1,x,tx2))
                                y = np.hstack((ty1,y,ty2))
                                y -= y.min()
                                midi = int(0.5*len(x))
                                guess = [np.max(y[:midi]),np.max(y[midi:]),x[0]+np.argmax(y[:midi]),x[0]+midi+np.argmax(y[midi:]),dev,dev]
                                p, success =  scipy.optimize.leastsq(res_gauss2, guess, args=(y,x))
                                tref.append(0.5*(p[2]+p[3]))
                    j+=1

        oref = ref.copy()
        tref = np.array(tref)
        dif = tref-ref
        coef = np.polyfit(ref,dif,nc2)

        coef2 = np.polyfit(np.arange(len(dif)),dif,1)

        residuals = dif - np.polyval(coef,ref)
        rms = np.sqrt(np.var(residuals))
        I = np.where(np.absolute(residuals)>3*rms)[0]
        cond = True
        if len(I)==0:
            cond = False
        while cond:
            im  = np.argmax(np.absolute(residuals))
            dif = np.delete(dif,im)
            ref = np.delete(ref,im)
            coef = np.polyfit(ref,dif,nc2)
            residuals = dif - np.polyval(coef,ref)
            rms = np.sqrt(np.var(residuals))
            I = np.where(np.absolute(residuals)>3*rms)[0]
            if len(I)==0:
                cond = False
        cdif = np.polyval(coef,oref)
        ref = oref + cdif

        mat[:,i] = ref
        i-=4
        
    i = medc+1
    ref = mat[:,medc]
    while i < sc.shape[1]:
        #print i
        d = sc[:,i]
        j = 0
        pos = np.around(ref).astype('int')
        tref = []
        if mode == 1 or mode == 2:
                if mode == 1:
                    #exap2 = exap + .5*exap
                    #dev = exap2/3.
                    exap2 = 1. + .5*exap
                    dev = exap2/4.
                else:
                    exap2 = exap + .2*exap
                    dev = exap2/4.
                while j < len(pos):
                    if pos[j]-exap2 < 0:
                            x = ejx[:int(pos[j]+exap2+1)]
                            y = d[:int(pos[j]+exap2+1)]
                    elif pos[j]+exap2+1 > len(d):
                            x = ejx[int(pos[j]-exap2):]
                            y = d[int(pos[j]-exap2):]
                    else:
                            x = ejx[int(pos[j]-exap2):int(pos[j]+exap2+1)]
                            y = d[int(pos[j]-exap2):int(pos[j]+exap2+1)]

                    if mode == 1:
                            if len(x) < 4:
                                tref.append(ref[j])
                            else:
                                tx1 = np.arange(x[0]-dev,x[0],1)
                                tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
                                ty1 = np.zeros(len(tx1)) + y.min()
                                ty2 = np.zeros(len(tx2)) + y.min()
                                x = np.hstack((tx1,x,tx2))
                                y = np.hstack((ty1,y,ty2))
                                p, success =  scipy.optimize.leastsq(errfunc, [y.min(),y.max()-y.min(),x.mean(),dev], args=(y,x))
                                tref.append(p[2])
                    else:
                            if len(x) < 7:
                                tref.append(ref[j])
                            else:
                                tx1 = np.arange(x[0]-dev,x[0],1)
                                tx2 = np.arange(x[-1]+1,x[-1]+dev+1,1)
                                ty1 = np.zeros(len(tx1)) + y.min()
                                ty2 = np.zeros(len(tx2)) + y.min()
                                x = np.hstack((tx1,x,tx2))
                                y = np.hstack((ty1,y,ty2))
                                y -= y.min()
                                midi = int(0.5*len(x))
                                guess = [np.max(y[:midi]),np.max(y[midi:]),x[0]+np.argmax(y[:midi]),x[0]+midi+np.argmax(y[midi:]),dev,dev]
                                p, success =  scipy.optimize.leastsq(res_gauss2, guess, args=(y,x))
                                tref.append(0.5*(p[2]+p[3]))
                
                    j+=1

        oref = ref.copy()
        tref = np.array(tref)
        dif = tref-ref

        coef = np.polyfit(ref,dif,nc2)

        residuals = dif - np.polyval(coef,ref)
        rms = np.sqrt(np.var(residuals))
        I = np.where(np.absolute(residuals)>3*rms)[0]
        cond = True
        if len(I)==0:
            cond = False
        while cond:
            im  = np.argmax(np.absolute(residuals))
            dif = np.delete(dif,im)
            ref = np.delete(ref,im)
            coef = np.polyfit(ref,dif,nc2)
            residuals = dif - np.polyval(coef,ref)
            rms = np.sqrt(np.var(residuals))
            I = np.where(np.absolute(residuals)>3*rms)[0]
            if len(I)==0:
                cond = False
        
        cdif = np.polyval(coef,oref)
        ref = oref + cdif
        #plot(ref,np.polyval(coef,ref))
        #show()
        mat[:,i] = ref
        i+=4
    Plot="True"
    if (Plot == "True"):
        plt.imshow(sc_or,vmax=2000,origin='lower')
    for i in range(mat.shape[0]):
        y = mat[i]
        x = np.arange(len(y))
        I = np.where(y!=0)[0]
        x,y = x[I],y[I]+startfrom
        coef = np.polyfit(x,y,ncoef)
        if (Plot == "True"):
            plt.plot(x,y,'bo')
        residuals = y - np.polyval(coef,x)
        rms = np.sqrt(np.var(residuals))
        I = np.where(np.absolute(residuals)>3*rms)[0]
        cond = True
        if len(I)==0:
            cond = False
        while cond:
            im = np.argmax(np.absolute(residuals))
            x = np.delete(x,im)
            y = np.delete(y,im)
            coef = np.polyfit(x,y,ncoef)
            residuals = y - np.polyval(coef,x)
            rms = np.sqrt(np.var(residuals))
            I = np.where(np.absolute(residuals)>3*rms)[0]
            if len(I)==0:
                cond = False
        #coef[-1] += startfrom
        if i == 0:
            acoefs = coef
        else:
            acoefs = np.vstack((acoefs,coef))
        if (Plot == "True"):
            plt.plot(np.polyval(coef,np.arange(len(mat[i]))),'k')


    if (Plot == "True"):
        plt.show()
    return acoefs, len(acoefs)


"""
===================================================================================================================
"""

"""
===================================================================================================================
"""

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    #return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))
    #return (1./(wid*np.sqrt(2*np.pi))) * np.exp(-(x-cen)**2 / (2*wid**2))
    return amp*np.exp(-(x-cen)**2/(2*wid**2))

"""
===================================================================================================================
"""

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


"""
===================================================================================================================
"""
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

"""
===================================================================================================================
"""
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

"""
===================================================================================================================
"""
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

"""
===================================================================================================================
"""
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
    
"""
===================================================================================================================
"""

def correct_for_curvature(img_order, tilt, shear, xwd):
#    oversample=3
#    zoomed=scipy.ndimage.zoom(img_order, oversample, order=0)
#    xwd=xwd*oversample
#    shear=0.
#    tilt2=np.zeros(len(tilt)*oversample)
#    count =0
#    for i in range(0,len(tilt2)-oversample,oversample):
#        tilt2[i]=tilt[count]
#        tilt2[i+1]=tilt[count]
#        tilt2[i+2]=tilt[count]
#        count += 1
#    tilt=tilt2/(oversample*oversample)
#    img_order=zoomed
    mask = ~np.ma.getmaskarray(img_order)
    xt = np.arange(img_order.shape[1])
    
    for y, yt in zip(range(int(xwd[0]) + int(xwd[1])), range(-int(xwd[0]), int(xwd[1]))):
        xi = xt + yt * tilt + yt ** 2 * shear

        img_order[y] = np.interp(
            xi, xt[mask[y]], img_order[y][mask[y]], left=0, right=0
        )

    xt = np.arange(img_order.shape[0])
    for x in range(img_order.shape[1]):
        img_order[:, x] = np.interp(
            xt, xt[mask[:, x]], img_order[:, x][mask[:, x]], left=0, right=0
        )

    
#    img_order = scipy.ndimage.zoom(img_order, 1/oversample, order=0)

    return img_order
    
"""
===================================================================================================================
"""
def polyfit2d(
    x, y, z, degree=1, max_degree=None, scale=True, plot=False, plot_title=None):
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
    C, *_ = scipy.linalg.lstsq(A, z)

    # Reorder coefficients into numpy compatible 2d array
    for k, (i, j) in enumerate(idx):
        coeff[i, j] = C[k]

    # # Backup copy of coeff
    if scale:
        coeff = polyscale2d(coeff, *norm, copy=False)
        coeff = polyshift2d(coeff, *offset, copy=False)

#    if plot:  # pragma: no cover
#        if scale:
#            x, y = _unscale(x, y, norm, offset)
#        plot2d(x, y, z, coeff, title=plot_title)

    return coeff

"""
===================================================================================================================
"""
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


"""
===================================================================================================================
"""
def _get_coeff_idx(coeff):
    idx = np.indices(coeff.shape)
    idx = idx.T.swapaxes(0, 1).reshape((-1, 2))
    # degree = coeff.shape
    # idx = [[i, j] for i, j in product(range(degree[0]), range(degree[1]))]
    # idx = np.asarray(idx)
    return idx

"""
===================================================================================================================
"""
def polyvander2d(x, y, degree):
    # A = np.array([x ** i * y ** j for i, j in idx], dtype=float).T
    A = np.polynomial.polynomial.polyvander2d(x, y, degree)
    return A

"""
===================================================================================================================
"""
def polyscale2d(coeff, scale_x, scale_y, copy=True):
    if copy:
        coeff = np.copy(coeff)
    idx = _get_coeff_idx(coeff)
    for k, (i, j) in enumerate(idx):
        coeff[i, j] /= scale_x ** i * scale_y ** j
    return coeff

"""
===================================================================================================================
"""
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
            b = scipy.special.binom(i, k) * scipy.special.binom(j, m)
            sign = (-1) ** ((i - k) + (j - m))
            offset = offset_x ** (i - k) * offset_y ** (j - m)
            coeff[k, m] += sign * b * coeff2[i, j] * offset
    return coeff


"""
===================================================================================================================
"""
def obtain_P(order_data, trace_coeffs, Apertures, RON, Gain, NSigma, S, N, Marsh_alg,col_range,npools):
    npars_paralel = []
    
    for i in range(len(trace_coeffs)):
        if 'int' in str(type(col_range[i][0])) or 'float' in str(type(col_range[i][0])):
            min_col =  int(col_range[i][0])
        if 'int' in str(type(col_range[i][1])) or 'float' in str(type(col_range[i][1])):
            max_col =  int(col_range[i][1])
        npars_paralel.append([trace_coeffs[i,:],Apertures[i],RON,Gain,NSigma,S,N,Marsh_alg,int(min_col),int(max_col),order_data])
    p = Pool(npools)
    spec = np.array((p.map(PCoeff2, npars_paralel)))
    p.terminate()
    return np.sum(spec,axis=0)
"""
===================================================================================================================
"""
def PCoeff2(pars):
    trace_coeffs = pars[0]
    Aperture = pars[1]
    RON=pars[2]
    Gain = pars[3]
    NSigma = pars[4]
    S = pars[5]
    N = pars[6]
    Marsh_alg = pars[7]
    min_col = pars[8]
    max_col = pars[9]
    order_data = pars[10]
    Result      = Marsh.ObtainP((order_data.flatten()).astype('double'), \
                                   np.polyval(trace_coeffs,np.arange(order_data.shape[1])).astype('double'), \
                                   order_data.shape[0], order_data.shape[1], order_data.shape[1], Aperture, RON, Gain, \
                                   NSigma, S, N, Marsh_alg,min_col,max_col)
    FinalMatrix = np.asarray(Result)                      # After the function, we convert our list to a Numpy array.
    FinalMatrix.resize(order_data.shape[0],order_data.shape[1])   # And return the array in matrix-form.
    return FinalMatrix

"""
===================================================================================================================
"""
def optimal_extraction(order_data,Pin,coefs,ext_apertures,RON,GAIN,MARSH,COSMIC,col_range,npools):
    npars_paralel = []
    for i in range(len(coefs)):
        if 'int' in str(type(col_range[i][0])) or 'float' in str(type(col_range[i][0])):
            min_col =  int(col_range[i][0])
        if 'int' in str(type(col_range[i][1])) or 'float' in str(type(col_range[i][1])):
            max_col =  int(col_range[i][1])
        npars_paralel.append([coefs[i,:],ext_apertures[i],RON,GAIN,MARSH,COSMIC,int(min_col),int(max_col),order_data,Pin])
    p = Pool(npools)
    spec = np.array((p.map(getSpectrum2, npars_paralel)))
    p.terminate()
    return spec


"""
===================================================================================================================
"""
def getSpectrum2(pars):
    trace_coeffs = pars[0]
    Aperture = pars[1]
    RON=pars[2]
    Gain = pars[3]
    S = pars[4]
    NCosmic = pars[5]
    min_col = pars[6]
    max_col = pars[7]
    order_data = pars[8]
    Pin = pars[9]
    
    Result,size = Marsh.ObtainSpectrum( (order_data.flatten()).astype('double'), \
                                            scipy.polyval(trace_coeffs,np.arange(order_data.shape[1])).astype('double'), \
                                            Pin.flatten().astype('double'), order_data.shape[0],\
                                            order_data.shape[1],order_data.shape[1],Aperture,RON,\
                                            Gain,S,NCosmic,min_col,max_col)
    FinalMatrix = np.asarray(Result)                      # After the function, we convert our list to a Numpy array.
    FinalMatrix.resize(3,size)                            # And return the array in matrix-form.
    return FinalMatrix
    
"""
===================================================================================================================
"""
def get_scat(sc,lim,span, typ='median', allow_neg=False,option=0):
    scat = np.zeros(sc.shape)
    ejeX = np.arange(sc.shape[0])

    for y in range(sc.shape[1]):
        lims = np.around(lim[:,y]).astype('int')
        nejX,nejY = np.array([]),np.array([])
        #plot(sc[:,y])
        for j in range(len(lims)):
            if j == 0:
                #print lims[j] - span
                if lims[j] - span < 0:
                    ejx, ejy = [],[]
                elif lims[j] - 2 * span < 0:
                    ejx=ejeX[:lims[j]-span]
                    ejy=sc[:lims[j]-span,y]
                else:
                    ejx=ejeX[lims[j]- 2 * span:lims[j]-span+1]
                    ejy=sc[lims[j]- 2 * span:lims[j]-span+1,y]
                

            else:
                if lims[j-1] + span >= sc.shape[0] or lims[j-1] + span < 0:
                    ejx,ejy = [],[]

                elif lims[j] - span + 1 > sc.shape[0]:
                    ejx=ejeX[lims[j-1] + span: ]
                    ejy=sc[lims[j-1] + span:, y]
                elif lims[j-1] + span >= lims[j]- span + 1:
                    ejx,ejy = [],[]
                else:
                    ejx=ejeX[lims[j-1] + span:lims[j]- span + 1 ]
                    ejy=sc[lims[j-1] + span:lims[j]- span + 1, y]

                if option == 1 and len(ejx) == 0:
                    tpos = int(np.around(0.5*(lims[j-1] + lims[j])))
                    if tpos >= 0 and tpos < sc.shape[0]:
                        ejx = np.array([ejeX[tpos]])
                        ejy = np.array([sc[tpos,y]])


            if len(ejy)>0:
                if typ== 'median':
                    value = np.median(ejy)
                elif typ == 'min':
                    value = np.min(ejy)
            
                if np.isnan(value)==True or np.isnan(-value)==True:
                    value = 0.
                if value < 0 and (not allow_neg):
                    value = 0.

            if len(ejx) > 0:
                if len(nejX) == 0:
                    nejX = np.hstack((nejX,np.median(ejx)))
                    nejY = np.hstack((nejY,value))
                elif np.median(ejx) > nejX[-1]:
                    nejX = np.hstack((nejX,np.median(ejx)))
                    nejY = np.hstack((nejY,value))
                if j == 1 and len(nejY)>1:
                    nejY[0] = nejY[1]

        if lims[-1]+span >= sc.shape[0]:
            ejx,ejy = [],[]
        elif lims[-1]+2*span > sc.shape[0]:
            ejx,ejy = ejeX[lims[-1]+span:],sc[lims[-1]+span:,y]
        else:
            ejx=ejeX[lims[-1]+span:lims[-1]+2*span]
            ejy=sc[lims[-1]+span:lims[-1]+2*span,y]

        if len(ejx)>0:
            value = np.median(ejy)
            if value < 0 or np.isnan(value)==True or np.isnan(-value)==True:
                value = 0.
            nejX = np.hstack((nejX,np.median(ejx)))
            nejY = np.hstack((nejY,value))
        tck = scipy.interpolate.splrep(nejX,nejY,k=1)

        scat[:lims[-1]+2*span,y] = scipy.interpolate.splev(ejeX,tck)[:lims[-1]+2*span]

    scat = scipy.signal.medfilt(scat,[15,15])

    return scat

"""
===================================================================================================================
"""

def Red_orders(data,oversample, Plot='False'):

    nord=64
    exap = 25*oversample
    data[3990*oversample:data.shape[0],0:2900*oversample] = 0
    x2=np.arange(0,data.shape[1])

    #Mask out the very top and bottom of the chip to avoid the missing (bottom) and incomplete (top) orders.
    for i in range(data.shape[1]):

        if oversample == 1:
            bot_ord=int(3.51725981e-14*i**4-1.85310262e-10*i**3+2.62172451e-05*i**2-7.38664076e-02*i+85)
            data[0:bot_ord*oversample,i] = 0
        if oversample == 5:
            bot_ord=int(3.19571995e-16*i**4-1.01910019e-12*i**3+5.01186634e-06*i**2-7.22285048e-02*i+425)
            data[0:bot_ord,i] = 0
        if oversample == 10:
            bot_ord = int(-2.50295199e-16*i**4+2.21005966e-11*i**3 + 1.94806583e-06*i**2 -6.64977567e-02*i + .904082141e+03)
            data[0:bot_ord,i] = 0

    if Plot == 'True':
        plt.imshow(data,origin='lower',vmin=0,vmax=1000)
        plt.show()


    #Define the x range over which to find the order centers, and the step
    pix_low=50*oversample
    pix_high=3800*oversample
    step = 10*oversample

    w, h = int((pix_high-pix_low)/step), nord

    order_pos = [[0 for x in range(w)] for y in range(h)]

    strip=0
    for move in range(pix_low,pix_high,step):
        d = data[:,move]
        ejx = np.arange(len(d))
        ccf=[]
        refw = 1*exap
        sigc = 0.5*exap
        i = 0
        
        if Plot == 'True':
            plt.plot(d)

        while i < data.shape[0]:
            if i-refw < 0:
                refw2=int(refw/2*oversample)
                if oversample == 5:
                    refw2=int(refw*1/5*oversample)
                if oversample == 10:
                    refw2=int(refw*1/10*oversample)
                x = ejx[:i+refw2+1]
                y = d[:i+refw2+1]
            elif i + refw +1 > data.shape[0]:
                x = ejx[i-refw:]
                y = d[i-refw:]
            else:
                x = ejx[i-refw:i+refw+1]
                y = d[i-refw:i+refw+1]

            #Given the image slicer, the orders have a tridant shape so create one for cross-correlation.
            amp=1000
            wid=3.5*oversample
            wid2=3.*oversample
            tridant= (amp+300)*np.exp(-(x-np.mean(x))**2/(2*wid**2))+amp*np.exp(-(x-np.mean(x)+10*oversample)**2/(2*wid2**2))+amp*np.exp(-(x-np.mean(x)-10*oversample)**2/(2*wid2**2))

            ccf.append(np.add.reduce(y*tridant))
            i+=1
#        if Plot == 'True':
#            plt.plot(tridant)
#            plt.show()
            
        #Find the peaks in the ccf
        peaks_new,_=find_peaks(ccf,distance=30*oversample)
        
        pos = []

        #Pick the peaks that make sense
        for i in range(len(peaks_new)):
            if peaks_new[i] <= 1000*oversample:
                pos.append(peaks_new[i])
            if peaks_new[i] > 1000*oversample and d[peaks_new[i]] > 200:
                pos.append(peaks_new[i])
        pos=(np.array(pos))
        
        #Check to see if the right number of orders has been found. If not, plot what is found
        if len(pos) == nord:
            for j in range(len(pos)):
                order_pos[j][strip] = pos[j]
            strip += 1
        else:
            if (Plot == 'True'):
                plt.plot((ccf/np.max(ccf))*np.max(d))
                plt.plot(pos,d[pos],'go')
                plt.xlabel(str(len(pos)))
                plt.show()

    if Plot == 'True':
        plt.imshow(data,origin='lower',vmin=0,vmax=1000,aspect='auto')
        
    for i in range(nord):
        x2=np.arange((data.shape[1]))
        x=np.arange(pix_low,pix_high,step)
        if Plot == 'True':
            plt.plot(x,order_pos[i][:],'bo')

        #Take the results and fit with a 4th order poly to the order centers and compile them to write out
        #Delete any points that are still zero
        y=np.array(order_pos[i][:])
        ii=np.where(y == 0)
        y = np.delete(y,ii)
        x = np.delete(x,ii)

        fit=np.polyfit(x,y,deg=4)
        y_fit  = fit[0]*x**4+fit[1]*x**3+fit[2]*x**2+fit[3]*x+fit[4]
        order_shape = fit[0]*x2**4+fit[1]*x2**3+fit[2]*x2**2+fit[3]*x2+fit[4]
        residual = y - y_fit
        
        if i ==0 :
            acoefs = fit
        else:
            acoefs = np.vstack((acoefs,fit))
        if Plot == 'True':
            plt.plot(x2,order_shape)
            
    if Plot == 'True':
        plt.show()

    return acoefs,len(acoefs)

"""
===================================================================================================================
"""

def Blu_orders(data,oversample, Plot='False'):
    nord=84
    exap = 23*oversample
#    data[:,0:40*oversample] = 0
    data[0:20*oversample,:] = 0
#    data[0:40*oversample,0:300*oversample] = 0
#
#    #Mask out the very top and bottom of the chip to avoid the missing (bottom) and incomplete (top) orders.
#    for i in range(data.shape[1]):
#
#        if oversample == 1:
#            bot_ord=int(3.51725981e-14*i**4-1.85310262e-10*i**3+2.62172451e-05*i**2-7.38664076e-02*i+85)
#            data[0:bot_ord*oversample,i] = 0
#        if oversample == 5:
#            bot_ord=int(3.19571995e-16*i**4-1.01910019e-12*i**3+5.01186634e-06*i**2-7.22285048e-02*i+425)
#            data[0:bot_ord,i] = 0
#        if oversample == 10:
#            bot_ord = int(-2.50295199e-16*i**4+2.21005966e-11*i**3 + 1.94806583e-06*i**2 -6.64977567e-02*i + .904082141e+03)
#            data[0:bot_ord,i] = 0
#
#    if Plot == 'True':
#        plt.imshow(data,origin='lower',vmin=0,vmax=1000)
#        plt.show()


    #Define the x range over which to find the order centers, and the step
    pix_low=50*oversample
    pix_high=2020*oversample
    step = 10*oversample

    w, h = int((pix_high-pix_low)/step), nord+10

    order_pos = [[np.nan for x in range(w)] for y in range(h)]
    final_ord = [[np.nan for x in range(w)] for y in range(nord)]

    strip=0
    for move in range(pix_low,pix_high,step):
        d = np.sum(data[:,move-5*oversample:move+5*oversample],axis=1)
        ejx = np.arange(len(d))
        ccf=[]
        refw = 1*exap
        sigc = 0.5*exap
        i = 0
        
        if Plot == 'True':
            plt.plot(d)

        while i < data.shape[0]:
            if i-refw < 0:
                refw2=int(refw/2*oversample)
                if oversample == 5:
                    refw2=int(refw*1/5*oversample)
                if oversample == 10:
                    refw2=int(refw*1/7*oversample)
                x = ejx[:i+refw2+1]
                y = d[:i+refw2+1]
            elif i + refw +1 > data.shape[0]:
                x = ejx[i-refw:]
                y = d[i-refw:]
            else:
                x = ejx[i-refw:i+refw+1]
                y = d[i-refw:i+refw+1]

            #Given the image slicer, the orders have a tridant shape so create one for cross-correlation.
            amp=500
            wid=2*oversample
            wid2=2.*oversample
            tridant= (amp+50)*np.exp(-(x-np.mean(x))**2/(2*wid**2))+amp*np.exp(-(x-np.mean(x)+6*oversample)**2/(2*wid2**2))+amp*np.exp(-(x-np.mean(x)-6*oversample)**2/(2*wid2**2))

            ccf.append(np.add.reduce(y*tridant))
            i+=1
#        if Plot == 'True':
#            plt.plot(tridant)
#            plt.show()
            
        #Find the peaks in the ccf
        peaks_new,_=find_peaks(ccf,distance=20*oversample)
        
        pos = []

        #Pick the peaks that make sense
        for i in range(len(peaks_new)):
            if peaks_new[i] <= 1000*oversample:
                pos.append(peaks_new[i])
            if (1000*oversample < peaks_new[i] <=2000*oversample):
                if d[peaks_new[i]] > 200*oversample:
                    pos.append(peaks_new[i])
            if (2000*oversample < peaks_new[i] < 4150*oversample) and d[peaks_new[i]] > 500*oversample:
                pos.append(peaks_new[i])
        pos=(np.array(pos))

        #Check to see if the right number of orders has been found. If not, plot what is found
        if len(pos) > 0.5*nord:
            for j in range(len(pos)):
                order_pos[j][strip] = pos[j]
            strip += 1
        else:
            if (Plot == 'True'):
                plt.plot((ccf/np.max(ccf))*np.max(d))
                plt.plot(pos,d[pos],'go')
                plt.xlabel(str(len(pos)))
                plt.show()
    order_pos = np.array(order_pos)
    Plot= 'True'
    if Plot == 'True':
        plt.imshow(data,origin='lower',vmin=0,vmax=100,aspect='auto')
        

    ord_count= 0
    for i in range(nord):
        x_count = 0+int(order_pos.shape[1]/2)
        x2=np.arange((data.shape[1]))
        x=np.arange(pix_low,pix_high,step)
        if Plot == 'True':
            plt.plot(x,order_pos[i][:],'bo')

        tmp_ord = []

        for j in range(int(order_pos.shape[1]/2), order_pos.shape[1],1):
            print("X_COUNT",x_count)
            if x_count == int(order_pos.shape[1]/2):
                print("count, i ,j ",ord_count, x_count, i ,j)
                final_ord[ord_count][x_count] = order_pos[i][j]
                x_count = x_count+1
            else:
                print("ORDERS",final_ord[ord_count][x_count-1],order_pos[i][j])
                if (final_ord[ord_count][x_count-1]-order_pos[i][j]) < 5:
                    final_ord[ord_count][x_count] = order_pos[i][j]
                    x_count +=1
                else:
                    for k in range(-5,5):
                        if -1 < i+k < nord+10:
                            print("ORDERS LOOP",final_ord[ord_count][x_count-1],order_pos[i+k][j] )
                            if (final_ord[ord_count][x_count-1]-order_pos[i+k][j])< 5:
                                final_ord[ord_count][x_count] = order_pos[i+k][j]
                                x_count += 1
                                break
                        else:
                            for l in range(-5,5):
                                if (j+l) < order_pos.shape[1]:
                                    print("ORDERS LOOP2",final_ord[ord_count][x_count-1],order_pos[i+k][j+l] )
                                    if (final_ord[ord_count][x_count-1]-order_pos[i+k][j+l])< 5:
                                        final_ord[ord_count][x_count] = order_pos[i+k][j+l]
                                        x_count += 1
                                        break
#                    print("ORDERS2",order_pos[i-1][j],final_ord[ord_count][j-1])
#                    if (final_ord[ord_count][j-1]-order_pos[i-1][j])< 5:
#                        final_ord[ord_count][j] = order_pos[i-1][j]
#                        x_count += 1
#                    else:
#                        print("ORDERS3",order_pos[i+1][j],final_ord[ord_count][j-1])
#                        if (final_ord[ord_count][j-1]-order_pos[i+1][j])< 5:
#                            final_ord[ord_count][j] = order_pos[i+1][j]
#                            x_count += 1
#                        else:
#                            print("ORDERS4",order_pos[i-2][j],final_ord[ord_count][j-1])
#                            if (final_ord[ord_count][j-1]-order_pos[i-2][j])< 5:
#                                final_ord[ord_count][j] = order_pos[i-2][j]
#                                x_count += 1
#                            else:
#                                print("ORDERS5",order_pos[i+2][j],final_ord[ord_count][j-1])
#                                if (final_ord[ord_count][j-1]-order_pos[i+2][j])< 5:
#                                    final_ord[ord_count][j] = order_pos[i+2][j]
#                                    x_count += 1
#                                else:
#                                    print("ORDERS6",order_pos[i-3][j],final_ord[ord_count][j-1])
#                                    if (final_ord[ord_count][j-1]-order_pos[i-3][j])< 5:
#                                        final_ord[ord_count][j] = order_pos[i-3][j]
#                                        x_count += 1
#                                    else:
#                                        print("ORDERS7",order_pos[i+3][j],final_ord[ord_count][j-1])
#                                        if (final_ord[ord_count][j-1]-order_pos[i+3][j])< 5:
#                                            final_ord[ord_count][j] = order_pos[i+3][j]
#                                            x_count += 1
#                                        else:
#                                            print("ORDERS8",order_pos[i-4][j],final_ord[ord_count][j-1])
#                                            if (final_ord[ord_count][j-1]-order_pos[i-4][j])< 5:
#                                                final_ord[ord_count][j] = order_pos[i-4][j]
#                                                x_count += 1
#                                            else:
#                                                print("ORDERS9",order_pos[i+4][j],final_ord[ord_count][j-1])
#                                                if (final_ord[ord_count][j-1]-order_pos[i+4][j])< 5:
#                                                    final_ord[ord_count][j] = order_pos[i+4][j]
#                                                    x_count += 1
#                                                else:
#                                                    print("ORDERS10",order_pos[i-5][j],final_ord[ord_count][j-1])
#                                                    if (final_ord[ord_count][j-1]-order_pos[i-5][j])< 5:
#                                                        final_ord[ord_count][j] = order_pos[i-5][j]
#                                                        x_count += 1
#                                                    else:
#                                                        print("ORDER11",order_pos[i+5][j],final_ord[ord_count][j-1])
#                                                        if (final_ord[ord_count][j-1]-order_pos[i+5][j])< 5:
#                                                            final_ord[ord_count][j] = order_pos[i+5][j]
#                                                            x_count += 1
        ord_count +=1

        #Take the results and fit with a 4th order poly to the order centers and compile them to write out
        #Delete any points that are still zero
        if i < 84:
            print(i)
            plt.plot(final_ord[ord_count-1][:],'o')
        y=np.array(order_pos[i][:])
        ii=np.where(y == 0)
        y = np.delete(y,ii)
        x = np.delete(x,ii)
        
        if len(x) > 0:
            fit=np.polyfit(x,y,deg=4)
            y_fit  = fit[0]*x**4+fit[1]*x**3+fit[2]*x**2+fit[3]*x+fit[4]
            order_shape = fit[0]*x2**4+fit[1]*x2**3+fit[2]*x2**2+fit[3]*x2+fit[4]
            residual = y - y_fit
        
        if i ==0 :
            acoefs = fit
        else:
            acoefs = np.vstack((acoefs,fit))
        if Plot == 'True':
            plt.plot(x2,order_shape)
            
    if Plot == 'True':
        plt.show()

    return acoefs,len(acoefs)

def get_hrs_echelle_order(lambda_ang):
    """
    Returns the echelle order that a specified wavelength is in the free
    spectral range of.  An example use case is to identify the spectral orders
    in a wavelength solution when the "order number" in the file is just the
    order index (starting from 0).  This function expects an input wavelength
    in Angstroms, but will reluctantly convert to nm if the input is in the
    range 350-900.

    Args:
        lambda_ang (Angstroms)

    Returns:
        order - echelle order - 137-103 on the green CCD and 102-71 on the red CCD
    """

    if lambda_ang < 900 and lambda_ang > 350:
        lambda_ang *= 10
        print("Converting input wavelength from nm to Ang in get_kpf_echelle_order.")
        
    ech_order = -1 # default value if match not found
    fsr = {
        85 : [5424.8    ,    5554    ],
        86 : [5363.2    ,    5489.4    ],
        87 : [5301.6    ,    5424.8    ],
        88 : [5242.7    ,    5363.2    ],
        89 : [5183.8    ,    5301.6    ],
        90 : [5127.5    ,    5242.7    ],
        91 : [5071.2    ,    5183.8    ],
        92 : [5017.25    ,    5127.45    ],
        93 : [4963.3    ,    5071.1    ],
        94 : [4911.6    ,    5017.2    ],
        95 : [4859.9    ,    4963.3    ],
        96 : [4810.3    ,    4911.6    ],
        97 : [4760.7    ,    4859.9    ],
        98 : [4713.1    ,    4810.3    ],
        99 : [4665.5    ,    4760.7    ],
        100 : [4619.8    ,    4713.1    ],
        101 : [4574.1    ,    4665.5    ],
        102 : [4530.1    ,    4619.8    ],
        103 : [4486.1    ,    4574.1    ],
        104 : [4443.8    ,    4530.1    ],
        105 : [4401.5    ,    4486.1    ],
        106 : [4360.75    ,    4443.85    ],
        107 : [4320    ,    4401.6    ],
        108 : [4280.7    ,    4360.8    ],
        109 : [4241.4    ,    4320    ],
        110 : [4203.55    ,    4280.75    ],
        111 : [4165.7    ,    4241.5    ],
        112 : [4129.2    ,    4203.6    ],
        113 : [4092.7    ,    4165.7    ],
        114 : [4057.4    ,    4129.2    ],
        115 : [4022.1    ,    4092.7    ],
        116 : [3988    ,    4057.4    ],
        117 : [3953.9    ,    4022.1    ],
        118 : [3921    ,    3988    ],
        119 : [3888.1    ,    3953.9    ],
        120 : [3856.2    ,    3921    ],
        121 : [3824.3    ,    3888.1    ],
        122 : [3793.5    ,    3856.2    ],
        123 : [3762.7    ,    3824.3    ],
        124 : [3732.8    ,    3793.5    ],
        125 : [3702.9    ,    3762.7    ],
        126 : [3674    ,    3732.8    ],
        127 : [3645.1    ,    3702.9    ],
    }
    for key, wavs in fsr.items():
        if lambda_ang < wavs[0] and lambda_ang > wavs[1]:
            ech_order = key
            
    return ech_order
