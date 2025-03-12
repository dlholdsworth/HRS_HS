#This code processes the HRS Reference Fibre HIGH STABILITY data.
#It includes Overscan removal, Trimming of the image, Gain correction,
#Bias calculation and removal

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse
import os,glob,copy
from datetime import datetime
import pytz
import dlh_curvature3,dlh_utils, dlh_background, dlh_wavecal, dlh_RV_calc, dlh_determine_pixel_offset
from dlh_wavecal import LineList
import level0corrections

import arrow
from astropy.time import Time
import barycorrpy
import scipy.constants as conts

from scipy.signal import find_peaks
from scipy.signal import savgol_filter

from lmfit import  Model
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate,signal,special,optimize,integrate
import scipy
from scipy.optimize import curve_fit, least_squares

from astroquery.simbad import Simbad

from tqdm import tqdm


if __name__ == '__main__':

    # clear the terminal
    os.system("clear")

    # read command line arguments
    parser = argparse.ArgumentParser(prog="HRS_Reduce_Stability.py",
                                     description="----------------- Base reduction for HRS data -----------------",
                                     epilog="HRSProcess.py \n"
                                            "Copyright (C) 2024 \n"
                                            "(D.L.Holdsworth, SAAO/SALT). \n"
                                            "This program comes with ABSOLUTELY NO WARRANTY. \n"
                                            "This is free software and you are welcome to use, modify and redistribute it. \n")

    parser.add_argument('--data_dir', type=str, default='./', action='store',help="Directory path of where image file(s) are located; <str>")
    parser.add_argument('--night', type=str, default='19991231', action='store',help="Night start of the data to reduce in YYYYMMDD; <str>")
    parser.add_argument('--mode', type=str, default='HS', action='store',help="HRS mode [LR,MR,HR,HS]; <str>")
    parser.add_argument('--arm', type=str, action='store',help="HRS arm [Blu,Red]; <str>")
    parser.add_argument('--plot', type=str, default='False', action='store',help="Diagnostic plots [True,False]; <str>")
    parser.add_argument('--sci_frame', type=str, default='Science', action='store',help="The type of Science frame to reduce (normally Science but can be Arc); <str>")
    
    """
    to run the Code:
    
    python HRSProcess8.py --arm Blu --sci_frame Arc --night 20230101 --data_dir ~/Desktop/SALT_HRS_DATA/
    
    The set up expects data to be in
    ~/Desktop/SALT_HRS_DATA/YYYY/MMDD/raw/
    and will create outputs in
    ~/Desktop/SALT_HRS_DATA/YYYY/MMDD/reduced/
    """

    args = parser.parse_args()
    data_dir = args.data_dir
    night = args.night
    mode = args.mode
    arm_colour = args.arm
    sci_frame = args.sci_frame
    Plot = args.plot
    
    year=night[0:4]
    mnday=night[4:8]

    flat_bkg = None

    # check that file directories exist

    if not os.path.exists(data_dir):
        print("\n   !!! ERROR: specified DATA directory does not"
              " exist: {} \n".format(data_dir))
        exit()
    
    if mode in ["LR","MR","HR"]:
        print ("\n   !!! Currently {} is not supported.\n".format(mode))
        exit()
        
    if arm_colour not in ["Blu","blu","Blue","blue","Red","red"]:
        print ("\n   !!! The arm requested ({}) is not supported. Options are Blu or Red.\n".format(arm))
        exit()
    
    if arm_colour in ["Blu","blu","Blue","blue"]:
        arm = "H"
        arm_colour = "Blu"
        naxis = 4050
        flat_std = 200
        gain = 2.406000
    if arm_colour in ["Red","red"]:
        arm = "R"
        arm_colour = "Red"
        naxis = 4130
        flat_std = 800
        gain = 2.301000
        
        
    #Set the oversampling parameter for regridding the data
    oversample = 10
    oversample = int(oversample)
    
    #Find all files in the data directory that correspond to the given arm and night
    year=night[0:4]
    mmdd=night[4:8]
    data_location = str(data_dir+arm_colour+'/'+year+'/'+mmdd+'/raw/*.fits')

    raw_files =sorted(glob.glob(data_location))
    
    #Set the out location for the science data.
    out_location_sci = str(data_dir+arm_colour+'/'+year+'/'+mmdd+'/reduced/')
    try:
        os.mkdir(out_location_sci)
    except Exception:
        pass
    
    #Often the FLAT files will not be obtained on the same night as the Science frame.
    #Therefore check the returned files for FLATS, if null search previous nights until
    #FLAT frames are found. Use the BIAS frames from that night to reduce the FLAT frames.
    
    Flat_files = []

    #Open the files in the current night to check for FLAT frames. If found, append to Flat_files
    for file in raw_files:
        hdu = fits.open(file)
        if (hdu[0].header["OBSTYPE"] == "Flat field" and hdu[0].header["FIFPORT"] == mode and hdu[0].header["PROPID"] == "CAL_FLAT"):
            Flat_files.append(file)
        hdu.close
    
    #If there are no flat files for the given night, iteratively check the previous nights/next for them.
    #Once found stop the search.
    if len(Flat_files) < 9:
        print("No suitable flats taken on night",night,"-- Looking for previous/next exposures...")
        prev_night = night
        next_night = night
        while len(Flat_files) < 9:
            prev_night = arrow.get(prev_night).shift(days=-1).format('YYYYMMDD')
            prev_year=prev_night[0:4]
            prev_mmdd=prev_night[4:8]
            prev_data_location = str(data_dir+arm_colour+'/'+prev_year+'/'+prev_mmdd+'/raw/*.fits')
            prev_data_dir = str(data_dir+arm_colour+'/'+prev_year+'/'+prev_mmdd+'/raw/')
            prev_raw_files =sorted(glob.glob(prev_data_location))
            for file in prev_raw_files:
                hdu = fits.open(file)
                if (hdu[0].header["OBSTYPE"] == "Flat field" and hdu[0].header["FIFPORT"] == mode and hdu[0].header["NAXIS1"] < naxis and np.std(hdu[0].data) > flat_std):
                    Flat_files.append(file)
                hdu.close
                
            if len(Flat_files) >= 9:
                print("Using FLATS from night",prev_night)
                flat_night = prev_night
                flat_raw_files = prev_raw_files
                flat_data_dir = prev_data_dir
                out_location = str(data_dir+arm_colour+'/'+prev_year+'/'+prev_mmdd+'/reduced/')
                try:
                    os.mkdir(out_location)
                except Exception:
                    pass
                break
            else:
                Flat_files = []
            
            next_night = arrow.get(next_night).shift(days=+1).format('YYYYMMDD')
            next_year=next_night[0:4]
            next_mmdd=next_night[4:8]
            next_data_location = str(data_dir+arm_colour+'/'+next_year+'/'+next_mmdd+'/raw/*.fits')
            next_data_dir = str(data_dir+arm_colour+'/'+next_year+'/'+next_mmdd+'/raw/')
            next_raw_files =sorted(glob.glob(next_data_location))
            
            for file in next_raw_files:
                hdu = fits.open(file)
                if (hdu[0].header["OBSTYPE"] == "Flat field" and hdu[0].header["FIFPORT"] == mode and hdu[0].header["NAXIS1"] < naxis and np.std(hdu[0].data) > flat_std):
                    Flat_files.append(file)
                hdu.close
                
            if len(Flat_files) >= 9:
                print("Using FLATS from night",next_night)
                flat_night = next_night
                flat_raw_files = next_raw_files
                flat_data_dir = next_data_dir
                out_location = str(data_dir+arm_colour+'/'+next_year+'/'+next_mmdd+'/reduced/')
                try:
                    os.mkdir(out_location)
                except Exception:
                    pass
                break
            else:
                Flat_files = []
    else:
        flat_raw_files = raw_files
        flat_data_dir = str(data_dir+arm_colour+'/'+year+'/'+mmdd+'/raw/')
        out_location = str(data_dir+arm_colour+'/'+year+'/'+mmdd+'/reduced/')
        try:
            os.mkdir(out_location)
        except Exception:
            pass
        flat_night = night

    #We need to correct the older flats with the older bias files, so do that here.
    
    #Step 0 -- Correct for gain and trim overscan, flip Red frames so orders run blue->red bottom to top (i.e. same as blue)
    print("\n   !!! Level 0 corrections\n")
    flat_all_files = level0corrections.execute(flat_raw_files,flat_data_dir,arm,out_location)

    #Step 1 -- Calcualte the master Bias frame for the FLAT
    print("\n   !!! Calculating Master Bias frame for Flat\n")
    flat_master_bias = dlh_utils.create_masterbias(flat_all_files,flat_data_dir, arm, flat_night, out_location,Plot)

    #Step 2 -- Create a master Flat frame
    print("\n   !!! Calculating Master Flat frame for Flat files\n")
    master_flat = dlh_utils.create_masterflat(flat_all_files,flat_data_dir,mode,arm,flat_night,flat_master_bias,out_location,Plot)
    
    #Free some memory
    del flat_master_bias
    
    #Step 3 -- Define the Orders
    print("\n   !!! Defining Orders\n")

    #Check if Orders are defined for the night that the flats were taken
    old_defined = glob.glob(out_location+"ord_default_"+arm+flat_night+"_OS_"+str(oversample)+".npz")
    
    if len(old_defined) == 0:
        print("\n      +++ Creating new orders \n")
        if oversample != 1:
            data=scipy.ndimage.zoom(master_flat, oversample, order=0)
        else:
            data=master_flat
        """We want to loop over al pixel rows to trace the orders in the column data. Then, identify orders as being peaks in the data, fit a gaussian to find the centre of the order in that pixel. Then define the polynomial for each order by fitting the centrals. Hopefully this will overcome the issues relating to the slit function (trident shaped).
        """

        #There are 84 orders expected from the Blu arm.
        #There are 64 orders expected from the Red arm.

        columns = data.shape[1]
        #Define the array to hold the postions
        order_locations = np.zeros((100,columns))#[[0 for x in range(w)] for y in range(h)]
        order_x = []
        order_y = []

        if arm == 'H':
            orders,nord = dlh_utils.get_them_B(data,23*oversample,4,oversample,startfrom=0,nsigmas=0.,mode=1,endat=-1,Plot=Plot)
            if nord != 84:
                print("Number of Orders identified: ",nord, " does not match expected 84. Exiting")
                exit()
        if arm == 'R':
            orders,nord = dlh_utils.get_them_R(data[0*oversample:data.shape[0]-100*oversample,:],28*oversample,4,oversample,startfrom=0,nsigmas=5.,mode=1,endat=-1,Plot=Plot)
            if nord != 64:
                print("Number of Orders identified: ",nord, " does not match expected 64. Exiting")
                exit()

        c_all = orders
        if (Plot == "True"):
            plt.title("New Orders Defined")
            plt.imshow(data,origin='lower',vmax=3000)
        x=np.arange(data.shape[1])
        for i in range(len(c_all)):
            y=c_all[i][0]*x**4 + c_all[i][1]*x**3 + c_all[i][2]*x**2 + c_all[i][3]*x + c_all[i][4]
            if (Plot == "True"):
                plt.plot(x,y)
        if (Plot == "True"):
            plt.show()

        order_file = out_location+"ord_default_"+arm+flat_night+"_OS_"+str(oversample)+".npz"
        n = np.arange(len(c_all))
        column_range = np.array([[0, columns] for i in n])
        if arm =='H':
            np.savez(order_file, orders=orders, column_range=column_range)
        if arm =='R':
            np.savez(order_file, orders=orders, column_range=column_range)

    else:
        #Read predetermined orders
        print("\n      +++ Reading order definition \n")
        order_file = out_location+"ord_default_"+arm+flat_night+"_OS_"+str(oversample)+".npz"
        file_dic = dict(np.load(old_defined[0],allow_pickle=True))
        column_range = file_dic["column_range"]
        c_all=file_dic["orders"]
        n_ord=len(c_all)
        
    #Now we repeat Step 0 and 1 for the calibrations for the night that the data were taken.
    print("\n   !!! Calibrating frames for target night\n")
    data_dir_orig = str(data_dir+arm_colour+'/'+year+'/')
    data_dir=str(data_dir+arm_colour+'/'+year+'/'+mmdd+'/raw/')
    
    all_files  = level0corrections.execute(raw_files,data_dir,arm,out_location_sci)
    master_bias = dlh_utils.create_masterbias(all_files,data_dir, arm, night, out_location_sci, Plot)
    
    print("\n   !!! Finding and correcting Science Frames\n")
    
    #Step 4 -- Find science frames and correct them
    Science_files = []
    if len(all_files) > 0:
        for file in all_files:
            #Check if a corrected file has been created.
            new_set = glob.glob(str(out_location_sci+"b"+file.removeprefix(out_location_sci)))
            if len(new_set) == 1:
                corrected = (new_set[0])
                hdu=fits.open(corrected)
                if (hdu[0].header["OBSTYPE"] == sci_frame and hdu[0].header["FIFPORT"] == mode):
                    Science_files.append(new_set[0])
                #We also need to have the daily reference arc reduced for the wavelenght solution at the end.
                if (hdu[0].header["I2STAGE"] == "Reference Fibre" and hdu[0].header["EXPTIME"] == 30. and hdu[0].header["PROPID"] == "CAL_STABLE"):
                    Science_files.append(new_set[0])
                hdu.close()
            else:
                hdu=fits.open(file)
                if (hdu[0].header["OBSTYPE"] == sci_frame and hdu[0].header["FIFPORT"] == mode):
                    Science_files.append(file)
                    hdu[0].data = (hdu[0].data) - master_bias
                    file_out=str(out_location_sci+"b"+file.removeprefix(out_location_sci))
                    #Need to add some comments to header for the files used to correct image.
                    hdu.writeto(file_out,overwrite=True)
                elif (hdu[0].header["I2STAGE"] == "Reference Fibre" and hdu[0].header["EXPTIME"] == 30. and hdu[0].header["PROPID"] == "CAL_STABLE"):
                    Science_files.append(file)
                    hdu[0].data = (hdu[0].data) - master_bias
                    file_out=str(out_location_sci+"b"+file.removeprefix(out_location_sci))
                    hdu.writeto(file_out,overwrite=True)
                hdu.close()
            del new_set
    else:
        print ("\n   !!! No files found in {}. Check the arm ({}) and night ({}). Exiting.\n".format(data_location,arm,night))
        exit()

    if len(Science_files) <1:
        print ("\n   !!! No Science files found in {}. Check the arm ({}) and night ({}). Exiting.\n".format(data_location,arm,night))
        exit()
        
    #Free some memory
    del master_bias
    
    #Step 5 -- Calculate the slit curvature (PyReduce).
    print("\n   !!! Calculating slit curvature\n")
    
    old_curve = glob.glob(out_location_sci+"Curvature_"+arm+night+"_OS_"+str(oversample)+".npz")
    if len(old_curve) == 0:
    
        #For this procedure, we can use the daily arc that is taken with ThAr down both Fibres. That is the 'Reference Fibre HIGH STABILITY Arc 30.0'
        dailyfiles = sorted(glob.glob(out_location_sci+"/bog*.fits"))
        for file in dailyfiles:
            with fits.open(file) as hdu:
                if (hdu[0].header["OBJECT"] == "Arc" and hdu[0].header["I2STAGE"] == "Reference Fibre" and hdu[0].header["PROPID"] == "CAL_STABLE" and hdu[0].header["EXPTIME"] == 30.):
                    daily_arc = file
                    hdu.close()
                    break
        if arm == 'R':
            daily_arc='/Users/daniel/Desktop/SALT_HRS_DATA/Red/2024/0508/raw/R202405080006.fits'
        print("Daily ref arc",daily_arc)
        tilt,shear = dlh_curvature3.execute(daily_arc,order_file,master_flat,Plot,night,oversample,arm,data_dir,out_location_sci)

    else:
        print("\n      +++ Reading slit curvature\n")
        curve = np.load(out_location_sci+"/Curvature_"+arm+night+"_OS_"+str(oversample)+".npz", allow_pickle=True)
        tilt = curve["tilt"]
        shear = curve["shear"]
    
    
    print("\n   !!! Extracting Science Frames")
    #Step 6 -- Extract the science frames
    
    #Find the file that contains the order trace, open file and store contents for use
    order_file = out_location+"ord_default_"+arm+flat_night+"_OS_"+str(oversample)+".npz"
    file_dic=np.load(order_file, allow_pickle=True)
    column_range = file_dic["column_range"]
    c_all=file_dic["orders"]
    n_ord=len(c_all)
    #print("Number of Orders:",n_ord)

    #Find and open the master flat file, this is to calculate the slit function
    if oversample != 1:
        flat = scipy.ndimage.zoom(master_flat, oversample, order=0)
    else:
        flat = master_flat
    nrow,ncol = flat.shape

    #Check if Slit Function output is defined for the night that the flats were taken
    old_sf = glob.glob(out_location+"P_"+arm+flat_night+".fits")
    
    #Check if optimally extracted flat exists
    old_flat_extraction=glob.glob(out_location+"Flat_extraction_"+arm+flat_night+".fits")

    #Find the science frames (crude right now)
    files = sorted(glob.glob(out_location_sci+"bog"+arm+night+"*.fits"))
    sci_files = []
    for file in files:
        hdr=fits.open(file)
        if (sci_frame == 'Science'):
            if (hdr[0].header['OBSTYPE'] == 'Science'):
                sci_files.append(file)
            if (hdr[0].header['OBSTYPE'] == 'Arc' and hdr[0].header['I2STAGE'] == 'Reference Fibre' and hdr[0].header["EXPTIME"] == 30. and hdr[0].header['PROPID'] == "CAL_STABLE"):
                sci_files.append(file)
        if (sci_frame == 'Arc'):
            if (hdr[0].header['OBSTYPE'] == 'Arc' and hdr[0].header['I2STAGE'] == 'Reference Fibre' and hdr[0].header["EXPTIME"] == 30. and hdr[0].header['PROPID'] == "CAL_STABLE"):
                sci_files.append(file)
        hdr.close()

    count2=1
    for frame in tqdm(range(len(sci_files)), desc="Frame"):
 #   for science_file in files:
        science_file = sci_files[frame]

        #Re-read the order information as we change it below (hard coded needs to change)
        c_all=file_dic["orders"]
        count2 +=1
        
        #Check if file is already processed
        file_out=str(out_location_sci+"HRS_E_"+science_file.removeprefix(out_location_sci))
        
        out_test=glob.glob(file_out)
        if len(out_test) == 0:
            #File has not been processed so continue
            
            #Open the science frame
            hdu_sci = fits.open(science_file)
            data=hdu_sci[0].data
            hdr= hdu_sci[0].header
            hdu_sci.close()
            if ((hdu_sci[0].header["OBSTYPE"] == sci_frame or hdu_sci[0].header["I2STAGE"] == 'Reference Fibre') and hdu_sci[0].header["FIFPORT"] == mode):
                if oversample != 1:
                    data = scipy.ndimage.zoom(data, oversample, order=0)
                #Set read noise noise (hard coded -- change)
                ronoise = 6.
                
                #Define the apertures
                if oversample != 1:
                    master_flat_zoom = scipy.ndimage.zoom(master_flat, oversample, order=0)
                else:
                    master_flat_zoom = master_flat
                if arm =='H':
                    ext_aperture,c_all = dlh_utils.order_width_B(n_ord,master_flat_zoom,oversample,c_all)
                    ext_aperture,_ = dlh_utils.order_width_B(n_ord,master_flat_zoom,oversample,c_all)

                #Fix the parameters so that they are all within in the chip
                extraction_width, column_range, c_all = dlh_utils.fix_parameters(ext_aperture, column_range, c_all, nrow, ncol, n_ord)
                ext_aperture,_,_=dlh_utils.fix_parameters(ext_aperture,column_range,c_all,nrow,ncol,n_ord)
                
                if (Plot =="True"):
                    #This plots the image and the orders over laid with the top (blue) and bottom (green) extent of the order to be extracted.
                    plt.imshow(flat,origin='lower',vmax=3000)
                    x=np.arange(data.shape[1])
                    for i in range(len(c_all)):
                        y=c_all[i][0]*x**4 + c_all[i][1]*x**3 + c_all[i][2]*x**2 + c_all[i][3]*x + c_all[i][4]
                        plt.plot(x,y,'r-')
                        y=c_all[i][0]*x**4 + c_all[i][1]*x**3 + c_all[i][2]*x**2 + c_all[i][3]*x + c_all[i][4]+ext_aperture[i][0]
                        plt.plot(x,y,'b-')
                        y=c_all[i][0]*x**4 + c_all[i][1]*x**3 + c_all[i][2]*x**2 + c_all[i][3]*x + c_all[i][4]-ext_aperture[i][1]
                        plt.plot(x,y,'g-')
                    plt.show()
            
                
                #Correct for the slit tilt in both the science and flat frames
                #This extracts each order, straigtens the tilt and reinserts to the original image.
                nrow, ncol = flat.shape
                x = np.arange(ncol)
                mask2 = np.full(data.shape, True)
                # Correct for tilt and shear
                # For each row of the rectified order, interpolate onto the shifted row

#                tilt= None
#                shere = None

                #Loop over all orders
                if tilt is not None and shear is not None:
                    #print("Correcting for slit tilt...")
                    for i in tqdm(range(n_ord), desc="Order tilt correction",leave=False):
                        x_left_lim = column_range[i, 0]
                        x_right_lim = column_range[i, 1]

                        # Rectify the image, i.e. remove the shape of the order
                        # Then the center of the order is within one pixel variations
                        ycen = np.polyval(c_all[i], x).astype(int)
                        yb, yt = ycen - extraction_width[i,0], ycen + extraction_width[i,1]
                        index = dlh_utils.make_index(yb, yt, x_left_lim, x_right_lim)
                        mask2[index] = False

                        #Do the actual correction
                        img_order = dlh_utils.correct_for_curvature(
                                data[index],
                                tilt[i, x_left_lim:x_right_lim],
                                shear[i, x_left_lim:x_right_lim],
                                extraction_width[i])

                        flat_order = dlh_utils.correct_for_curvature(
                                flat[index],
                                tilt[i, x_left_lim:x_right_lim],
                                shear[i, x_left_lim:x_right_lim],
                                extraction_width[i])
                                
#                        plt.imshow(data[index],origin='lower',vmin=0,vmax=100,aspect='auto')
#                        plt.show()
#                        plt.imshow(img_order,origin='lower',vmin=0,vmax=100,aspect='auto')
#                        plt.show()

                        #Reinsert the corrected orders to the original image
                        data[index]=img_order
                        flat[index]=flat_order


                #Remove the oversampling and redefine the orders
                if oversample != 1:
                    data_small = scipy.ndimage.zoom(data, 1/oversample,order=0)
                    flat_small = scipy.ndimage.zoom(flat, 1/oversample,order=0)
                else:
                    data_small = data
                    flat_small = flat
                
                oversample_2 = 1
                oversample_2 = int(oversample_2)
                
                nrow_small, ncol_small = flat_small.shape
                
                #Redefine the Orders on the smaller image
                print("\n   !!! Defining Orders\n")
        
                #Check if Orders are defined for this night
                old_orders_small = glob.glob(out_location+"ord_default_"+arm+flat_night+".npz")

                if len(old_orders_small) == 0:
                    #There are 84 orders expected from the Blu arm.
                    #There are 64 orders expected from the Blu arm.
                    
                    columns_small = flat_small.shape[1]
                    #Define the array to hold the postions
                    order_locations = np.zeros((100,columns_small))#[[0 for x in range(w)] for y in range(h)]
                    order_x = []
                    order_y = []

                    if arm == 'H':
                        orders_small,nord = dlh_utils.get_them_B(flat_small,23*oversample_2,4,oversample_2,startfrom=0,nsigmas=-0.1,mode=1,endat=-1,Plot=Plot)
                        if nord != 84:
                            print("Number of Orders identified: ",nord, " does not match expected 84. Exiting")
                            exit()
                    if arm == 'R':
                        #orders_small,nord =dlh_utils.get_them_R(flat_small[0:flat_small.shape[0]-150*oversample_2,:],28*oversample_2,4,oversample_2,startfrom=0,nsigmas=5.,mode=1,endat=-1)
                        orders_small,nord = dlh_utils.Red_orders(flat_small,oversample_2,Plot=Plot)
                        if nord !=64:
                            print("Number of Orders identified: ",nord, " does not match expected 64. Exiting")
                            exit()

                    c_all_small = orders_small
                    if (Plot == "True"):
                        plt.imshow(flat_small,origin='lower',vmax=3000)
                    x=np.arange(columns_small)
                    for i in range(len(c_all_small)):
                        y=c_all_small[i][0]*x**4 + c_all_small[i][1]*x**3 + c_all_small[i][2]*x**2 + c_all_small[i][3]*x + c_all_small[i][4]
                        if (Plot == "True"):
                            plt.plot(x,y)
                    if (Plot == "True"):
                        plt.show()

                    order_file_small =out_location+"ord_default_"+arm+flat_night+".npz"
                    n = np.arange(len(c_all_small))
                    column_range_small = np.array([[0, columns_small] for i in n])
                    np.savez(order_file_small, orders=orders_small, column_range=column_range_small)
                else:
                    print("\n      +++ Reading Order File", old_orders_small[0])
                    file_dic_small = dict(np.load(old_orders_small[0],allow_pickle=True))
                    column_range_small = file_dic_small["column_range"]
                    c_all_small=file_dic_small["orders"]
                    n_ord=len(c_all_small)
                
                
                if arm =="H":
                    #Define the apertures
                    ext_aperture_small,c_all_small = dlh_utils.order_width_B(n_ord,flat_small,oversample_2,c_all_small)
                    ext_aperture_small,_ = dlh_utils.order_width_B(n_ord,flat_small,oversample_2,c_all_small)
 
                    #Fix the parameters so that they are all within in the chip
                    extraction_width_small, column_range_small, c_all_small = dlh_utils.fix_parameters(ext_aperture_small, column_range_small, c_all_small, nrow_small, ncol_small, n_ord)
                    ext_aperture_small,_,_=dlh_utils.fix_parameters(ext_aperture_small,column_range_small,c_all_small,nrow_small,ncol_small,n_ord)

                
                elif arm =="R":
                #Define the aperture for each order as the aperture expands and we go up the chip
                    xdw = np.zeros((n_ord, 2))
                    mid=np.zeros((n_ord))
                    ext_aperture =np.zeros((n_ord, 2))
                    for i in range(n_ord):
                        if i % 2: #Width of order for tilt correction for odd order (top P Fiber)
                            ext_aperture[i][0]=13*oversample_2+np.sqrt(i)
                            ext_aperture[i][1]=17*oversample_2

                        else: #Width of order for tilt correction
                            ext_aperture[i][0]=13*oversample_2+np.sqrt(i)
                            ext_aperture[i][1]=16*oversample_2

                        #Using the width information, change the centre of the order trance so we are symetrical about the centre for the optimal extraction.
                        c_all_small[i][4]=c_all_small[i][4]+0.52*(ext_aperture[i][0]-ext_aperture[i][1])
                        ext_aperture[i][0],ext_aperture[i][1]=np.mean(ext_aperture[i])*1.01,np.mean(ext_aperture[i])*1.01

                    #Fix the parameters so that they are all within in the chip
                    extraction_width_small, column_range_small, c_all_small = dlh_utils.fix_parameters(ext_aperture, column_range_small, c_all_small, nrow_small, ncol_small, n_ord)
                    ext_aperture,_,_=dlh_utils.fix_parameters(ext_aperture_small,column_range_small,c_all_small,nrow_small,ncol_small,n_ord)

                if (Plot =="True"):
                    #This plots the image and the orders over laid with the top (blue) and bottom (green) extent of the order to be extracted.
                    plt.imshow(flat_small,origin='lower',vmax=3000)
                    x=np.arange(data_small.shape[1])
                    for i in range(len(c_all_small)):
                        y=c_all_small[i][0]*x**4 + c_all_small[i][1]*x**3 + c_all_small[i][2]*x**2 + c_all_small[i][3]*x + c_all_small[i][4]
                        plt.plot(x,y,'r-')
                        y=c_all_small[i][0]*x**4 + c_all_small[i][1]*x**3 + c_all_small[i][2]*x**2 + c_all_small[i][3]*x + c_all_small[i][4]+ext_aperture_small[i][0]
                        plt.plot(x,y,'b-')
                        y=c_all_small[i][0]*x**4 + c_all_small[i][1]*x**3 + c_all_small[i][2]*x**2 + c_all_small[i][3]*x + c_all_small[i][4]-ext_aperture_small[i][1]
                        plt.plot(x,y,'g-')
                    plt.show()

                    
#                    #This calculates the background using the science data.
                print("\n   !!! Calculating background scatter")
                
                ext_aperture_2 = []
                for i in range(n_ord):
                    ext_aperture_2.append(ext_aperture_small[i][0])
                
                sigma_cutoff = 2.
                scatter_degree=3
                bw = 15 #Boarder width for background
                
                background = dlh_background.execute(data_small,sigma_cutoff,scatter_degree,bw, n_ord,ncol_small,nrow_small, ext_aperture_small,c_all_small,column_range_small, Plot)
                data_small -=  background

                
                if arm =="H":
                        #Set parameters for optimal extraction
                    NSigma_Marsh = 6.       #Sigma detection
                    S_Marsh = .4            #Separation of the polynomials to be fit
                    N_Marsh = 3             #Number of polynomial coefficients
                    Marsh_alg = 0           #0 for March curved, 1 for Horne?
                    npools=10               #Num ccd pools
                    NCosmic_Marsh = 2       #Cosmic ray rejection (if too small also rejects bright arc lines!)
                    gain = 1.               #Already corrected
                
                if arm =="R":
                        #Set parameters for optimal extraction
                    NSigma_Marsh = 6.       #Sigma detection
                    S_Marsh = .4            #Separation of the polynomials to be fit
                    N_Marsh = 3             #Number of polynomial coefficients
                    Marsh_alg = 0           #0 for March curved, 1 for Horne?
                    npools=10               #Num ccd pools
                    NCosmic_Marsh = 2      #Cosmic ray rejection (if too small also rejects bright arc lines!)
                    gain = 1.                #Already corrected
                    

                #Obtain P using the flat frame
                if len(old_sf) == 0:
                    print("\n   !!! Calculating new definition of Slit Profile")
                    P = dlh_utils.obtain_P(flat_small,c_all_small,ext_aperture_2,ronoise,gain,NSigma_Marsh, S_Marsh,N_Marsh, Marsh_alg, column_range_small, npools)
                    nans = np.isnan(P)
                    nans_II=np.where(nans == True)
                    if len(nans_II[1]) == 0:
                        hdu = fits.PrimaryHDU( P )
    #                    plt.imshow(P,origin='lower')
    #                    plt.show()
                        hdu.writeto(out_location+"P_"+arm+flat_night+".fits",overwrite=True)
                        old_sf = glob.glob(out_location+"P_"+arm+flat_night+".fits")
                        print("     +++ Writting Slit profile to {}\n".format(old_sf[0]))
                        print("\n   !!! Optimally extracting flat ...")
                        flat_S = dlh_utils.optimal_extraction(flat_small,P,c_all_small,ext_aperture_2,ronoise,gain,S_Marsh,10.*NCosmic_Marsh, column_range_small,npools)
                        
                        #Write the optimal flat extraction to file to be re-read
                        hdu=fits.PrimaryHDU(flat_S)
                        hdu.writeto(out_location+"Flat_extraction_"+arm+flat_night+".fits",overwrite=True)
                        old_flat_extraction=glob.glob(out_location+"Flat_extraction_"+arm+flat_night+".fits")
                    else:
                        print("\n     +++ Failed to define the Slit Function. Exiting.")
                        plt.imshow(P,origin='lower')
                        plt.show()
                        exit()

                else:
                    print("\n      +++ Reading definition of Slit Profile"+old_sf[0])
                    
                    P = fits.getdata(old_sf[0])
                    
                if (Plot == "True"):
                    plt.imshow(P,origin='lower')
                    plt.show()
                
                #Obtain optimally extracted flat
                if len(old_flat_extraction) == 0:
                    print("\n   !!! Optimally extracting flat ...")
                    flat_S = dlh_utils.optimal_extraction(flat_small,P,c_all_small,ext_aperture_2,ronoise,gain,S_Marsh,10.*NCosmic_Marsh, column_range_small,npools)
                    
                    #Write the optimal flat extraction to file to be re-read
                    hdu=fits.PrimaryHDU(flat_S)
                    hdu.writeto(out_location+"Flat_extraction_"+arm+flat_night+".fits",overwrite=True)
                    old_flat_extraction=glob.glob(out_location+"Flat_extraction_"+arm+flat_night+".fits")
                else:
                    print("\n      +++ Reading optimally extracted flat")
                    flat_S = fits.getdata(old_flat_extraction[0])


                print("\n   !!! Optimally extracting "+science_file+" ...")

                #Extract the science data from both fibres
                sci_S  = dlh_utils.optimal_extraction(data_small,P,c_all_small,ext_aperture_2,ronoise,gain,S_Marsh,10.*NCosmic_Marsh, column_range_small,npools)


                print("\n   !!! Correcting for blaze ...")
                for i in range(0,n_ord,2):
                    #tmp = savgol_filter(flat_S[i][1][:],11,3)
#                    plt.plot(tmp)
#                    plt.plot(flat_S[i][1][:])
#                    plt.plot(sci_S[i][1][:])
                    #tmp /=np.max(tmp)
                    
                    sci_S[i][1][:] /= flat_S[i][1][:]/np.max(flat_S[i][1][:])
#                    troughs, _= find_peaks(-sci_S[i][1][:])
#                    sci_S[i][1][:] -= np.median(sci_S[i][1][:][troughs])

#                    plt.plot(sci_S[i][1][:])
#                    plt.show()

                #Add some Keywords to the header for info.
                DATE_EXT=str(datetime.now(tz=pytz.UTC).strftime("%Y-%m-%d"))
                UTC_EXT = str(datetime.now(tz=pytz.UTC).strftime("%H:%M:%S.%f"))
                hdr['DATE-EXT'] = (DATE_EXT,'Date spectrum extracted')
                hdr['UTC-EXT'] = (UTC_EXT,'UTC spectrum extracted')
                hdr['NSig_Mar'] = (NSigma_Marsh,'Sigma for Marsh Detection')
                hdr['N_Marsh'] = (N_Marsh, 'Order of Marsh Polynomial')
                hdr['S_Marsh'] = (S_Marsh,'Separation of Marsh Polynomials')
                hdr['Mar_alg'] = (Marsh_alg, '0 for March curved, 1 for Horne')
                hdr['Cosm_cut'] = (NCosmic_Marsh*100.,'Cosmic ray rejection cutoff')
                hdr['Mar_gain'] = (gain, 'Gain value used (photon/ADU)')
                hdr['Mar_RON'] = (ronoise,'Readout noise used (e-)')
                hdr['Num_Ords'] = (n_ord,'Number of orders extracted')

                #Create new FITS file for the data.
                hdu = fits.PrimaryHDU(data=sci_S,header=hdr)
                file_out=str(out_location_sci+"./HRS_E_"+science_file.removeprefix(out_location_sci))
                hdu.writeto(file_out,overwrite=True)
                
                
                print("Finished")
            else:
                print("File ",science_file, " has been processed here,",file_out, " skipping...")
                
    files_processed = sorted(glob.glob(out_location_sci+"HRS_E_bog*.fits"))
    print("Processing complete, ", len(files_processed)," files processed")
    
    
#Step 8 -- wavelength calibration
#Do this using the "files_processed" list. The upper P fibre has the arc [odd numbers], the lower O fibre has the object

    print("\n   !!! Calculating Wavelength Solution ...")

    if arm == 'H':
        #Load the results of the wavelength calibration initial step
        linelist = np.load("/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/DLH_Codes_combined/TEST_Wave_Sol/hrs_hs.H.linelist.npz",allow_pickle=True)
        nord=42
        ncol=2047
    if arm == 'R':
        linelist = LineList.load("/Users/daniel/Documents/Work/SALT_Pipeline/PyReduce-HRS/datasets/HRS/reduced/hrs_hs.R.linelist.npz")

    for file in files_processed:
    
        #Check to see if the wavelength file has been created
        wavelength_file = sorted(glob.glob(out_location_sci+"HRS_E_W"+file[-22:]))
        if len(wavelength_file) != 1:
    
            hdu=fits.open(file)
            header = hdu[0].header
            hdu.close
            
            #Firstly, process the Reference Fibre image so that can be the starting point for the wavelength solution for the science images

            if (header["I2STAGE"] == 'Reference Fibre' and header["FIFPORT"] == mode):
                degree=[7,11]
                #Calculate the pixel offset between the linelist and the daily reference file
                
                
                offset = dlh_determine_pixel_offset.execute(file,linelist,nord,ncol)
                bc_wave, O_Fibre, wave_rest, P_Fibre = dlh_wavecal.execute(file, arm, "Arc",Plot,degree,offset)
            
            #If we are trying to process the directory again, we may my have already done the refernce file so we need to check and read the results
            
            try:
                wave_rest
            except:
                print("No reference arc processed this run, so read from file")
                for file1 in files_processed:
                    wavelength_files = sorted(glob.glob(out_location_sci+"HRS_E_W"+file1[-22:]))
                    hdu_wave=fits.open(wavelength_files[0])
                    header_wave = hdu_wave[0].header
                    if (header_wave[["I2STAGE"] == 'Reference Fibre' and header["FIFPORT"] == mode]):
                        wave_rest = hdu_wave[0].data[0]
                    hdu_wave.close
                    break
            
            if (header["OBSTYPE"] == sci_frame and header["FIFPORT"] == mode):
            
                #Open the data file
                hdu=fits.open(file)
                data=hdu[0].data
                hdr=hdu[0].header
                hdu.close
                
                n_ord = int(data.shape[0]/2)
                
                P_Fibre = np.zeros((n_ord,data.shape[2]))
                O_Fibre = np.zeros((n_ord,data.shape[2]))
    
                for w in range(0,n_ord):
                    O_Fibre[w] = data[w*2][1]
                    P_Fibre[w] = data[(w*2)+1][1]

#                _, O_Fibre, _, P_Fibre = dlh_wavecal.execute(file, arm, sci_frame,Plot)
                
                
                #Get information to apply BC
                obs_date = hdr["DATE-OBS"]
                ut = hdr["TIME-OBS"]

                if obs_date is not None and ut is not None:
                    obs_date = f"{obs_date}T{ut}"
                    fwmt = hdr["EXP-MID"]
                    et = hdr["EXPTIME"]

                if fwmt > 0.:
                    mid = float(fwmt)/86400.
                else:
                    mid =  float(float(et)/2./86400.)

                jd = Time(obs_date,scale='utc',format='isot').jd + mid

                lat = -32.3722685109
                lon = 20.806403441
                alt = header["SITEELEV"]
    
                object = hdr["OBJECT"]
                BCV =(barycorrpy.get_BC_vel(JDUTC=jd,starname = object, lat=lat, longi=lon, alt=alt, leap_update=True))
                BJD = barycorrpy.JDUTC_to_BJDTDB(jd, starname = object, lat=lat, longi=lon, alt=alt)
                
                #Add BJD to header
                header['BJD'] = (BJD[0][0],'BJD Mid exposure')

                #Apply the BC
                bc_wave = ((wave_rest)*(1.0+(BCV[0]/conts.c)))

            data4=np.array((wave_rest,O_Fibre,bc_wave,P_Fibre))

            hdu_out = fits.PrimaryHDU(data=data4,header=header)
            base_out = file.removeprefix(out_location_sci)
            base_out = base_out[6:]
            file_out=str(out_location_sci+"./HRS_E_W_"+base_out )
            hdu_out.writeto(file_out,overwrite=True)
        
#Step 8 -- Calculate the RV

    print("Calculating RV based on a line mask of a G8 star...")
    
    files_processed = sorted(glob.glob(out_location_sci+"HRS_E_W_bog*.fits"))
    
    for file in files_processed:
        hdu=fits.open(file)
        hdr=hdu[0].header
        
        if (hdr["OBSTYPE"] == sci_frame and hdr["FIFPORT"] == mode):
        
            #try:
           #     hdr['RV']
           # except:
            print("Processing "+file+" ...")
            data=hdu[0].data
            hdu.close
            
            customSimbad = Simbad()
            customSimbad.add_votable_fields('rv_value')
            obj = customSimbad.query_object(hdr["OBJECT"])
            known_rv = obj[0]['RV_VALUE']*1000.#m/s
            rv,rv_err=dlh_RV_calc.execute(hdr,data,known_rv)
            print("DLH",rv, rv_err, known_rv, np.abs(rv-known_rv))
            #Add some Keywords to the header for info.
            hdr['RV'] = (rv,'Weighted RV from multi orders (m/s)')
            hdr['RV_ERR'] = (rv_err,'RV error (m/s)')
            hdr['Cat_RV'] = (known_rv,'RV from SIMBAD catalogue (m/s)')

            #Create new FITS file for the data.
            hdu = fits.PrimaryHDU(data=data,header=hdr)
            file_out=str(file)
            hdu.writeto(file_out,overwrite=True)
        else:
            hdu.close
