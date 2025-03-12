#This does level zero corrections (overscan removal, gain correction) and flips the Red data so orders run bottom to top blue to red (as with the Blue arm data)

from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import re
import numpy as np



def execute(raw_files,data_dir,arm,out_location):
   
    corr_files = []

    for file in raw_files:
        #Check if a corrected file has been created.
        new_set = glob.glob(str(out_location+"og"+file.removeprefix(data_dir)))
        if len(new_set) == 1:
            corr_files.append(new_set[0])
        else:
            hdu=fits.open(file)
            
            
            #Make sure the FIFPORT is set correctly.
            if (hdu[0].header["OBSMODE"] == "HIGH STABILITY"):
                fits.setval(file, 'FIFPORT', value='HS')
                hdu[0].header["FIFPORT"] = 'HS'
            #Check if Bias frame -- force FIFPORT to HS -- will need to think of a better way if extending this code to other modes.
            if (hdu[0].header["OBSTYPE"] == "Bias"):
                fits.setval(file, 'FIFPORT', value='HS')
                hdu[0].header["FIFPORT"] = 'HS'
        
            namps=hdu[0].header["NAMPS"]
            
            if arm == "H":
                if namps == 1:
                    overscan_region = hdu[0].header["BIASSEC"]
                    data_region = hdu[0].header["DATASEC"].strip()
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    xy=re.split(regex_pattern, data_region)
                    x1=int(xy[1])
                    x2=int(xy[2])
                    y1=int(xy[3])
                    y2=int(xy[4])
                    
                    trimmed = hdu[0].data[y1:y2,x1:x2]
                    
                    #Correct for the gain
                    gain = hdu[0].header["GAIN"]
                    gain1 = float(gain.split()[0])
                    gain2 = float(gain.split()[1])
                    trimmed = trimmed*gain1
                    
                    hdu[0].data = np.float32(trimmed)
                    file_out=str(out_location+"og"+file.removeprefix(data_dir))
                    #Need to add some comments to header for the files used to correct image.
                    hdu.writeto(file_out,overwrite=True)
                    
                if namps != 1:
                    print("Amplifier not yet supported.")
                    break
                
                corr_files.append(file_out)
                    
            elif arm == "R":
                if namps == 1:
                    #Remove the overscan region
                    overscan_region = hdu[0].header["BIASSEC"]
                    data_region = hdu[0].header["DATASEC"].strip()
                    delimiters = ":", ",", "[","]"
                    regex_pattern = '|'.join(map(re.escape, delimiters))
                    xy=re.split(regex_pattern, data_region)
                    x1=int(xy[1])
                    x2=int(xy[2])
                    y1=int(xy[3])
                    y2=int(xy[4])
                    
                    trimmed = hdu[0].data[y1:y2,x1:x2]
                    
                    #Correct for the gain
                    gain = hdu[0].header["GAIN"]
                    gain1 = float(gain.split()[0])
                    gain2 = float(gain.split()[1])
                    gain3 = float(gain.split()[2])
                    gain4 = float(gain.split()[3])
                    
                    trimmed = trimmed*gain1
                    
                    #Now flip the image
                    
                    trimmed = trimmed[::-1,::]
                    
                    hdu[0].data = np.float32(trimmed)
                    file_out=str(out_location+"og"+file.removeprefix(data_dir))
                    #Need to add some comments to header for the files used to correct image.
                    hdu.writeto(file_out,overwrite=True)
                    
                if namps != 1:
                    print("Amplifier not yet supported.")
                    break
                
                #Correct for the gain
                
                corr_files.append(file_out)
            
            else:
                print("Arm unknown! Given "+str(arm)+" expected H or R. Exiting")
                exit()
            hdu.close

    return corr_files
