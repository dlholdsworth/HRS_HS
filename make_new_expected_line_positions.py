import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
#Good starting file
file="/Users/daniel/Desktop/SALT_HRS_DATA/Blu/2023/0312/reduced/HRS_E_W_bogH202303120016.fits"

data=fits.getdata(file)

wave=data[0]
flux=data[1]

nord=wave.shape[0]

#line list
list_file = "thar_list_new.txt"

ThAr_vac_wave = np.loadtxt(list_file,usecols=1)

out_pixel = []
out_wave = []
out_ord = []
for ord in range(nord):
    peaks = find_peaks(flux[ord],width=1,distance=10,height=200,prominence=400)
#    print(peaks)
#    plt.plot(flux[ord])
#    plt.plot(peaks[0],flux[ord][peaks[0]],'x')
#    plt.show()

#    plt.plot(wave[ord],flux[ord])
#    plt.plot(wave[ord][peaks[0]],flux[ord][peaks[0]],'x')
#    plt.vlines(ThAr_vac_wave,0,1000,'r')
    wave_of_peaks=wave[ord][peaks[0]]
    pixel_of_peaks = peaks[0]
    for i in range(len(wave_of_peaks)):
        line_dict = {}
        thar_line=find_nearest(ThAr_vac_wave, value=wave_of_peaks[i])
        if (np.logical_and(np.abs(thar_line-wave_of_peaks[i]) < 0.2, int(pixel_of_peaks[i]) <2035)):
#            plt.vlines(thar_line,0,10000,'g')
            out_ord.append(int(ord))
            out_pixel.append(int(pixel_of_peaks[i]))
            out_wave.append(thar_line)
#            line_dict['order'] = ord
#            line_dict['pixel'] = pixel_of_peaks[i]
#            line_dict['wave'] = thar_line
#            lines_dict = line_dict
#            print(pixel_of_peaks[i],thar_line)
#    plt.show()
    
#    output_dict[ord] = lines_dict
    
#ii=np.where(output_dict['order'] == 3)[0]
#print(output_dict[2]['pixel'],output_dict[2]['wave'])
data_out=np.array([(out_ord),(out_pixel),out_wave])
Filename='Good_thar_pix_locations.txt'
np.savetxt(Filename, data_out.T, header="Order Pixel Wavelength(vac)", delimiter= " ",fmt='%.0f %.0f %.4f ')

