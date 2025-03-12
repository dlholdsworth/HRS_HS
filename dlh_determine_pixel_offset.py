import numpy as np
from astropy.io import fits
from lmfit import  Model,Parameter
import matplotlib.pyplot as plt

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return amp*np.exp(-(x-cen)**2/(2*wid**2))
    
#A very simple 'ccf' to determine the offset between the reference line list and the current daily reference spectrum.

def execute(file,linelist,nord,ncol):

    lines=linelist['cs_lines']

    data=fits.getdata(file)
    all_diff = []
    
    P_Fibre = np.zeros((nord,ncol))
    O_Fibre = np.zeros((nord,ncol))

    for w in range(0,nord):
        O_Fibre[w] = data[w*2][1]
        P_Fibre[w] = data[(w*2)+1][1]
    
    for order in range(nord):
        spectrum = O_Fibre[order]/np.nanmax(O_Fibre[order])
        ii=np.where(lines['order'] == order)[0]
        line_center =(lines['posc'][ii])
        line_weight = lines['height'][ii]
       
        pixel_loop = np.arange(-20,20,1)
        x_pixel_wave = np.arange(len(spectrum))


        diff_ord = []
        for step in pixel_loop:
            diff = []
            for i in range(len(ii)):
                if np.logical_and((lines['posc'][ii[i]].astype('int')+step) > 0, (lines['posc'][ii[i]].astype('int')+step < len(spectrum))):
                    diff.append(spectrum[lines['posc'][ii[i]].astype('int')+step]-line_weight[i])
            diff = np.median(np.array(diff))
            diff_ord.append(diff)
            
        #Fit a gaussian to the peak offset
        gmod = Model(gaussian)
        

        cen = pixel_loop[np.where(diff_ord == np.max(diff_ord))[0]]
        if len(cen) > 0:
            cen = cen[0]
            result_ref = gmod.fit((diff_ord-np.median(diff_ord)), x=pixel_loop, amp=Parameter('amp',value=1),cen=Parameter('cen',value=cen,min=pixel_loop[0],max=pixel_loop[-1]),wid=Parameter('wid',value=2.,min=0.5,max=1.5))
        
            all_diff.append(result_ref.params['cen'].value)

#           plt.plot(order,result_ref.params['cen'].value,'o')

    all_diff = np.array(all_diff)
#    plt.show()

    return np.median(all_diff)

