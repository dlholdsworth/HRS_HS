import numpy as np
import matplotlib.pyplot as plt
import dlh_utils

def execute(image,sigma_cutoff,scatter_degree,bw,n_ord,ncol,nrow,ext_aperture,c_all,column_range,Plot):

    mask_bkg = np.full(image.shape, True)
    if bw is not None and bw != 0:
        mask_bkg[:bw] = mask_bkg[-bw:] = mask_bkg[:, :bw] = mask_bkg[:, -bw:] = False
    for i in range(n_ord):
        left, right = column_range[i]
        left -= ext_aperture[i][1] * 2
        right += ext_aperture[i][0] * 2
        left = max(0, left)
        right = min(ncol, right)

        x_order = np.arange(left, right)
        y_order = np.polyval(c_all[i], x_order)

        y_above = y_order + ext_aperture[i][1]+1
        y_below = y_order - ext_aperture[i][0]-1

        y_above = np.floor(y_above)
        y_below = np.ceil(y_below)

        index = dlh_utils.make_index(y_below, y_above, left, right, zero=True)
        np.clip(index[0], 0, nrow - 1, out=index[0])

        mask_bkg[index] = False

    if Plot == "True":
        plt.imshow(mask_bkg*image,origin='lower',vmin=0,vmax=100)
        plt.show()

    mask_bkg &= ~np.ma.getmask(image)

    y, x = np.indices(mask_bkg.shape)
    y, x = y[mask_bkg].ravel(), x[mask_bkg].ravel()
    z = np.ma.getdata(image[mask_bkg]).ravel()

    mask_bkg = z <= np.median(z) + sigma_cutoff * z.std()
    y, x, z = y[mask_bkg], x[mask_bkg], z[mask_bkg]

    coeff = dlh_utils.polyfit2d(x, y, z, degree=scatter_degree, scale=True, plot=1, plot_title="none")

    # Calculate scatter at interorder positionsq
    yp, xp = np.indices(image.shape)
    back = np.polynomial.polynomial.polyval2d(xp, yp, coeff)

    if Plot =="True":
        plt.subplot(121)
        plt.title("Input Image + In-between Order traces")
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        vmin, vmax = np.percentile(image - back, (5, 95))
        plt.imshow(image - back, vmin=vmin, vmax=vmax, aspect="equal", origin="lower")
        #plt.plot(x, y, ",")

        plt.subplot(122)
        plt.title("2D fit to the scatter between orders")
        plt.xlabel("x [pixel]")
        plt.ylabel("y [pixel]")
        plt.imshow(back, vmin=0, vmax=abs(np.max(back)), aspect="equal", origin="lower")

        plt.show()

    return back
