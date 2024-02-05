import numpy as np
import math as m
from pathlib import Path
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import pickle as pkl

from krttdkit.acquire import get_goes as gg
from krttdkit.acquire import abi
from krttdkit.acquire import gesdisc
from krttdkit.products import FeatureGrid
from krttdkit.products import ABIL1b
import krttdkit.visualize.guitools as gt
import krttdkit.visualize.geoplot as gp
import krttdkit.visualize.TextFormat as TF
import krttdkit.operate.enhance as enh
from krttdkit.operate.geo_helpers import get_geo_range
from krttdkit.operate import abi_recipes

def plot_scatter(px, py, ps, tau_lines:list, cre_lines:list,
              tau_labels:list, cre_labels:list, phi, vza, sza,
                 fig_path=None):
    fig, ax = plt.subplots()
    #ax.set(aspect=1)

    for i in range(len(tau_lines)):
        ax.plot(*zip(*tau_lines[i]), color="black", linewidth=.5)
        if not i%2:
            anno_loc = list(tau_lines[i][-1])
            anno_loc[1] -= .02
            #tau_labels = [ f"{l:.2f}" for l in tau_labels]
            tau_labels = [int(10*l)/10 for l in tau_labels]
            ax.annotate(tau_labels[i], xytext=anno_loc,
                        xy=tau_lines[i][-1], )

    for i in range(len(cre_lines)):
        ax.plot(*zip(*cre_lines[i]), color="black", linewidth=.5)
        if not i%2:
            anno_loc = list(cre_lines[i][-1])
            #anno_loc[0] += .02
            #cre_labels = [ f"{l:.2f}" for l in cre_labels]
            cre_labels = [int(10*l)/10 for l in cre_labels]
            ax.annotate(cre_labels[i], xytext=anno_loc,
                        xy=cre_lines[i][-1], )
    #im = ax.pcolormesh(xcoords, ycoords, image, norm="log", cmap="nipy_spectral")
    im = ax.scatter(px, py, s=ps)
    plt.xlabel("ABI Band 2 ($0.64\mu m$) Reflectance")
    plt.ylabel("ABI Band 6 ($2.24 \mu m$) Reflectance")
    #plt.ylabel("ABI Band 7 ($3.9 \mu m$) Reflectance")
    plt.title("Cloud reflectances wrt COD and CRE; " + \
            f"PHI={phi}, VZA={vza}, SZA={sza}")
    if fig_path:
        plt.savefig(fig_path.as_posix())
    plt.show()

def plot_mesh(image, xcoords, ycoords, tau_lines:list, cre_lines:list,
              tau_labels:list, cre_labels:list):
    """ """
    fig, ax = plt.subplots()
    ax.set(aspect=1)

    for i in range(len(tau_lines)):
        ax.plot(*zip(*tau_lines[i]), color="black", linewidth=.5)
        if not i%2:
            anno_loc = list(tau_lines[i][-1])
            anno_loc[1] -= .02
            #tau_labels = [ f"{l:.2f}" for l in tau_labels]
            tau_labels = [int(100*l)/100 for l in tau_labels]
            ax.annotate(tau_labels[i], xytext=anno_loc,
                        xy=tau_lines[i][-1], )

    for i in range(len(cre_lines)):
        ax.plot(*zip(*cre_lines[i]), color="black", linewidth=.5)
        if not i%2:
            anno_loc = list(cre_lines[i][-1])
            #anno_loc[0] += .02
            #cre_labels = [ f"{l:.2f}" for l in cre_labels]
            cre_labels = [int(100*l)/100 for l in cre_labels]
            ax.annotate(cre_labels[i], xytext=anno_loc,
                        xy=cre_lines[i][-1], )
    im = ax.pcolormesh(xcoords, ycoords, image, cmap="nipy_spectral")
    plt.xlabel("ABI Band 2 ($0.64\mu m$) Reflectance")
    plt.ylabel("ABI Band 6 ($2.24 \mu m$) Reflectance")
    plt.title("Dense cloud reflectances wrt COD and CRE; PHI=142, VZA=4.5, sza=36")
    plt.show()

if __name__=="__main__":
    pkl_path = Path("data/FG_subgrid_aes770hw1.pkl")
    #lu2,lu2_args = pkl.load(Path("data/lut_rad_ABI2.pkl").open("rb"))
    #lu6,lu6_args = pkl.load(Path("data/lut_rad_ABI6.pkl").open("rb"))
    lu2,lu2_args = pkl.load(Path("data/lut_rad_ABI2.pkl").open("rb"))
    lu6,lu6_args = pkl.load(Path("data/lut_rad_ABI6.pkl").open("rb"))

    bispec_fig = Path("figures/bispec_deepcloud.png")
    domain_fig = Path("figures/bispec_deepcloud_domain.png")

    """ Load the FeatureGrid and add ABI recipes to it """
    fg = FeatureGrid.from_pkl(pkl_path)
    for label,recipe in abi_recipes.items():
        fg.add_recipe(label,recipe)

    """ Ask the user to choose a rectangle of pixels """
    vrange, hrange= gt.region_select(fg.data("truecolor"))
    vidx, hidx = map(np.asarray, zip(
        *[(j,i) for i in range(*hrange) for j in range(*vrange)]))

    """ Make a truecolor with the selected region"""
    rect_rgb = gt.rect_on_rgb(fg.data("norm256 truecolor"), vrange, hrange)
    gt.quick_render(rect_rgb)
    #gp.generate_raw_image(rect_rgb, domain_fig)

    """ Get a reduced lookup table using angle averages """
    phi_idx = np.argmin(abs(lu2_args["phis"]-np.average(
        fg.data("raa")[vidx, hidx])))
    uzen_idx = np.argmin(abs(lu2_args["uzens"]-np.average(
        fg.data("vza")[vidx, hidx])))
    sza_idx = np.argmin(abs(lu2_args["szas"]-np.average(
        fg.data("sza")[vidx, hidx])))
    lut_lines = np.stack((lu2,lu6),axis=0)[:,sza_idx,:,:,phi_idx,uzen_idx]
    #print(phi_idx, uzen_idx, sza_idx)
    #exit(0)

    """ Subset the reflectance arrays by to selected region """
    b2 = fg.data("2-ref")[vidx, hidx]
    b6 = fg.data("6-ref")[vidx, hidx]

    #from get_retrieval import atmospheric_reflectance, rayleigh_tau
    #ar = atmospheric_reflectance(fg, rayleigh_tau(fg))

    """ Load model grid lines from the lookup table """
    kappa0 = np.array([0.0019486,0.0415484])
    lut_lines[0] *= kappa0[0]
    lut_lines[1] *= kappa0[1]
    tau_lines, cre_lines = [], []
    for i in range(lut_lines.shape[1]):
        tau_lines.append([])
        for j in range(lut_lines.shape[2]):
            tau_lines[i].append(tuple(lut_lines[:,i,j]))
    for i in range(lut_lines.shape[2]):
        cre_lines.append([])
        for j in range(lut_lines.shape[1]):
            cre_lines[i].append(tuple(lut_lines[:,j,i]))

    """ Get a histogram of pixel values """
    hist, coords = enh.get_nd_hist((b2,b6), 128)

    # Mesh grid (good for big rectangles)
    #plot_mesh(hist.T, *coords, tau_lines, cre_lines,
    #          lu2_args["taus"], lu2_args["cres"])

    # Scatter plot (good for small rectangles)
    idx_y, idx_x = np.where(hist != 0)
    py = coords[0][idx_y]
    px = coords[1][idx_x]
    ps = hist[idx_y,idx_x]*2

    #plot_scatter([], [], [], tau_lines, cre_lines,
    plot_scatter(py, px, ps, tau_lines, cre_lines,
                 lu2_args["taus"], lu2_args["cres"],
                 phi=lu2_args["phis"][phi_idx],
                 vza=lu2_args["uzens"][uzen_idx],
                 sza=lu2_args["szas"][sza_idx],
                 fig_path=bispec_fig,
                 )

    exit(0)
    """
    Plot the bispectral diagram alone (without obs) of 3.9um channel
    """
    lu7,lu7_args = pkl.load(Path("data/lut_rad_ABI7.pkl").open("rb"))
    lut_lines = np.stack((lu2,lu7),axis=0)[:,sza_idx,:,:,phi_idx,uzen_idx]

    """ Load model grid lines from the lookup table """
    tau_lines, cre_lines = [], []
    for i in range(lut_lines.shape[1]):
        tau_lines.append([])
        for j in range(lut_lines.shape[2]):
            tau_lines[i].append(tuple(lut_lines[:,i,j]))
    for i in range(lut_lines.shape[2]):
        cre_lines.append([])
        for j in range(lut_lines.shape[1]):
            cre_lines[i].append(tuple(lut_lines[:,j,i]))

    #idx_y, idx_x = np.where(hist != 0)
    plot_scatter([], [], [], tau_lines, cre_lines,
                 lu2_args["taus"], lu2_args["cres"],
                 phi=lu2_args["phis"][phi_idx],
                 vza=lu2_args["uzens"][uzen_idx],
                 sza=lu2_args["szas"][sza_idx],
                 )
