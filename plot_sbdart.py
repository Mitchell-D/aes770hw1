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

def plot_mesh(image, xcoords, ycoords, tau_lines:list, cre_lines:list,
              tau_labels:list, cre_labels:list):
    fig, ax = plt.subplots()
    ax.set(aspect=1)

    for i in range(len(tau_lines)):
        ax.plot(*zip(*tau_lines[i]), color="black", linewidth=.5)
        anno_loc = list(tau_lines[i][-1])
        anno_loc[1] -= .02
        ax.annotate(tau_labels[i], xytext=anno_loc, xy=tau_lines[i][-1], )
    for i in range(len(cre_lines)):
        ax.plot(*zip(*cre_lines[i]), color="black", linewidth=.5)
        anno_loc = list(cre_lines[i][-1])
        anno_loc[0] += .02
        ax.annotate(cre_labels[i], xytext=anno_loc, xy=cre_lines[i][-1], )
    im = ax.pcolormesh(xcoords, ycoords, image, norm="log", cmap="nipy_spectral")
    plt.show()

if __name__=="__main__":
    pkl_path = Path("data/FG_subgrid_aes770hw1.pkl")
    lut_path = Path("data/lut_15.pkl")

    """ Load the FeatureGrid and add ABI recipes to it """
    fg = FeatureGrid.from_pkl(pkl_path)
    for label,recipe in abi_recipes.items():
        fg.add_recipe(label,recipe)

    kappa0 = np.array([0.0019486,0.0415484])
    lut,wls,taus,cres,phi,uzen = pkl.load(lut_path.open("rb"))
    lut_lines = lut[:,:,:,10,10]

    lut[0] *= kappa0[0]
    lut[1] *= kappa0[1]
    tau_lines, cre_lines = [], []
    for i in range(lut_lines.shape[1]):
        tau_lines.append([])
        for j in range(lut_lines.shape[2]):
            tau_lines[i].append(tuple(lut_lines[:,i,j]))
    for i in range(lut_lines.shape[2]):
        cre_lines.append([])
        for j in range(lut_lines.shape[1]):
            cre_lines[i].append(tuple(lut_lines[:,j,i]))

    notcloud = np.logical_not(fg.data("dense_cloud"))
    band2 = fg.data("2-ref", mask=notcloud)
    band6 = fg.data("6-ref", mask=notcloud)
    print(np.average(fg.data("2-ref")[fg.data("dense_cloud")]))
    print(np.average(fg.data("6-ref")[fg.data("dense_cloud")]))
    hist, coords = enh.get_nd_hist((fg.data("2-ref")[fg.data("dense_cloud")],
                                    fg.data("6-ref")[fg.data("dense_cloud")]))
    plot_mesh(hist.T, *coords, tau_lines, cre_lines, taus, cres)
