"""
Script for generating heatmaps of arrays from boolean masks of classes
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from datetime import timedelta

from krttdkit.acquire import get_goes as gg
from krttdkit.acquire import abi
from krttdkit.acquire import gesdisc
from krttdkit.products import FeatureGrid
from krttdkit.products import ABIL1b
import krttdkit.visualize.guitools as gt
import krttdkit.visualize.TextFormat as TF
import krttdkit.operate.enhance as enh
from krttdkit.operate.geo_helpers import get_geo_range
from krttdkit.operate import abi_recipes

""" Settings """
## List of labels of boolean masks marking each class as True in fg
categories = ["ocean", "sparse_cloud", "dense_cloud", "land"]
## 2-tuple (ylabel, xlabel) of bands to compare in the heatmap
#hm_labels = ("13-tb", "2-ref")
hm_labels = ("6-ref", "2-ref")
## 2-tuple (ybins, xbins) of brightness levels to bin each axis into
hm_bin_counts = (128,128)
## Path to a pkl containing the above category masks
pkl_path = Path("data/FG_aes770hw1_masked_2.pkl")
## Title format
title_fmt = "{px_type} ({wl1}$\mu m$ vs {wl2}$\mu m$)"
## Path to figure; {mask_type} field is replaced with mask name
fig_path_fmt = "figures/heatmap/heatmap_b2b6_{mask_type}.png"
## Optional (ymin,ymax,xmin,xmax) in data coordinates, or None
#data_ranges = [(240,330),(0,.8)]
data_ranges = [(0,.8),(0,.8)]
#data_ranges = None
## If True, shows each anti-mask designating classes
review_masks = False
## If True, shows heatmap figures before generating
show = True

plot_spec={
    "cmap":"nipy_spectral",
    #"ylabel":"ABI Band 13 Brightness Temp ($10.3\mu m$)",
    "ylabel":"ABI Band 6 Reflectance ($2.2\mu m$)",
    "xlabel":"ABI Band 2 Reflectance ($0.64\mu m$)",
    "cb_label":"Count (log scale)",
    "cb_orient":"horizontal",
    "cb_size":.4,
    #"imshow_aspect":1/200,
    "imshow_aspect":1,
    #"imshow_extent":(0,1,0,1),
    "dpi":200,
    "vmax":None
    }


""" Load the FeatureGrid and add ABI recipes to it """
fg = FeatureGrid.from_pkl(pkl_path)
for label,recipe in abi_recipes.items():
    fg.add_recipe(label,recipe)
""" View each original array """
[gt.quick_render(fg.data(l)) for l in hm_labels]
""" Get the center wavelength of each axis for labels """
wls = [fg.info(l)["ctr_wl"] for l in hm_labels]
""" Dictionary of masks in the featuregrid named like each category above """
anti_masks = {k.replace("_","-"):np.logical_not(fg.data(k))
              for k in categories}

if review_masks:
    for k in anti_masks:
        print(TF.BLUE("Showing Anti-Mask: ",bright=True),TF.WHITE(k,bold=True))
        gt.quick_render(fg.data("truecolor", mask=anti_masks[k]))
"""
Get a heatmap of reflectance in band 2 (.64um) and 6 (2.24um) over clouds
"""
assert len(hm_labels)==2
for am in anti_masks.keys():
    plot_spec["title"] = title_fmt.format(
            px_type=am, wl1=wls[0], wl2=wls[1])
    fg.heatmap(
            *[fg.data(l, mask=anti_masks[am], mask_value=np.nan)
              for l in hm_labels],
            nbin1=hm_bin_counts[0],
            nbin2=hm_bin_counts[0],
            show=show,
            fig_path=Path(fig_path_fmt.format(
                mask_type=am,)),
            ranges=data_ranges,
            plot_spec=plot_spec,
            )
