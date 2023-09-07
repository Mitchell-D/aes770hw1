"""
Straightforward script for downloading concurrent ABI L1b and L2 data,
and merging them into a common FeatureGrid.
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
import krttdkit.operate.enhance as enh
from krttdkit.operate.geo_helpers import get_geo_range
from krttdkit.operate import abi_recipes

def get_nldas_urls(t0:datetime, tf:datetime):
    """ Downloads the target NLDAS data """
    print(gesdisc.hourly_nldas2_urls(t0=t0, tf=tf))

def get_abi_l2(data_dir:Path, products:list, target_time:datetime, sat="16"):
    """
    Downloads the target ABI data and returns a FeatureGrid of the product's
    characteristic
    """
    l2_prods = [abi.download_l2_abi(data_dir,p,target_time, satellite=sat)[0]
            for p in products]
    data,labels,metas = [],[],{}
    for d,l,m in [abi.get_l2(f) for f in l2_prods]:
        for i in range(len(l)):
            # Skip repeated labels
            if l[i] in labels:
                continue
            labels.append(l[i])
            data.append(d[i])
        # Update the new combined meta-dictionary
        for k in m:
            if k in metas.keys():
                metas[k].append(m[k])
            else:
                metas[k] = [m[k]]
    # return a FeatureGrid
    return FeatureGrid(labels=labels, data=data, meta=metas)

def get_abi_l1b(data_dir:Path, target_time:datetime, sat="16", scan="C"):
    """
    Downloads the target ABI data and returns a FeatureGrid containing
    all reflectance and brightness temperature data, scan angles,
    latitude and longitude, and a invalid value mask.
    """
    # Download the closest file to the target time
    nc_files = ABIL1b.get_l1b(
            data_dir=data_dir,
            satellite=sat,
            scan=scan,
            bands=list(range(1, 17)),
            start_time=target_time,
            replace=False,
            )[0]
    # Load L1b data from the files
    fg = ABIL1b.from_l1b_files(
            path_list=nc_files,
            convert_tb=True,
            convert_ref=True,
            get_latlon=True,
            get_mask=True,
            get_scanangle=True,
            resolution="2",
            )
    return fg



if __name__=="__main__":
    debug = True

    """ Download Settings """
    # Pacific time is UTC-7
    target_time = datetime(2022,10,7,19,27)
    #target_time = datetime(2023,8,7,19,27)
    time_window = timedelta(hours=2)
    # Directory where netCDF files will be downloaded
    nc_dir = Path("data/abi_nc")
    # List of L2 products to download alongside L1b products
    l2_2km_prods = ["ACMC", "ACTPC", "BRFC", "CODC", "CPSC"]
    sat = "17"
    # ABI Scan;
    scan = "C"
    # Pickle written with the full array of raw data
    #pkl_path = Path("data/FG_aes770hw1_all.pkl")
    #pkl_path = Path("data/FG_aes770hw1_l2.pkl")

    """ Subgrid Settings """
    # Center lat/lon of subgrid
    target_latlon = (29,-123)
    # Width and height of subgrid
    dx_px,dy_px = (640,512)

    """ Classification Settings """
    do_classification = False
    # Pickle written with the subgrid containing MLC and MDC masks
    out_pkl = Path("data/FG_aes770hw1_l2-subgrid.pkl")
    # Categories for maximum-likelihood classification
    categories = ["ocean", "sparse_cloud", "dense_cloud", "land"]
    # Bands used for MLC and MDC classification
    class_bands = ["2-ref", "3-ref", "4-ref", "6-ref", "7-rad",
                 "10-tb", "13-tb", "15-tb"]

    """
    Make sure the pkl exists. If not, extract data from the netCDF files.
    If the netCDF files are missing re-download them. Create a new data pkl.
    """
    if not pkl_path.exists():
        """ Get a FeatureGrid for the target time including all ABI bands """
        fgl1b = get_abi_l1b(
                data_dir=nc_dir,
                target_time=target_time,
                sat=sat,
                scan=scan,
                )

        """ Get L2 files for achac (cloud height) and acmc (clear sky mask) """
        fgl2 = get_abi_l2(
                data_dir=nc_dir,
                products=l2_2km_prods,
                target_time=target_time,
                sat=sat,
                )
        """ Read in 2km data and merge then into one FeatureGrid """
        fg = FeatureGrid.merge(fgl1b, fgl2, drop_duplicates=True, debug=debug)
        fg.to_pkl(pkl_path)
    else:
        fg = FeatureGrid.from_pkl(pkl_path)

    """ Get a NLDAS-2 file for the target time """
    #get_nldas_urls(target_time, target_time+timedelta(hours=1))

    """ Get a subgrid according to pixel boundaries provided above """
    latlon = np.dstack((fg.data("lat"), fg.data("lon")))
    yrange,xrange = get_geo_range(
            latlon, target_latlon, dx_px, dy_px,
            from_center=True, boundary_error=True, debug=debug)
    fg = fg.subgrid(vrange=yrange, hrange=xrange)

    """ Use MLC and MDC to get masks for land, ocean, and sparse clouds """
    if do_classification:
        # Add ABI-relevant recipes to the FeatureGrid
        for label,recipe in abi_recipes.items():
            fg.add_recipe(label,recipe)
        # Do maximum-likelihood classification
        class_arr, labels, samples = fg.do_mlc(
                select_recipe="truecolor",
                categories=categories,
                labels=class_bands,
                threshold=None)
        # Add maximum-likelihood results as an integer array
        fg.add_data("mlc_ints", class_arr,
                    info={"samples":samples,"labels":labels})
        # Do minimum-distance classification
        class_arr, labels, samples = fg.do_mdc(
                select_recipe="dcp",
                categories=categories,
                labels=class_bands)
        # Add mdc results as a boolean array for each sample class
        for i in range(len(labels)):
            print(f"adding {labels[i]}")
            fg.add_data(labels[i],class_arr==i,
                        info={"samples":samples[labels[i]]})
    fg.to_pkl(out_pkl)
