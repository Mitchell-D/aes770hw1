"""
Script for downloading ABI files within a target range.
"""
from pathlib import Path
from datetime import datetime
import pickle as pkl
import multiprocessing as mp
import numpy as np

from krttdkit.acquire import abi
from krttdkit.acquire.get_goes import parse_goes_path
from krttdkit.products import FeatureGrid, ABIL1b
from krttdkit.operate.geo_helpers import get_geo_range
from krttdkit.operate.recipe_book import abi_recipes
from krttdkit.visualize import TextFormat as TF
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp

def _mp_generate_images(args:tuple):
    """
    args is a 2-tuple like (recipes, abil1b, image_path, index)
    if image_path has a curly brace variable {recipe}, or {index}
    it will be formatted with the string recipe label
    """
    recipes, abil1b, image_path, index = args
    return None

def _mp_timestep_to_abil1b(args:dict):
    """
    args is a dictionary matching the keyword arguments of timestep_to_abil1b
    """
    return files_to_abil1b(**args)

def files_to_abil1b(
        file_paths:list, pkl_path:Path=None, resolution=2,
        target_latlon:tuple=None, dx_px:int=None, dy_px:int=None,
        from_center:bool=False, boundary_error:bool=True, debug=False):
    """
    Parses a list of unique ABI L1b netCDF files, all of which have the same
    start time, into an ABIL1b object, optionally serialized as a pkl.
    """
    fg = ABIL1b.from_l1b_files(
            path_list=file_paths,
            convert_tb=True,
            convert_ref=True,
            get_latlon=True,
            get_mask=True,
            resolution=resolution,
            )
    if any(not a is None for a in (target_latlon, dx_px, dy_px)):
        if not all(not a is None for a in (target_latlon, dx_px, dy_px)):
            raise ValueError(
                    f"target_latlon, dx_px, and dy_px must be given together")
        latlon = np.dstack((fg.data("lat"), fg.data("lon")))
        yrange,xrange = get_geo_range(latlon, target_latlon, dx_px, dy_px,
                                      from_center, boundary_error, debug)
        fg = fg.subgrid(vrange=yrange, hrange=xrange)
    fg.to_pkl(pkl_path, overwrite=overwrite)
    return fg

def clear_buf_pkls(buf_dir:Path, buf_key:str, debug:bool=False):
    """
    Deletes all pkl files in buf_dir that contain the substring buf_key

    :@param buf_dir: Pickle buffer directory
    :@param buf_key: Substring of all pkl files in buf_dir that are removed.
    """
    for p in buf_dir.iterdir():
        if p.suffix==".pkl" and buf_key in p.name:
            if debug: print(f"Removing buffer pkl {p.as_posix()}")
            p.unlink()

def load_buffer(buf_dir:Path, buf_key:str, debug:bool=False):
    """
    Loads all pkl files in buf_dir that contain the substring buf_key

    :@param buf_dir: Pickle buffer directory
    :@param buf_key: Substring of all pkl files in buf_dir that are loaded.

    :@return: List of ABIL1b objects ordered by stime.
    """
    datafiles = []
    for p in buf_dir.iterdir():
        if p.suffix==".pkl" and buf_key in p.name:
            if debug: print(f"Loading buffer pkl {p.as_posix()}")
            datafiles.append(p)
    return (FeatureGrid.from_pkl(p) for p in sorted(datafiles))

if __name__=="__main__":
    debug = False
    data_dir = Path("data/abi")
    start = datetime(2023,8,7,18)
    end = datetime(2023,8,7,19)
    bands = list(range(1, 17))
    satellite = "18"
    scan = "C"
    buf_dir = Path("data/buffer")
    #gif_dir = Path("figures/mp4_factory")
    gif_dir = Path("figures/video_factory")
    buf_key = "tmp"
    overwrite = True
    resolution=1 # km
    workers=4
    target_latlon = (35.681, -122.3761)
    dx_px = 640*2
    dy_px = 512*2
    #clear_buf_pkls(buf_dir, "tmp2", debug=debug)

    """ List of recipes to generate animations of """
    animation_recipes = (
            "lingamma truecolor",
            "histeq airmass",
            "histeq diffwv",
            "histeq wv"
            )

    """
    Download files for all requested bands in range of the time selection
    """
    if overwrite:
        clear_buf_pkls(buf_dir, buf_key, debug=debug)
        # Download any files in range that aren't already downloaded.
        data_paths = ABIL1b.get_l1b(
                data_dir=data_dir,
                satellite=satellite,
                scan=scan,
                bands=bands,
                start_time=start,
                end_time=end,
                replace=False,
                )
        # Establish arguments for parsing the files
        args = [{
            "file_paths":data_paths[t],
            "pkl_path":buf_dir.joinpath(f"{buf_key}_{t}.pkl"),
            "resolution":resolution,
            "target_latlon":target_latlon,
            "dx_px":dx_px,
            "dy_px":dy_px,
            "from_center":True,
            "boundary_error":False,
            "debug":debug,
            } for t in range(len(data_paths))]
        with mp.Pool(workers) as pool:
            data = pool.map(_mp_timestep_to_abil1b, args)

    """
    Load each FeatureGrid in the time series from the buffer and generate
    images from
    """
    image_paths = []
    for fg in load_buffer(buf_dir, buf_key):
        for label, recipe in abi_recipes.items():
            fg.add_recipe(label, recipe)
        for recipe in animation_recipes:
            glob_str = fg._info[0]["stime"]
            new_path = gif_dir.joinpath(
                    f"{recipe.replace(' ','-')}_{glob_str}.png")
            image_paths.append(gp.generate_raw_image(fg.data(recipe), new_path))
    print([p.as_posix() for p in image_paths])
