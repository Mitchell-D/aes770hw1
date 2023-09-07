
import netCDF4 as nc
import numpy as np
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from krttdkit.acquire import abi
from krttdkit.products import FeatureGrid
from krttdkit.visualize import guitools as gt

data, labels, meta = abi.get_l2(abi.download_l2_abi(
    Path("/tmp"), "ACHAC", datetime(2022,10,7,19,27), satellite="17")[0])
fg = FeatureGrid(labels, data, meta=meta)
#print(fg.data("height"))
height = fg.data("height", mask=fg.data("height")>20000)
gt.quick_render(height)
print(fg.shape)
print(fg.get_pixels("colorize height", labels=["height"]))
#gt.quick_render(fg.data("colorize height"))
