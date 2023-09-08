"""
Script for generating lookup tables for layer albedo at a given wavelength.
The returned table is shaped like: (sza, tau)
"""
#from sbdart_info import default_params
from pathlib import Path
from krttdkit.acquire.sbdart import dispatch_sbdart, parse_iout
import krttdkit.visualize.guitools as gt
import numpy as np
import pickle as pkl

def make_alb_lut(tmp_dir:Path, sbdart_args:dict, wl, taus, szas, cre=10,
                 print_stdout=False):
    """
    Make a lookup table for the albedo between zout1 and zout2, as specified
    in sbdart_args at a given wavelength, ASSUMING CONSERVATIVE SCATTERING.

    The returned lookup table is shaped like (szas, taus)
    """
    tmp_sza = []
    for s in szas:
        tmp_tau = []
        for t in taus:
            new_args = {"wlinf":wl, "wlsup":wl, "wlinc":0.,
                        "sza":s, "tcloud":t, "nre":cre, "iout":1}
            sbdart_args.update(new_args)
            sb_out=dispatch_sbdart(sbdart_args, tmp_dir)
            out = parse_iout(iout_id=1, sb_out=sb_out,
                             print_stdout=print_stdout)["sflux"]
            # Assume conservative scattering
            tmp_tau.append(1-out[5]/out[2])
        tmp_sza.append(tmp_tau)
    return np.asarray(tmp_sza)

if __name__=="__main__":
    tmp_dir = Path("test/tmp2")
    tmp_out = Path("test/sbdart.out")
    pkl_path = Path("data/lut_alb_ABI2.pkl")
    wavelength = .64 # um
    optical_depths = [0.1, 0,2, 0.4, 0.6, 0.8]+list(range(1,58))
    solar_zeniths = [2*i for i in range(40)] # deg

    """
    The below dictionaries are sufficient for generating a lookup with shape:
    (sza, tau)
    """
    sbdart_args = {
            "idatm":2, # Mid-latitude summer
            "isat":3, # GOES West
            "idatm":4,
            "isat":0,
            "nphi":20,
            "phi":"0,180",
            "nzen":20,
            "uzen":"0,85",
            #"sza":15,
            "iout":5,
            "isalb":7, # Ocean water
            "btemp":292,
            "zcloud":2
            }
    lut_args = {
            "tmp_dir":tmp_dir,
            "wl":wavelength, # ABI channel 6
            "szas":solar_zeniths,
            "taus":optical_depths,
            "sbdart_args":sbdart_args,
            }

    sbdart_args["zout"]="2,3"
    lut = make_alb_lut(**lut_args)
    pkl.dump((lut, lut_args), pkl_path.open("wb"))
