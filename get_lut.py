"""
Script for generating lokup tables at a given wavelength.
The returned tables are shaped like: (sza, tau, cre, phi, uzen)
for each solar zenith angle, optical depth, cloud effective radius, relative
azimuth, and viewing zenith angle in the specified ranges.
"""
#from sbdart_info import default_params
from pathlib import Path
from krttdkit.acquire.sbdart import dispatch_sbdart, parse_iout
import krttdkit.visualize.guitools as gt
import numpy as np
import pickle as pkl

def make_lut(tmp_dir:Path, wl:float, szas, taus:list, cres:list,
                    sbdart_args:dict={}, zcloud=3, print_stdout=False):
    """
    For each wavelength, optical depth, and effective radius in the provided
    lists, this method provides a 2d array of spectral radiances expected for
    the viewing geometries specified by parameters (nphi, phi, nzen, uzen)

    (sza, tau, cre, phi, uzen)

    :@param wl: Wavelength of the lookup table in um
    :@param szas: List of solar zenith angles in degrees
    :@param taus: List of optical depths
    :@param cres: List of cloud effective radii
    :@param sbdart_args: dict of SBDART arguments (If wlinf, wlsup, wlinc,
        tcloud, sza, zcloud, or nre are in the provided dict, they will be
        overwritten.)
    """
    uzen, phi = None, None
    srad = []
    for s in szas:
        print(f"Next sza: {s}")
        tmp_tau = []
        for t in taus:
            tmp_cre = []
            for r in cres:
                new_args = {"wlinf":wl, "wlsup":wl, "tcloud":t, "sza":s,
                            "nre":r, "wlinc":0., "zcloud":zcloud, "iout":5}
                sbdart_args.update(new_args)
                sb_out=dispatch_sbdart(sbdart_args, tmp_dir)
                out = parse_iout(iout_id=5, sb_out=sb_out,
                                 print_stdout=print_stdout)
                if uzen is None:
                    phi = out["phi"]
                    uzen = out["uzen"]
                # We're doing each wavelength independently, so squeeze out
                # the 1-element wavelength dimension
                tmp_cre.append(np.squeeze(out["srad"]))
            tmp_tau.append(np.stack(tmp_cre, axis=0))
        srad.append(np.stack(tmp_tau, axis=0))
    return np.stack(srad, axis=0), phi, uzen

if __name__=="__main__":
    tmp_dir = Path("test/tmp2")
    tmp_out = Path("test/sbdart.out")
    pkl_path = Path("data/lut_ABI6.pkl")

    """
    The below dictionaries are sufficient for generating a lookup with shape:
    (sza, tau, cre, phi, uzen)
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
            }
    lut_args = {
            "tmp_dir":tmp_dir,
            #"wl":.64, # ABI channel 2
            "wl":2.24, # ABI channel 6
            "szas":[2*i for i in range(40)],
            "taus":[0.1, 0,2, 0.4, 0.6, 0.8]+list(range(1,58)),
            "cres":list(range(2,64)),
            #"szas":[0, 15, 30],
            #"taus":[5, 10],
            #"cres":[5, 10, 20, 30],
            "sbdart_args":sbdart_args,
            "zcloud":2
            }

    bs, phi, uzen = make_lut(**lut_args, print_stdout=False)
    lut_args["phis"] = phi
    lut_args["uzens"] = uzen
    pkl.dump((bs, lut_args), pkl_path.open("wb"))
