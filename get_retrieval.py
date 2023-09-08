""" Driver script for COD and CRE retrival based on ABI DCOMP """

import numpy as np
import math as m
from pathlib import Path
from datetime import datetime
from datetime import timedelta
import pickle as pkl

from krttdkit.acquire import get_goes as gg
from krttdkit.acquire import abi
from krttdkit.acquire import gesdisc
from krttdkit.products import FeatureGrid
from krttdkit.products import ABIL1b
import krttdkit.visualize.guitools as gt
import krttdkit.visualize.TextFormat as TF
import krttdkit.operate.enhance as enh
import krttdkit.operate.geo_helpers as gh
from krttdkit.operate import abi_recipes

def rayleigh_tau(fg, lapse_rate=-9.76, sfc_rayleigh=.044):
    """
    Adds rayleigh-corrected band 2 reflectances using the dry adiabatic lapse
    rate, the integrated hydrostatic equation, and the relative infrared-window
    brightness temperatures between the dense-cloud and ocean classes.

    This implicitly assumes that the dense clouds have an emissivity
    approaching 1 at 11.2\mu m, which isn't unreasonable for water clouds,
    which readily absorb infrared-range radiation.

    Expects a FeatureGrid with l1b 14-tb and dense_cloud and ocean bool masks
    """
    ctemp = sorted(fg.data("14-tb")[fg.data("dense_cloud")])
    otemp = sorted(fg.data("14-tb")[fg.data("ocean")])
    omode = otemp[len(otemp)//2]
    cmode = ctemp[len(ctemp)//2]
    tdiff = fg.data("14-tb")-omode
    mlt = (cmode+omode)/2
    scale_height = 287*mlt/9.8
    # full grid of cloud heights
    height = tdiff/(lapse_rate*1000)
    # Rayleigh optical depth
    return np.exp(-height/scale_height) * sfc_rayleigh

def atmospheric_reflectance(fg, r_tau, lapse_rate=-9.76, sfc_rayleigh=.044):
    """
    Uses the rayleigh scattering phase function from DCOMP to approximate the
    radiance scattered into the aperature from the atmosphere above the cloud,
    with no cloud interaction. Based on (Wang & King, 1997) 10.1029/97JD02225

    Assumes sza, vza, and scattering angle (scata) fields have been added to
    the FeatureGrid already
    """
    s_mu = np.cos(np.deg2rad(fg.data("sza")))
    v_mu = np.cos(np.deg2rad(fg.data("vza")))
    # From DCOMP ATBD; unclear if scattering angle should be squared.
    r_phase = 3/4 * (1+np.cos(np.deg2rad(fg.data("scata")))**2)
    r_tau = rayleigh_tau(fg, lapse_rate, sfc_rayleigh)
    return (r_tau * r_phase) / (4 * s_mu * v_mu)

def get_vaa(fg):
    """
    Adds north-relative viewing azimuth angles to the FeatureGrid given
    sa-ns and sa-ew
    """
    vaa = np.arctan(np.sin(fg.data("sa-ew"))/np.sin(fg.data("sa-ns")))
    vaa *= 180/m.pi
    if "vaa" in fg.labels:
        fg.drop_data("vaa")
    fg.add_data("vaa", vaa)
    return fg

def get_vza(fg):
    """
    Adds viewing zenith angles to the FeatureGrid given sa-ns and sa-ew
    """
    vza = np.tan((np.sin(fg.data("sa-ns"))**2 + \
            np.sin(fg.data("sa-ew"))**2)**(1/2))
    vza *= 180/m.pi
    if "vza" in fg.labels:
        fg.drop_data("vza")
    fg.add_data("vza", vza)
    return fg

def get_geometry(fg, target_time:datetime):
    """
    Given a FeatureGrid with at least sa-ns,sa-ew,lat,lon variables and a time
    in UTC, adds (sza, saa, vza, vaa, raa, scata) to the FeatureGrid.

    :@param fg: FeatureGrid with "sa-ns" and "sa-ew" variables for scan angles,
        with values in radians
    :@param target_time: Observation time in UTC for solar calculations.
    """
    for label in ("saa", "sza", "vaa", "vza", "raa", "scata"):
        if label in fg.labels:
            fg.drop_data(label)
    # Get the viewing azimuth angle
    sza, saa = gh.sun_angles(
        target_time, fg.data("lat"), fg.data("lon"))
    fg.add_data("sza",sza)
    fg.add_data("saa",saa)
    # Get the viewing azimuth angle
    fg = get_vaa(fg)
    # Get the viewing zenith angle
    fg = get_vza(fg)
    # Calculate and append the relative azimuth angle
    fg.add_data("raa", fg.data("saa")-fg.data("vaa"))
    # Calculate and append the scattering angle
    scata = np.arccos(np.sin(np.deg2rad(fg.data("sza"))) * \
            np.sin(np.deg2rad(fg.data("vza"))) + \
            np.cos(np.deg2rad(fg.data("sza"))) * \
            np.cos(np.deg2rad(fg.data("vza"))))
    fg.add_data("scata", scata, info={"desc":"Scattering angle"})
    return fg

def get_albedo(lut_albedo:np.ndarray, sza:float, tau:float, szas, taus):
    """
    Approximate the albedo of the cloud layer given a lookup table for ABI
    band 2 shaped like (szas, taus)
    """
    szas, taus = map(np.asarray, (szas, taus))
    sza_idx = np.argmin(abs(szas-sza))
    tau_idx = np.argmin(abs(taus-tau))
    return lut_albedo[sza_idx,tau_idx]

def get_radiance(lut2:np.ndarray, lut6:np.ndarray,
                 sza:float, tau:float, cre:float, phi:float, uzen:float,
                 szas, taus, cres, phis, uzens):
    """
    Given a lookup tables for ABI bands 2 and 6, shaped like
    (sza, tau, cre, phi, uzen), and guesses for COD (tau), CRE (cre),
    relative azimuth (phi) and satellite zenith (uzen)
    uses forward-differencing to determine an approximate cloud
    reflectance in band 2 and 6 given a specific COD,

    :@param lut2: Lookup table for ABI band 2, shaped as above
    :@param lut6: Lookup table for ABI band 6, shaped as above

    :@param sza: Solar zenith angle of a pixel
    :@param tau: Approximate cloud optical depth of a pixel
    :@param cre: Approximate effective radius of a cloudy pixel
    :@param phi: Relative zenith angle of a pixel
    :@param uzen: Satellite zenith angle of a pixel

    :@param szas: Solar zenith angle of a pixel
    :@param taus: Approximate cloud optical depth of a pixel
    :@param cres: Approximate effective radius of a cloudy pixel
    :@param phis: Relative zenith angle of a pixel
    :@param uzens: Satellite zenith angle of a pixel
    """
    #wls,taus,cres,phis,uzens = map(np.asarray, lut[1:])

    # Make sure all the guesses are in range of the table
    assert sza>szas[0] and sza<=szas[-1]
    assert tau>taus[0] and tau<=taus[-1]
    assert cre>cres[0] and cre<=cres[-1]
    assert phi>phis[0] and phi<=phis[-1]
    assert uzen>uzens[0] and uzen<=uzens[-1]

    # Make an ordered tuple of the coordinate values
    coords = tuple(map(np.asarray, (szas, taus, cres, phis, uzens)))
    # Check that all coordinate arrays correspond to LUT dimensions
    assert all(len(c.shape)==1 for c in coords)
    assert tuple(c.size for c in coords) == lut2.shape
    assert len(lut2.shape) == 5
    assert lut2.shape==lut6.shape
    L = np.stack((lut2, lut6), axis=0)

    sza_idx = np.argmin(abs(szas-sza))
    uzen_idx = np.argmin(abs(uzens-uzen))
    phi_idx = np.argmin(abs(phis-phi))
    for i in range(len(taus)):
        if taus[i]>tau:
            tau_idx = i-1
            break
    for i in range(len(cres)):
        if cres[i]>cre:
            cre_idx = i-1
            break
    # Get the increment of tau and cre guesses wrt existing grid points
    tau_diff = (tau-taus[tau_idx])/(taus[tau_idx+1]-taus[tau_idx])
    cre_diff = (cre-cres[cre_idx])/(taus[cre_idx+1]-taus[cre_idx])
    # Use the adjacent grid points to estimate the partial derivatives of
    # LUT reflectance in both bands wrt tau and cre
    dr_dtau = (L[:,sza_idx,tau_idx,cre_idx,phi_idx,uzen_idx] - \
            L[:,sza_idx,tau_idx+1,cre_idx,phi_idx,uzen_idx]) / \
            (taus[tau_idx+1] - taus[tau_idx])
    dr_dcre = L[:,sza_idx,tau_idx,cre_idx,phi_idx,uzen_idx] - \
            L[:,sza_idx,tau_idx,cre_idx+1,phi_idx,uzen_idx] / \
            (cres[cre_idx+1] - cres[cre_idx])

    # Make a jacobian matrix like [[dR2/dtau,dR2/dcre], [dR6/dtau,dR6/dcre]]
    jacobian = np.stack((dr_dtau,dr_dcre)).T
    # Assume CRE and COD are orthogonal and calculate the new reflectances
    rad_0 = L[:,sza_idx,tau_idx, cre_idx, phi_idx, uzen_idx]
    rad_guess = rad_0 + cre_diff*dr_dcre + tau_diff*dr_dtau
    # Return the 2-vector radiances guess and 2x2 jacobian.
    return rad_guess, jacobian

def get_cost(exp_refs, model_refs, prior_state, new_state, Sy, Sa):
    """
    :@param exp_refs: Observed reflectance
    :@param exp_refs: Model reflectance
    :@param exp_refs: A-priori state (tau, cre)
    :@param exp_refs: Current state (tau, cre)
    """
    exp_refs, model_refs, prior_state, new_state = map(
            np.asarray, (exp_refs, model_refs, prior_state, new_state))
    assert len(model_refs)==2
    assert len(exp_refs)==2
    assert len(prior_state)==2
    assert len(new_state)==2
    # Background error (from DCOMP ATBD)
    ref_diff = exp_refs-model_refs
    state_diff = prior_state-new_state
    model = np.matmul(ref_diff.T, np.matmul(np.linalg.inv(Sy), ref_diff))
    prior = np.matmul(state_diff.T, np.matmul(np.linalg.inv(Sa), state_diff))
    return model + prior

def optimal_estimate(lut2, lut6, lut_albedo, obs,
                     atmo_reflectance, rayleigh_depth,
                     sza, phi, uzen, # pixel geometry
                     szas, taus, cres, phis, uzens, # coordinates
                     max_count=1000, prior=None):
    """
    :@param lut2: Lookup table for band 2, shaped: (sza, tau, cre, phi, uzen)
    :@param lut6: Lookup table for band 6, shaped: (sza, tau, cre, phi, uzen)
    :@param lut_albedo: Lookup table for cloud albedo shaped: (sza, tau)
    :@param obs: Observation reflectance vector with reflectances in channel
        2 and 6 like (ref_2, ref_6)

    :@param atmo_reflectance: Atmospheric reflectance at this pixel from
        rayleigh single-scatter and no cloud interaction
    :@param rayleigh_depth: Rayleigh optical depth at this pixel given the
        approximate cloud height.

    :@param phi: Pixel relative azimuth angle
    :@param uzen: Pixel viewing zenith angle
    :@param sza: Pixel solar zenith angle

    :@param szas: Solar zenith angle of a pixel
    :@param taus: Approximate cloud optical depth of a pixel
    :@param cres: Approximate effective radius of a cloudy pixel
    :@param phis: Relative zenith angle of a pixel
    :@param uzens: Satellite zenith angle of a pixel

    :@param max_count: Maximum number of iterations per pixel
    :@param prior: Prior (tau, cre) vector; If none is provided
    """
    # (band2, band6) kappa0 values for converting rad -> ref
    kappa0 = np.array([0.0019486,0.0415484])
    szas, taus, cres, phis, uzens = map(
            np.array, (szas, taus, cres, phis, uzens))
    obs = np.asarray(obs)
    # Establish an a-priori guess based on a 10um (water cloud) effective
    # radius and the reflectance in conservative-scattering channel 2
    if prior is None:
        prior_cre = 10 # um
        sza_idx = np.argmin(abs(szas-sza))
        cre_idx = np.argmin(abs(cres-prior_cre))
        phi_idx = np.argmin(abs(phis-phi))
        uzen_idx = np.argmin(abs(uzens-uzen))
        # Expected channel 2 radiance
        exp_rad = obs[0]/kappa0[0]
        tau_idx = np.argmin(abs(
            lut2[sza_idx,:,cre_idx,phi_idx,uzen_idx] - exp_rad))
        prior_tau = taus[tau_idx]
        prior = np.array((prior_tau, prior_cre))

    # Background error: calibration, forward-model, plane-parallel, offset
    Sy = (0.05+0.01+0.1) * prior * np.asarray([[1,0],[0,1]])
    Sy = (Sy+np.asarray([[.02,0],[0,.02]]))**2
    # Prior value error (from DCOMP ATBD)
    Sa = np.asarray([[.04, 0],[0,.25]])
    X = prior

    count = 0
    convergence = 100000
    # expects (wl, tau, cre, phi, uzen)
    while count < max_count:
        # Get radiance and jacobian for the current state
        rad, jac = get_radiance(
                lut2,
                lut6,
                sza=sza,
                tau=X[0],
                cre=X[1],
                phi=phi,
                uzen=uzen,
                szas=szas,
                taus=taus,
                cres=cres,
                phis=phis,
                uzens=uzens,
                )
        # Convert radiances to reflectances by scaling by kappa0
        ref = rad*kappa0
        jac *= np.stack((kappa0,kappa0)).T

        """
        Use reflectance components from Wang & King to correct the conservative
        scattering (0.64um) band for back-scattering effects given the
        single-scatter assumption
        """
        # Get the cloud albedo for the current optical depth
        c_alb = get_albedo(lut_albedo, sza, X[0], szas, taus)
        s_mu = np.cos(np.deg2rad(sza))
        v_mu = np.cos(np.deg2rad(uzen))
        amf = 1/s_mu + 1/v_mu
        # Atmospheric scatters into aperature without cloud interaction
        R1 = atmo_reflectance
        # Scattered from atmosphere, then reflected by cloud
        R2 = rayleigh_depth/(2*s_mu) * c_alb * np.exp(-rayleigh_depth/v_mu)
        # Reflected from cloud, then scattered by atmosphere
        R3 = rayleigh_depth/(2*v_mu) * c_alb * np.exp(-rayleigh_depth/s_mu)
        obs[0] = obs[0]*np.exp(-rayleigh_depth*amf) - R1 - R2 - R3

        # Get current state's error coveriance matrix
        Sx = np.linalg.inv(np.linalg.inv(Sa) + \
                np.matmul(jac.T,np.matmul(np.linalg.inv(Sy),jac)))
        ydiff = obs-ref
        xdiff = prior-X
        # State differential step. DCOMP has a typo in the delta X calculation,
        # so it's not clear whether the prior value uncertainty should be
        # included in the matrix multiple with the inverse jacobian.
        '''
        sdiff = np.matmul(jac.T,np.matmul(np.linalg.inv(Sy), ydiff)) + \
                np.matmul(np.linalg.inv(Sa), xdiff)
        sdiff = np.matmul(Sx, sdiff)
        '''
        #'''
        sdiff = np.matmul(jac.T,(np.matmul(np.linalg.inv(Sy), ydiff))) + \
                np.matmul(np.linalg.inv(Sa), xdiff)
        sdiff = np.matmul(Sx, sdiff)
        #'''
        Xnew = X + sdiff
        # Cost function
        #pcost = get_cost(obs, ref, X, Xnew, Sy, Sa)
        convergence = np.matmul((X-Xnew).T, np.matmul(
            np.linalg.inv(Sx), (X-Xnew)))
        if convergence <= .01:
            break
        X = Xnew
        count += 1
    return X

if __name__=="__main__":
    """ Settings """
    # Path to a pkl containing the above category masks
    pkl_path = Path("data/FG_subgrid_aes770hw1.pkl")
    # UTC time of observation
    target_time = datetime(2022,10,7,19,27)

    """ Load the relevant lookup tables """
    LU_B2, args2 = pkl.load(Path("data/lut_rad_ABI2.pkl").open("rb"))
    LU_B6, args6 = pkl.load(Path("data/lut_rad_ABI6.pkl").open("rb"))
    LU_ALB, args_alb = pkl.load(Path("data/lut_alb_ABI2.pkl").open("rb"))

    """ Load the FeatureGrid, adding ABI recipes and sun/pixel/sat geometry """
    fg = FeatureGrid.from_pkl(pkl_path)
    for label,recipe in abi_recipes.items():
        fg.add_recipe(label,recipe)
    fg = get_geometry(fg, target_time)

    """ Get an estimate for the cloud height """
    ray_tau = rayleigh_tau(fg)
    ref_atmo = atmospheric_reflectance(fg, ray_tau)

    """ Get reflectances in bands 2 and 6 as well as a dense cloud mask """
    not_cloud = np.logical_not(fg.data("dense_cloud"))
    clouds2= fg.data("2-ref-rc")
    clouds6 = fg.data("6-ref")

    """ Get the optimal estimate for each pixel value """
    #prior = np.array([19,8])  # prior guess for (COD, CRE)
    retrieval = np.zeros((*clouds2.shape,2))
    for j in range(clouds2.shape[0]):
        for i in range(clouds2.shape[1]):
            sza = fg.data("sza")[j,i]
            # Ignore values that aren't dense clouds, or with high SZA
            if not_cloud[j,i] or sza>70:
                continue
            obs = clouds2[j,i], clouds6[j,i]
            uzen = fg.data("vza")[j,i]
            phi = fg.data("raa")[j,i]
            retrieval[j,i] = optimal_estimate(
                    lut2=LU_B2,
                    lut6=LU_B6,
                    lut_albedo=LU_ALB,
                    obs=obs,
                    rayleigh_depth=ray_tau[j,i],
                    atmo_reflectance=ref_atmo[j,i],
                    #prior=prior,
                    sza=sza,
                    phi=phi,
                    uzen=uzen,
                    szas=args2["szas"],
                    taus=args2["taus"],
                    cres=args2["cres"],
                    phis=args2["phis"],
                    uzens=args2["uzens"]
                    )
    np.save(Path("data/retrieval.npy"), retrieval)
    if "ret_tau" in fg.labels:
        fg.drop_data("ret_tau")
    if "ret_cre" in fg.labels:
        fg.drop_data("ret_cre")
    fg.add_data("ret_tau", retrieval[:,:,0])
    fg.add_data("ret_cre", retrieval[:,:,1])
    fg.to_pkl(pkl_path)
    #gt.quick_render(enh.linear_gamma_stretch(clouds2))
