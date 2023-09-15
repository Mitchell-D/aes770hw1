"""
Script for generating figures using a FeatureGrid and retrieval
"""
import netCDF4 as nc
import numpy as np
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from krttdkit.acquire import abi
from krttdkit.products import FeatureGrid
from krttdkit.visualize import guitools as gt
from krttdkit.visualize import geoplot as gp
from krttdkit.operate import abi_recipes
from krttdkit.operate import enhance as enh

def scalar_masked(scalar:np.ndarray, mask:np.ndarray, data_range:tuple):
    """
    Make an RGB by applying a boolean mask, setting True values to
    black.
    """
    print("unmasked:", enh.array_stat(scalar[np.logical_not(mask)]))
    scalar[mask] = np.amin(scalar[np.logical_not(mask)])
    np.clip((scalar-data_range[0])/(data_range[1]-data_range[0]),0,1)
    rgb = gt.scal_to_rgb(scalar)
    rgb[mask] = np.asarray([0,0,0])
    return rgb

def fg_figs(fg:FeatureGrid, mask):
    """ Make RGBs of bands 2 and 6 reflectances """
    plot_spec = {
            "border_width":1,
            "cb_label":"Reflectance",
            "title":"ABI Band 2 (0.64$\mu m$) Reflectance",
            "cb_cmap":"nipy_spectral",
            "cb_orient":"horizontal",
            "cb_label_format":"{x:.2f}",
            "title_size":18,
            "cb_tick_size":14,
            "dpi":200,
            }
    gp.geo_scalar_plot(
            fg.data("2-ref"),
            fg.data("lat"),
            fg.data("lon"),
            plot_spec=plot_spec,
            fig_path=Path("report/figs/scal_2-ref.png"),
            show=False,
            )
    plot_spec["title"] = "ABI Band 6 (2.24$\mu m$) Reflectance"
    gp.geo_scalar_plot(
            fg.data("6-ref"),
            fg.data("lat"),
            fg.data("lon"),
            plot_spec=plot_spec,
            fig_path=Path("report/figs/scal_6-ref.png"),
            show=False,
            )

    """ Make RGBs of L2 COD and CRE values """
    plot_spec["title"] = "ABI L2 Cloud Optical Depth (DCOMP)"
    plot_spec["cb_label"] = "Cloud Optical Depth"
    plot_spec["cb_label_format"] = "{x:.0f}"
    cod = fg.data("cod")
    cod[mask] = 0
    gp.geo_scalar_plot(
            cod,
            fg.data("lat"),
            fg.data("lon"),
            plot_spec=plot_spec,
            fig_path=Path("report/figs/scal_l2-cod.png"),
            show=False,
            )
    plot_spec["title"] = "ABI L2 Cloud Effective Radius (DCOMP)"
    plot_spec["cb_label"] = "Cloud Effective Radius ($\mu m$)"
    cre = fg.data("cre")
    cre[mask] = 0
    gp.geo_scalar_plot(
            np.clip(cre,0,80),
            fg.data("lat"),
            fg.data("lon"),
            plot_spec=plot_spec,
            fig_path=Path("report/figs/scal_l2-cre.png"),
            show=False,
            )

if __name__=="__main__":
    # Path to a pkl containing the above category masks
    fg_path = Path("data/FG_subgrid_aes770hw1.pkl")

    """
    Load the FeatureGrid, adding ABI recipes and sun/pixel/sat geometry
    """
    fg = FeatureGrid.from_pkl(fg_path)
    for label,recipe in abi_recipes.items():
        fg.add_recipe(label,recipe)

    """
    Make a mask for everything that isn't a cloud, incorporating both
    classification results and the DCOMP retrieval mask.
    """
    notcloud = np.logical_or(
            np.logical_not(np.logical_or(
                fg.data("dense_cloud"),
                fg.data("sparse_cloud"),
                )),
            fg.data("cre_mask")
            )

    """ Generate scalar figures """
    #fg_figs(fg, notcloud)

    #gt.quick_render(fg.data("histeq selectgamma truecolor"))
    #gt.quick_render(fg.data("selectgamma diffwv"))

    """ Load retrieval values """
    #retrieval_path = Path("data/retrieval_3.npy")
    #retrieval_path = Path("data/retrieval_7.npy")
    retrieval_path = Path("data/retrieval_8.npy")
    #retrieval_path = Path("data/retrieval_brute.npy")
    ret = np.load(retrieval_path)
    cod_r = ret[:,:,0]
    cre_r = ret[:,:,1]
    notcloud = np.logical_or(notcloud, np.isnan(cod_r))
    iscloud = np.logical_not(notcloud)

    cod_r[notcloud] = 0
    cre_r[notcloud] = 0

    #cod_r[np.where(np.isnan(cod_r))] = np.nanmin(cod_r)
    #cre_r[np.where(np.isnan(cre_r))] = np.nanmin(cre_r)

    #cod_err = np.abs((cod_r-fg.data("cod")))
    #cre_err = np.abs((cre_r-fg.data("cre")))
    cod_err = cod_r-fg.data("cod")
    cre_err = cre_r-fg.data("cre")

    cod_err[notcloud] = 0
    cre_err[notcloud] = 0
    cod_err[np.abs(cod_err)>=15] = 15
    cre_err[np.abs(cre_err)>=15] = 15
    cod_err[notcloud] = -15
    cre_err[notcloud] = -15
    retid = "8"

    """ Plot my retrieval values """
    plot_spec = {
            "border_width":1,
            "cb_cmap":"nipy_spectral",
            "cb_orient":"horizontal",
            "cb_label_format":"{x:.0f}",
            "title_size":18,
            "cb_tick_size":14,
            "dpi":200,
            }
    #'''
    plot_spec["title"] = "Cloud optical depth",
    plot_spec["cb_label"] = "Optical Depth",
    gp.geo_scalar_plot(
            cod_r,
            fg.data("lat"),
            fg.data("lon"),
            plot_spec=plot_spec,
            fig_path=Path(f"figures/ret{retid}_cod.png"),
            #fig_path=Path("report/figs/brute_cod.png"),
            show=False,
            )
    plot_spec["title"] = "Cloud effective radius"
    plot_spec["cb_label"] = "$r_e$ $(\mu m)$"
    gp.geo_scalar_plot(
            cre_r,
            fg.data("lat"),
            fg.data("lon"),
            plot_spec=plot_spec,
            fig_path=Path(f"figures/ret{retid}_cre.png"),
            #fig_path=Path("report/figs/brute_cre.png"),
            show=False,
            )
    #'''
    plot_spec["title"] = "Cloud Optical Depth Error Magnitude"
    plot_spec["cb_label"] = None
    #plot_spec["cb_cmap"] = "YlOrRd"
    #plot_spec["cb_cmap"] = "jet"
    gp.geo_scalar_plot(
            #cod_r,
            cod_err,
            fg.data("lat"),
            fg.data("lon"),
            plot_spec=plot_spec,
            fig_path=Path(f"figures/ret{retid}_cod-err.png"),
            show=False,
            )
    plot_spec["title"] = "Cloud Effective Radius Error Magnitude"
    plot_spec["cb_label"] = None
    gp.geo_scalar_plot(
            #cre_r,
            cre_err,
            fg.data("lat"),
            fg.data("lon"),
            plot_spec=plot_spec,
            fig_path=Path(f"figures/ret{retid}_cre-err.png"),
            show=False,
            )
