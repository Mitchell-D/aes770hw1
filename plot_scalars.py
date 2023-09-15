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

def validation_curve(A, B, nbin=128):
    X = enh.linear_gamma_stretch(np.stack((np.ravel(A),np.ravel(B)), axis=1))
    X = np.rint(X*(nbin-1)).astype(int)
    V = np.zeros((nbin,nbin))
    print(X.shape)
    for i in range(X.shape[0]):
        V[*X[i]] += 1
    gp.plot_heatmap(V, plot_spec={"imshow_norm":"log"})

if __name__=="__main__":
    # Path to a pkl containing the above category masks
    fg_path = Path("data/FG_subgrid_aes770hw1.pkl")
    retrieval_path = Path("data/retrieval_3.npy")

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
    ret = np.load(retrieval_path)
    cod_r = ret[:,:,0]
    cre_r = ret[:,:,1]
    cod_r[np.where(np.isnan(cod_r))] = np.nanmin(cod_r)
    cre_r[np.where(np.isnan(cre_r))] = np.nanmin(cre_r)
    #gt.quick_render(scalar_masked(cod_r, notcloud, (0,50)))
    #gt.quick_render(scalar_masked(cre_r, notcloud, (0,50)))

    validation_curve(cod_r[np.logical_not(notcloud)],
                     fg.data("cod")[np.logical_not(notcloud)])
    validation_curve(cre_r[np.logical_not(notcloud)],
                     fg.data("cre")[np.logical_not(notcloud)])

    """ Plot my retrieval values """
    plot_spec = {
            "border_width":1,
            "cb_label":"Optical Depth",
            "title":"Cloud optical depth",
            "cb_cmap":"ripy_spectral",
            "cb_orient":"horizontal",
            "cb_label_format":"{x:.0f}",
            "title_size":18,
            "cb_tick_size":14,
            "dpi":200,
            }
    gp.geo_scalar_plot(
            cod_r,
            fg.data("lat"),
            fg.data("lon"),
            plot_spec=plot_spec,
            fig_path=Path("figures/ret_cod.png"),
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
            fig_path=Path("figures/ret_cre.png"),
            #fig_path=Path("report/figs/brute_cre.png"),
            show=False,
            )
