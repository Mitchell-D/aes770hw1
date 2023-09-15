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

def validation_curve(A, B, fig_path=None, nbin=128, show=False, plot_spec={}):
    X = enh.linear_gamma_stretch(np.stack((np.ravel(A),np.ravel(B)), axis=1))
    X = np.rint(X*(nbin-1)).astype(int)
    rmse = np.sqrt(np.sum((X[:,0]-X[:,1])**2)/X[:,0].size)
    print(f"RMSE: {rmse}")
    V = np.zeros((nbin,nbin))
    for i in range(X.shape[0]):
        V[*X[i]] += 1
    gp.plot_heatmap(V, fig_path=fig_path, plot_spec=plot_spec,
                    show_ticks=False, show=show, plot_diagonal=True)

def load_retrievals(fg):
    ret = np.load("data/retrieval_2.npy")
    ret[np.where(np.isnan(ret))] == 0
    fg.add_data("ret2_cod", ret[:,:,0],
                info={"desc":"convergence < 1 ; following DCOMP"})
    fg.add_data("ret2_cre", ret[:,:,1],
                info={"desc":"convergence < 1 ; following DCOMP"})

    ret = np.load("data/retrieval_3.npy")
    ret[np.where(np.isnan(ret))] == 0
    fg.add_data("ret3_cod", ret[:,:,0],
                info={"desc":"convergence < 0.1 ; max iter 20"})
    fg.add_data("ret3_cre", ret[:,:,1],
                info={"desc":"convergence < 0.1 ; max iter 20"})

    ret = np.load("data/retrieval_4.npy")
    ret[np.where(np.isnan(ret))] == 0
    fg.add_data("ret4_cod", ret[:,:,0],
                info={"desc":"cost doesn't decrease ; max iter 20"})
    fg.add_data("ret4_cre", ret[:,:,1],
                info={"desc":"cost doesn't decrease ; max iter 20"})

    ret = np.load("data/retrieval_5.npy")
    ret[np.where(np.isnan(ret))] == 0
    fg.add_data("ret5_cod", ret[:,:,0],
                info={"desc":"cost doesn't decrease ; Sa not squared ; " + \
                        " max iter 10"})
    fg.add_data("ret5_cre", ret[:,:,1],
                info={"desc":"cost doesn't decrease ; Sa not squared ; " + \
                        " max iter 10"})

    ret = np.load("data/retrieval_6.npy")
    ret[np.where(np.isnan(ret))] == 0
    fg.add_data("ret6_cod", ret[:,:,0],
                info={"desc":"cost doesn't decrease ; Sa is diagonal 1"})
    fg.add_data("ret6_cre", ret[:,:,1],
                info={"desc":"cost doesn't decrease ; Sa is diagonal 1"})

    ret = np.load("data/retrieval_brute.npy")
    ret[np.where(np.isnan(ret))] == 0
    fg.add_data("retB_cod", ret[:,:,0],
                info={"desc":"Find close to observed reflectance within " + \
                        "a radius ; return closest to a-priori"})
    fg.add_data("retB_cre", ret[:,:,1],
                info={"desc":"Find close to observed reflectance within " + \
                        "a radius ; return closest to a-priori"})
    return fg


if __name__=="__main__":
    # Path to a pkl containing the above category masks
    fg_path = Path("data/FG_retrievals_aes770hw1.pkl")
    retrieval_path = Path("data/retrieval_3.npy")

    """
    Load the FeatureGrid, adding ABI recipes and sun/pixel/sat geometry
    """
    fg = FeatureGrid.from_pkl(fg_path)
    for label,recipe in abi_recipes.items():
        fg.add_recipe(label,recipe)

    fg = fg.subgrid((
        "2-ref", "6-ref", "14-tb", "cod", "cre", "ocean", "sparse_cloud",
        "dense_cloud", "land", "cre_mask"))
    fg = load_retrievals(fg)

    # Print the area
    #print(np.sum(4*np.cos(np.deg2rad(fg.data("vza")))**(-3)))

    '''
    """ Make a plot of surface classes """
    classes = np.zeros_like(fg.data("2-ref"))
    classes[np.where(fg.data("ocean"))] = 0
    classes[np.where(fg.data("land"))] = 1
    classes[np.where(fg.data("sparse_cloud"))] = 2
    classes[np.where(fg.data("dense_cloud"))] = 3
    labels = ["ocean", "land", "sparse_cloud", "dense_cloud"]
    colors = ["#0033cc", "#ffb366", "#cccccc", "#ffffff"]
    gp.plot_classes(classes, labels, colors, show=True,
                    fig_path=Path("figures/classes.png"),
                    plot_spec={
                        "title": "Minimum-distance Surface Classes"
                        })
    '''

    """ Establish a mask setting valid data value indeces to True """
    mask = np.logical_and(
            np.logical_or(fg.data("sparse_cloud"), fg.data("dense_cloud")),
            #fg.data("dense_cloud"),
            np.logical_not(fg.data("cre_mask")))


    '''
    """ Get histograms for the channel reflectances """
    b2_dict = enh.do_histogram_analysis(fg.data("2-ref")[mask], 64)
    b6_dict =  enh.do_histogram_analysis(fg.data("6-ref")[mask], 64)
    gp.basic_plot(b2_dict["domain"], b2_dict["hist"])
    gp.basic_plot(b6_dict["domain"], b6_dict["hist"])
    print(b2_dict["hist"].shape)
    print(b2_dict["domain"].shape)
    '''

    """ Load retrieval values """
    rnum = 6
    cod = fg.data(f"ret{rnum}_cod")
    cre = fg.data(f"ret{rnum}_cre")
    mask = np.logical_and(mask, np.logical_not(np.isnan(cod)))
    validation_curve(fg.data("cod")[mask],fg.data(f"ret{rnum}_cod")[mask],
                     fig_path=Path(f"figures/val_ret{rnum}_cod.png"),
                     #fig_path=Path(f"figures/val_retB_cod.png"),
                     show=False,
                     plot_spec={
                         "imshow_norm":"log",
                         "title":f"Retrieval {rnum} Cloud Optical Depth Validation",
                         #"title":"Brute-Force COD Validation; Dense Clouds",
                         #"title":"Brute-Force COD Validation",
                         "xlabel":"Retrieval COD",
                         "ylabel":"DCOMP L2 Retrieval COD",
                         "cb_orient":"horizontal",
                         "cb_label":"Count",
                         "line_width":1.5,
                         "cb_size":.5,
                         })
    validation_curve(fg.data("cre")[mask],fg.data(f"ret{rnum}_cre")[mask],
                     fig_path=Path(f"figures/val_ret{rnum}_cre.png"),
                     #fig_path=Path(f"figures/val_retB_cre.png"),
                     show=False,
                     plot_spec={
                         "imshow_norm":"log",
                         "title":f"Retrieval {rnum} Effective Radius Validation",
                         #"title":"Brute-Force CRE Validation; Dense Clouds",
                         #"title":"Brute-Force CRE Validation",
                         "xlabel":"Retrieval CRE",
                         "ylabel":"DCOMP L2 Retrieval CRE",
                         "cb_orient":"horizontal",
                         "cb_label":"Count",
                         "line_width":1.5,
                         "cb_size":.5,
                         })

    exit(0)

    """ Plot my retrieval values """
    plot_spec = {
            "border_width":1,
            "cb_label":"Optical Depth",
            "title":"Cloud optical depth",
            "cb_cmap":"nipy_spectral",
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
