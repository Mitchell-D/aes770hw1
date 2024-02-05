# aes770hw1

Cloud property retrieval for Satellite Remote Sensing II

Using GOES ABI data for a 640 by 512 region of stratocumulus clouds,
perform cloud optical depth and cloud effective radii retrievals and
discuss your results. Also validate your results with existing GOES
level 2 (or other satellite) products.

<p align="center">
  <img width="300" src="https://github.com/Mitchell-D/aes770hw1/blob/main/report/figs/bispec_seabreeze.png" />
  <img width="300" src="https://github.com/Mitchell-D/aes770hw1/blob/main/report/figs/bispec_seabreeze_domain.png" />
</p>

## scripts

### get\_heatmaps\_aes770h21.py

Given boolean masks covering valid cloud pixels, plot a heatmap of
RED (.64um) vs NIR (2.24um) reflectances.

### get\_lut\_alb.py

Given a lists solar zenith angles, cloud optical depths, and
a single wavelength, use `krttdkit.acquire.sbdart` to create a lookup
table like (szas, taus) for the total hemispherical radiative power
at the given wavelength for each combination, saving the result in
a pkl file like (lut, lut\_args) where lut\_args contains coords.

### get\_lut\_rad.py

Given a list of solar zenith angles, cloud optical depths, and cloud
particle effective radii, and a specific cloud layer height and
wavelength, generate a lookup table like (sza, tau, cre, phi, uzen)
depicting the TOA radiance through the atmospheric column. Save the
result as a pkl file like (lut, lut\_args) where lut\_args contains
each of the coordinates as a dictionary.

### get\_modis.py

Use `krttdkit.products.MOD021KM` to download a series of useful bands
from a MODIS granule on the LAADS DAAC.

### get\_retrieval.py

1. Calculate sun/pixel/satellite geometry on the native grid
2. Estimate cloud heights using the 11.2um LWIR band and
3. Calculate the exponential "tau" coefficient to correct for
   atmospheric path radiance from rayleigh scattering.
4. Apply an inverted cloud mask
5. Using the lookup tables for RED and NIR, execute optimal estimate
   inversion on each pixel independently to determine cloud optical
   depth (COD) and effective cloud particle radius (CRE)
6. Save the subsequent cloud optical depth retrieval as a npy file.

### get\_subgrid\_aes770hw1.py

Use `krttdkit.products.ABIL1b` to download cotemporal ABI L1b and L2
products, and merge them onto a common grid, before loading and
saving them in the same `krttdkit.products.FeatureGrid`. Optionally
run maximum likelihood classification and minimum distance
classification using user-selected pixel samples.

### plot\_bispectral.py

Provides plotting methods for selecting a region of an ABI image, via
a GUI interface and plotting a bispectral diagram of RED and NIR
reflectances from pixels within, overlayed with isolines from the
lookup tables for CRE and COD.

### plot\_scalars.py

Use retrievals saved in npy files and corresponding FeatureGrid pkls
to plot derived scalar-gridded values, such as mean error rates.

### video\_script.py

Methods for using ffmpeg to render a video as an mp4 from a directory
of images, and for rasterizing text onto the video.
