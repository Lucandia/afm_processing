# Afm img processing

A Streamlit app for viewing and processing AFM height images from JPK files, with
publication-ready figure export.

Load a folder of `.jpk` scans, flatten and clean each image through a configurable
pipeline, tune the display interactively, and save processed figures (PNG/SVG) with a
JSON record of every setting used.

## Features

- **JPK loading** — trace, retrace, or the average of both channels.
- **Flattening pipeline** — plane subtraction, row median alignment, masked 2D
  polynomial background removal, and background-based row polynomial alignment, run for
  one or more iterations.
- **Scar removal** — fast line-artefact correction via a compiled Cython module
  (`none` / `default` / `aggressive`).
- **Optional cleanup** — MAD outlier clipping, Gaussian and median smoothing.
- **Crop & rotate** — square crop at any position, with rotation, applied before or
  after processing.
- **Display control** — adjustable Z range, mask threshold, scale bar, and any
  Matplotlib colormap (plus a custom `gwy_sky`).
- **Export** — high-DPI transparent PNG/SVG plus a `*_metadata.json` capturing the full
  config, crop, and scan geometry for reproducibility.

## Get the code

Clone the repository:

```bash
git clone https://github.com/Lucandia/afm_processing.git
cd afm_processing
```

Or download the ZIP from the [GitHub page](https://github.com/Lucandia/afm_processing),
unpack it, and `cd` into the extracted folder.

## Install

```bash
pip install numpy scipy matplotlib matplotlib-scalebar streamlit AFMReader cython
```

The scar-removal module is written in Cython and must be compiled once:

```bash
python setup.py build_ext --inplace
```

## Run

```bash
streamlit run AFM_analysis.py
```

Then enter the path to a folder containing `.jpk` files in the sidebar. Files inside
`imaging-*` subfolders are picked up automatically.

## Layout

| File | Purpose |
|------|---------|
| `AFM_analysis.py` | Main Streamlit app and processing pipeline |
| `scar_removal.pyx` | Cython scar-removal implementation |
| `AFM_Functions.py` | Pure-Python scar-removal reference |
| `setup.py` | Builds the Cython extension |
| `luts.py`, `preprocessing.py` | Colormap and preprocessing helpers |
