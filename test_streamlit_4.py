import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Literal

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import streamlit as st
from streamlit import session_state as st_state
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar

from AFMReader.jpk import load_jpk
# from topostats.scars import remove_scars
# from AFM_Functions import remove_scars, remove_scars_fast
from scar_removal import remove_scars
# from topostats.filters import Filters

from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sky_colors = [
    "#000000", # Black
    "#262965", # Dark Blue
    "#4B6477",
    "#CA7A3F", 
    "#FCD355",
    "#FFFFFF", # White
]
# playing around to have diverging palette
# These hex codes approximate the Gwyddion Sky transition
# sky_colors = [
#     "#262965", # Dark Blue
#     "#191B43",
#     # "#0D0E22",
#     "#000000", # Black
#     "#653D1F",
#     "#CA7A3F", 
#     "#FCD355",
#     "#FFFFFF", # White
# ]

# twilight = [
# "#e2d9e2",
# "#b3c6ce",
# "#7ba1c2",
# "#6276ba",
# "#5e43a5",
# "#4e186f",
# "#2f1436",
# "#581647",
# "#8e2c50",
# "#b25652",
# "#c6896c",
# "#d4bcac",
# "#e2d9e2",
# ]

# # sky_colors_diverging
# sky_colors = [
# "#4B6477",
# "#262965",
# "#191B43",
# # "#6276ba",
# # "#5e43a5",
# # "#4e186f",
# "#2f1436",
# # "#581647",
# # "#8e2c50",
# # "#b25652",
# "#CA7A3F",
# "#FCD355",
# "#FFFFFF",
# ]

# # gold palette
# sky_colors = [
# "#000000",
# "#581C00",
# "#BC8000",
# "#FCFC80"
# ]

# # gold palette
# sky_colors = [
# "#000000",
# "#7C3044",
# "#C22F2F",
# "#DE7E1D",
# # "#E8E8E8",
# "#FFFFFF"
# ]


# Create the colormap
# N=256 provides a smooth gradient
gwy_sky = LinearSegmentedColormap.from_list("gwy_sky", sky_colors, N=256)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CHANNEL_TRACE   = "height_trace"
CHANNEL_RETRACE = "height_retrace"
POLY_DEGREE     = 1


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    mask_threshold_nm: float = 2.0
    poly_degree: int = POLY_DEGREE
    iterations: int = 1
    channel: Literal["trace", "retrace", "average"] = "trace"
    use_gaussian: bool = False
    gaussian_sigma: float = 0.8
    use_median: bool = False
    median_size: int = 3
    outlier_mad_factor: float = 20.0
    zmin: float = 0.0
    zmax: float = 5.0
    set_colorbar_at_0: bool = True
    scalebar_location: str = "lower right"
    scalebar_color: str = "white"
    scalebar_frameon: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clamp_outliers(img: np.ndarray, mad_factor: float = 5.0) -> np.ndarray:
    """
    Clip pixel values beyond median ± mad_factor × MAD.
    Prevents single hot/dead pixels from skewing polynomial fits and color range.
    """
    flat = img[np.isfinite(img)]
    med  = np.median(flat)
    mad  = np.median(np.abs(flat - med))
    lo   = med - mad_factor * mad
    hi   = med + mad_factor * mad
    return np.clip(img, lo, hi)


def subtract_2d_plane(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Fit and subtract a 2D plane from the entire image.
    If mask is provided (True = background), only background pixels are used for fitting.
    Corrects both X-tilt and slow Y-direction drift in a single step.
    """
    rows, cols = np.indices(img.shape)
    if mask is not None:
        valid = mask & np.isfinite(img)
    else:
        valid = np.isfinite(img)
    A      = np.column_stack([rows[valid], cols[valid], np.ones(valid.sum())])
    coeffs, _, _, _ = np.linalg.lstsq(A, img[valid], rcond=None)
    plane  = coeffs[0] * rows + coeffs[1] * cols + coeffs[2]
    return img - plane


def align_rows_polynomial(
    z: np.ndarray,
    degree: int = 1,
    mask: Optional[np.ndarray] = None,
    mask_mode: Literal["exclude", "include"] = "exclude",
    preserve_global_mean: bool = True,
) -> np.ndarray:
    """
    Row-wise polynomial alignment / leveling for AFM images.

    Fits a polynomial independently to each scan row and subtracts the fitted
    baseline from that row. This is commonly used to remove row offsets,
    row-wise tilt, bow, or slow scanner drift in AFM topography images.

    Parameters
    ----------
    z : np.ndarray
        2D AFM height image with shape (ny, nx), where rows are scan lines and
        columns are samples along each row. Input is converted to float64.

    degree : int, default=1
        Degree of the polynomial fitted to each row.

        Typical choices:
        - 0 : subtract row mean / constant offset
        - 1 : remove row tilt
        - 2 : remove row bow / quadratic curvature
        - 3+ : stronger flattening, but higher risk of removing real structure

    mask : np.ndarray of bool, optional
        Boolean mask with the same shape as `z`.

        Meaning depends on `mask_mode`:
        - if mask_mode="exclude": True pixels are excluded from the fit
        - if mask_mode="include": True pixels are the only pixels used in the fit

        This is useful in AFM when foreground features, particles, steps, or
        defects should not influence the estimated row baseline.

    mask_mode : {"exclude", "include"}, default="exclude"
        Controls how `mask` is interpreted.

        - "exclude":
            Use unmasked pixels (`mask == False`) for polynomial fitting.
            This is the most common AFM use case when the mask marks features
            to ignore.

        - "include":
            Use only masked pixels (`mask == True`) for fitting.
            Useful when the mask explicitly marks trusted background regions.

    preserve_global_mean : bool, default=True
        If True, shifts the corrected image after row alignment so that the
        global mean of the output matches the global mean of the input.

        This preserves the overall height reference after per-row baseline
        subtraction.

    Returns
    -------
    np.ndarray
        Corrected image of the same shape as `z`, with row-wise polynomial
        baselines removed.

    Notes
    -----
    - Only finite values are used in polynomial fitting.
    - If a row does not contain enough valid points to fit the requested
      polynomial degree, the function falls back to subtracting that row's
      finite mean.
    - This is a row-by-row correction. It does not remove column-wise drift,
      2D background curvature, or line-to-line discontinuities beyond the
      fitted row baseline.
    - In AFM processing, degree 1 or 2 is usually safer than degree 3+.

    Raises
    ------
    ValueError
        If `z` is not 2D, if `degree < 0`, if `mask` has the wrong shape, or
        if `mask_mode` is invalid.

    Examples
    --------
    Remove row tilt from an AFM image:

    >>> zcorr = align_rows_polynomial(z, degree=1)

    Exclude foreground features from the row fit:

    >>> zcorr = align_rows_polynomial(
    ...     z,
    ...     degree=2,
    ...     mask=feature_mask,
    ...     mask_mode="exclude"
    ... )

    Fit only known background pixels:

    >>> zcorr = align_rows_polynomial(
    ...     z,
    ...     degree=2,
    ...     mask=background_mask,
    ...     mask_mode="include"
    ... )
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("z must be a 2D array.")
    if degree < 0:
        raise ValueError("degree must be >= 0.")

    ny, nx = z.shape
    x = np.arange(nx, dtype=np.float64)
    out = z.copy()

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != z.shape:
            raise ValueError("mask must have the same shape as z.")

    for i in range(ny):
        row = z[i]
        finite = np.isfinite(row)

        if mask is None:
            ok = finite
        else:
            if mask_mode == "exclude":
                ok = (~mask[i]) & finite
            elif mask_mode == "include":
                ok = mask[i] & finite
            else:
                raise ValueError("mask_mode must be 'exclude' or 'include'.")

        if np.count_nonzero(ok) >= degree + 1:
            coeff = np.polyfit(x[ok], row[ok], deg=degree)
            baseline = np.polyval(coeff, x)
        else:
            baseline = np.full_like(row, np.nanmean(row))

        out[i] = row - baseline

    if preserve_global_mean:
        out += np.nanmean(z) - np.nanmean(out)

    return out


def apply_smoothing(
    img: np.ndarray,
    use_gaussian: bool,
    gaussian_sigma: float,
    use_median: bool,
    median_size: int
) -> np.ndarray:
    """
    Optional post-alignment smoothing.
    Gaussian and median can be combined (Gaussian first, then median).
    """
    out = img.copy()
    if use_gaussian:
        out = ndi.gaussian_filter(out, sigma=gaussian_sigma)
    if use_median:
        out = ndi.median_filter(out, size=median_size)
    return out


def full_pipeline(
    image_nm: np.ndarray,
    cfg: PipelineConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    aligned = image_nm.copy()

    for _ in range(cfg.iterations):
        # Step 0 — coarse row leveling for mask generation
        aligned = align_rows_polynomial(aligned, mask=None, degree=1)
        aligned -= float(np.nanmin(aligned))
        if cfg.outlier_mad_factor > 0.0:
            aligned = clamp_outliers(aligned, cfg.outlier_mad_factor)

        # Step 1 — scar removal
        aligned = remove_scars(aligned)

        # Step 2 — background mask from current leveled image
        mask = aligned < cfg.mask_threshold_nm

        # Step 3 — subtract global plane using background only
        aligned = subtract_2d_plane(aligned, mask=mask)

        # Step 4 — masked polynomial row alignment using BACKGROUND pixels
        aligned = align_rows_polynomial(
            aligned,
            degree=cfg.poly_degree,
            mask=mask,
            mask_mode="include",   # because True means background
        )

        # Step 5 — optional floor shift
        aligned -= float(np.nanmin(aligned))

        # Step 6 — final scar pass
        # aligned = remove_scars(aligned)

    aligned = apply_smoothing(
        aligned,
        cfg.use_gaussian, cfg.gaussian_sigma,
        cfg.use_median,   cfg.median_size
    )

    return aligned, mask

# ═══════════════════════════════════════════════════════════════════════════════
# FILE I/O
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading file…")
def load_file(
    filepath: str,
    channel: Literal['trace', 'retrace', 'average'] = 'trace',
) -> Tuple[np.ndarray, float]:
    """
    Load a JPK file.  If channel='average' and height_retrace exists,
    return pixel-wise average of trace and retrace (reduces random noise).

    Returns
    -------
    image_nm     : height array in nm
    scan_size_um : physical scan width in µm
    px_to_nm     : nanometres per pixel (used for scale bar)
    """
    if channel in ('trace', 'average'):
        channel_name = CHANNEL_TRACE
    elif channel == 'retrace':
        channel_name = CHANNEL_RETRACE
    trace, px_to_nm = load_jpk(file_path=filepath, channel=channel_name)
    image_nm = trace.copy().astype(float)

    if channel == 'average':
        try:
            retrace, _ = load_jpk(file_path=filepath, channel=CHANNEL_RETRACE)
            retrace_nm = retrace.copy().astype(float)
            image_nm   = 0.5 * (image_nm + retrace_nm)
        except Exception:
            st.warning("Retrace channel not found — using trace only.")

    scan_size_um = image_nm.shape[1] * (px_to_nm / 1e3)  # convert nm to µm
    return image_nm, scan_size_um, float(px_to_nm)


@st.cache_data(show_spinner="Processing…")
def run_pipeline(
    image_nm: np.ndarray,
    cfg_json: str           # serialised PipelineConfig (hashable for cache)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    cfg                       = PipelineConfig(**json.loads(cfg_json))
    processed, mask = full_pipeline(image_nm, cfg)
    return processed, mask

# ═══════════════════════════════════════════════════════════════════════════════
# SCALE BAR HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def add_scale_bar(
    ax: plt.Axes,
    xy_unit: str,
    location: str = "lower right",
    color: str = "white",
    frameon: bool = False,
) -> None:

    sb = ScaleBar(
        # set dx to 1.0 because the axes image has already been calibrated by
        1.0, # setting its ``extent``
        xy_unit,
        length_fraction=0.25,
        location=location,
        color=color,
        box_color="black",
        box_alpha=0.0 if not frameon else 0.4,
        frameon=frameon,
        font_properties={"size": 10},
        dimension="si-length",
    )

    ax.add_artist(sb)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
def get_display_extent(scan_size_um: float):
    """
    Choose axis unit based on scan size:
    - If scan size >= 1 µm → use µm
    - If scan size < 1 µm  → convert to nm to avoid scientific notation
    Returns (extent, unit_label)
    """
    if scan_size_um >= 1.0:
        return [0, scan_size_um, 0, scan_size_um], "µm"
    else:
        scan_size_nm = scan_size_um * 1e3
        return [0, scan_size_nm, 0, scan_size_nm], "nm"


def _colorbar_ticks(fig, im, ax, label: str, vmin: float, vmax: float,
                    set_at_0: bool) -> None:
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()
    if set_at_0:
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{v - vmin:.2f}" for v in ticks])


def make_figure(
    image_nm:  np.ndarray,
    aligned:   np.ndarray,
    mask:      np.ndarray,
    extent:    list,
    cfg:       PipelineConfig,
    filename:  str,
    xy_unit:   str = "µm",
    colormap = "afmhot"
) -> plt.Figure:
    """3-panel diagnostic figure: Raw | Mask | Processed."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(filename, fontsize=10)

    raw_vmin = float(np.nanmin(image_nm))
    raw_vmax = float(np.nanmax(image_nm))

    # Panel 1 — Raw
    if colormap == 'gwy_sky':
        colormap = gwy_sky
    im_raw = axes[0].imshow(image_nm, cmap=colormap, origin="upper",
                            extent=extent, interpolation="nearest",
                            vmin=raw_vmin, vmax=raw_vmax)
    axes[0].set_title("Raw height (nm)")
    axes[0].set_xlabel(f"X ({xy_unit})")
    axes[0].set_ylabel(f"Y ({xy_unit})")
    _colorbar_ticks(fig, im_raw, axes[0], "Height (nm)", raw_vmin, raw_vmax,
                    cfg.set_colorbar_at_0)

    # Panel 2 — Background mask
    axes[1].imshow(mask.astype(float), cmap="gray", origin="upper",
                   extent=extent, interpolation="nearest", vmin=0, vmax=1)
    axes[1].set_title(f"Background mask (< {cfg.mask_threshold_nm:.2f} nm)\n"
                      "white = background · black = features")
    axes[1].set_xlabel(f"X ({xy_unit})")
    axes[1].set_ylabel(f"Y ({xy_unit})")

    # Panel 3 — Processed
    im_al = axes[2].imshow(np.clip(aligned, cfg.zmin, cfg.zmax),
                           cmap=colormap, origin="upper", extent=extent,
                           interpolation="nearest",
                           vmin=cfg.zmin, vmax=cfg.zmax)
    axes[2].set_title("Processed height (nm)")
    axes[2].set_xlabel(f"X ({xy_unit})")
    axes[2].set_ylabel(f"Y ({xy_unit})")
    _colorbar_ticks(fig, im_al, axes[2], "Height (nm)", cfg.zmin, cfg.zmax,
                    cfg.set_colorbar_at_0)
    add_scale_bar(
        axes[2],
        xy_unit,
        cfg.scalebar_location,
        cfg.scalebar_color,
        cfg.scalebar_frameon
    )

    plt.tight_layout()
    return fig


def make_processed_only_fig(
    aligned:  np.ndarray,
    extent:   list,
    cfg:      PipelineConfig,
    xy_unit:  str,
    colormap = "afmhot"
) -> plt.Figure:
    """Clean single-panel publication figure with scale bar, no axes."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    if colormap == 'gwy_sky':
        colormap = gwy_sky
    im = ax.imshow(np.clip(aligned, cfg.zmin, cfg.zmax),
                   cmap=colormap, origin="upper", extent=extent,
                   interpolation="nearest", vmin=cfg.zmin, vmax=cfg.zmax)

    ax.axis("off")      # clean: no tick marks for publication

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (nm)")
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()

    if cfg.set_colorbar_at_0:
        ticks       = np.linspace(cfg.zmin, cfg.zmax, 5)
        label_ticks = np.linspace(0, cfg.zmax - cfg.zmin, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{v:.2f}" for v in label_ticks])

    add_scale_bar(ax, xy_unit,
                cfg.scalebar_location,
                cfg.scalebar_color,
                cfg.scalebar_frameon)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    if 'fold_path' not in st_state:
        st_state.fold_path = ''

    st.set_page_config(page_title="AFM Viewer", layout="wide",
                    initial_sidebar_state="expanded")
    st.title("🔬 AFM JPK Image Viewer")

    # ── Sidebar ───────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📂 File selection")
        folder = st.text_input("Folder path", value=st_state.fold_path,
                            placeholder="/path/to/jpk/files")
        if folder and folder != st_state.fold_path:
            st_state.fold_path = folder
            st.rerun()

        if not folder:
            st.info("Enter a folder path to start")
            folds = [f for f in sorted(os.listdir(str(Path.home())),
                                       key=lambda x: x.lower())
                        if (os.path.isdir(str(Path.home()/f)) 
                            and not f.startswith('.'))]
            picked_folder = st.pills('Or pick a folder',
                                     folds)
            if picked_folder:
                st_state.fold_path = str(Path.home()/picked_folder)
                st.rerun()
            st.stop()
        if not os.path.isdir(folder):
            st.error("Path does not exist.")
            st.stop()

        # Collect .jpk files (flat + FastScan subfolders)
        jpk_files = []
        for f in sorted(os.listdir(folder)):
            if f.endswith(".jpk"):
                jpk_files.append(f)
            elif os.path.isdir(os.path.join(folder, f)) and f.startswith("imaging-"):
                for subf in sorted(os.listdir(os.path.join(folder, f))):
                    if subf.endswith(".jpk"):
                        jpk_files.append(os.path.join(f, subf))

        if not jpk_files:
            st.warning("No .jpk files found.")
            folds = [f for f in sorted(os.listdir(folder), key=lambda x: x.lower())
                        if (os.path.isdir(os.path.join(folder, f)) 
                            and not f.startswith('.'))]
            picked_folder = st.pills('Or pick a folder',
                                     folds)
            if picked_folder:
                st_state.fold_path = os.path.join(folder, picked_folder)
                st.rerun()
            st.stop()

        # create topostats log file
        st_state.topo_log_path = os.path.join(folder, "topostats_log.txt")

        selected_ind  = st.select_slider("Select file",
                                        options=range(len(jpk_files)), value=0)
        selected_file = jpk_files[selected_ind]
        st.write(f"Selected: `{selected_file}`")
        filepath  = os.path.join(folder, selected_file)
        save_stem = os.path.splitext(
                        os.path.join(folder, os.path.basename(filepath))
                    )[0]

        # ── Processing ────────────────────────────────────────────────────────────
        with st.container(horizontal=True, 
                        horizontal_alignment="distribute",
                        vertical_alignment="bottom"):
            st.header("⚙️ Processing")

            channel = st.segmented_control(
                "Height channel",
                ["trace", 'retrace', 'average'],
                default="average",
                help="Load trace, retrace, or average of both (if retrace exists; reduces noise)."
            )

        # Load file
        image_nm, scan_size_um, px_to_nm = load_file(filepath, channel)

        # ── Crop options ─────────────────────────────────────────────────────────────
        orig_image_nm = image_nm
        orig_h, orig_w = orig_image_nm.shape

        with st.container(horizontal=True, horizontal_alignment="distribute",
                        vertical_alignment="center"):

            with st.popover("✂️ Crop image"):
                use_crop = st.toggle("Enable crop", value=False)

                crop_w = st.number_input(
                    "Crop width (px)",
                    min_value=1,
                    max_value=orig_w,
                    value=orig_w,
                    step=1,
                    disabled=not use_crop,
                )

                crop_h = st.number_input(
                    "Crop height (px)",
                    min_value=1,
                    max_value=orig_h,
                    value=orig_h,
                    step=1,
                    disabled=not use_crop,
                )

                max_x = max(0, orig_w - int(crop_w))
                max_y = max(0, orig_h - int(crop_h))

                crop_x = st.slider(
                    "X position (px)",
                    min_value=0,
                    max_value=max_x + 1,  # add small epsilon to allow max value when max_x=0
                    value=0,
                    step=1,
                    disabled=not use_crop or max_x == 0,
                )

                crop_y = st.slider(
                    "Y position (px)",
                    min_value=0,
                    max_value=max_y + 1,  # add small epsilon to allow max value when max_y=0
                    value=0,
                    step=1,
                    disabled=not use_crop or max_y == 0,
                )

                st.caption(
                    f"Crop: x={crop_x}, y={crop_y}, "
                    f"width={int(crop_w)}, height={int(crop_h)} px"
                )

            if use_crop:
                x0 = int(crop_x)
                y0 = int(crop_y)
                x1 = x0 + int(crop_w)
                y1 = y0 + int(crop_h)

                image_nm = orig_image_nm[y0:y1, x0:x1]

                scan_size_um = image_nm.shape[1] * (px_to_nm / 1e3)

                crop_meta = {
                    "enabled": True,
                    "x_px": x0,
                    "y_px": y0,
                    "width_px": int(crop_w),
                    "height_px": int(crop_h),
                    "original_width_px": int(orig_w),
                    "original_height_px": int(orig_h),
                }
            else:
                crop_meta = {
                    "enabled": False,
                    "x_px": 0,
                    "y_px": 0,
                    "width_px": int(orig_w),
                    "height_px": int(orig_h),
                    "original_width_px": int(orig_w),
                    "original_height_px": int(orig_h),
                }

            with st.popover("Smoothing"):
                col1, col2 = st.columns(2)
                with col1:
                    use_gaussian    = st.toggle("Gaussian", value=False)
                    gaussian_sigma  = st.number_input("σ (px)", 0.3, 5.0, 0.8, 0.1,
                                                    disabled=not use_gaussian)
                with col2:
                    use_median  = st.toggle("Median", value=False)
                    median_size = st.number_input("Kernel (px)", 2, 9, 3, 1,
                                                disabled=not use_median)
                    
            # ── Slider max: coarse-leveled raw image range ────────────────────────────
            # _coarse = clamp_outliers(_coarse, mad_factor=10.0)
            # _auto_max = float(np.nanmax(_coarse))
            # slider_max = round(_auto_max * 1.1, 1)   # 10% headroom, 1 decimal

            p5, p95 = np.nanpercentile(image_nm, [5, 95])
            slider_max = 1.2 * (p95 - p5)        # robust range with 20% headroom

            with st.popover("Processing settings"):
                poly_degree = st.number_input("Polynomial degree for row alignment",
                                            min_value=0, max_value=5, value=POLY_DEGREE, step=1)
                outlier_mad = st.number_input("Outlier MAD factor for mask generation",
                                            min_value=0.0, value=0.0, step=0.1,
                                            help="Factor for identifying outliers in mask generation." 
                                            " Higher values are more permissive (fewer outliers) and lower values are more aggressive (more outliers). " 
                                                "Set to 0 to disable outlier masking.")
                iterations  = st.number_input("Number of processing iterations", min_value=1, value=1, step=1)
                override_slider_max = st.toggle("Override slider max", value=False)
                if override_slider_max:
                    slider_max = st.number_input(
                        "Slider max (nm)",
                        min_value=0.1,
                        value=float(slider_max),
                        step=0.5,
                    )

        mask_thresh = st.slider("Mask threshold (nm)", 0.0, slider_max,
                                slider_max, step = 0.05,
                                help="Pixels below this are background for poly fitting.")



        # ── Display range ─────────────────────────────────────────────────────────
        st.header("🎨 Display range")

        with st.container(horizontal=True, horizontal_alignment="distribute",
                        vertical_alignment="center"):
            use_fixed = st.toggle("Fixed Z window", value=False)
            if use_fixed:
                window_size = st.number_input("Window size (nm)", min_value=0.0,
                                             value = 2.8, step=0.05)

        if use_fixed:
            zmin = st.slider("Z baseline (nm)", -2.0, slider_max, 0.0, 0.05)
            zmax = zmin + window_size
        else:
            zmin, zmax = st.slider("Z range (nm)", -2.0, slider_max,
                                (0.0, slider_max), 0.05)

        set_colorbar_at_0 = st.checkbox(
            "Show processed colorbar relative to display minimum",
            value=True,
            help="Display the processed colorbar as 0 → (zmax-zmin) without changing the underlying data."
        )

        # ── Scale bar options ─────────────────────────────────────────────────────
        with st.popover("📏 Scale bar options"):
            sb_location = st.selectbox(
                "Location",
                ["lower right", "lower left", "upper right", "upper left"],
                index=0
            )
            sb_color  = st.color_picker("Bar & text color", "#FFFFFF")
            sb_frame  = st.checkbox("Show background box", value=False)

        options = list(mpl.colormaps) + ["gwy_sky"]
        colormap = st.selectbox(
            "Colormap",
            options,
            index=options.index("gwy_sky"), #options.index("afmhot"),
            help="Colormap for raw and processed images."
        )

    # ── Validation ────────────────────────────────────────────────────────────────
    if zmin >= zmax:
        st.error("Z min must be less than Z max.")
        st.stop()

    # ── Build config & run ────────────────────────────────────────────────────────
    cfg = PipelineConfig(
        mask_threshold_nm = mask_thresh,
        poly_degree       = poly_degree,
        iterations        = iterations,
        channel           = channel,
        use_gaussian      = use_gaussian,
        gaussian_sigma    = float(gaussian_sigma),
        use_median        = use_median,
        median_size       = int(median_size),
        outlier_mad_factor= outlier_mad,
        zmin              = zmin,
        zmax              = zmax,
        set_colorbar_at_0 = set_colorbar_at_0,
        scalebar_location = sb_location,
        scalebar_color    = sb_color,
        scalebar_frameon  = sb_frame,
    )

    processed, mask = run_pipeline(
        image_nm, cfg.to_json()
    )
    extent, xy_unit = get_display_extent(scan_size_um)

    # ── Version & save path ───────────────────────────────────────────────────────
    v_num     = st.number_input("Save version number", min_value=0, value=0, step=1)
    save_path = f"{save_stem}_v{v_num}.png"
    save_path_svg = f"{save_stem}_v{v_num}.svg"
    meta_path = f"{save_stem}_v{v_num}_metadata.json"

    if os.path.isfile(save_path):
        st.warning(f"Existing PNG found → `{save_path}`")
        with st.container(horizontal=True, horizontal_alignment="distribute",
                        vertical_alignment="center"):
            redo = st.toggle("Reprocess JPK file", value=False)
        if not redo:
            st.info(f"Displaying saved PNG → `{save_path}`")
            st.image(plt.imread(save_path))
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    st.json(f.read())
            st.stop()

    # ── Render ────────────────────────────────────────────────────────────────────
    processed_fig = None
    show_processed_only = st.toggle("Show processed image only", value=False)

    if show_processed_only:
        processed_fig = make_processed_only_fig(processed, extent, cfg, xy_unit, colormap=colormap)
        st.pyplot(processed_fig)
    else:
        fig = make_figure(image_nm, processed, mask, extent, cfg,
                        selected_file, xy_unit, colormap=colormap)
        st.pyplot(fig)
        plt.close(fig)

    # ── Save ──────────────────────────────────────────────────────────────────────
    with st.container(horizontal=True, horizontal_alignment="distribute",
                    vertical_alignment="center"):
        saved = False
        if st.button("💾 Save Processed PNG", use_container_width=True):
            if processed_fig is None:
                processed_fig = make_processed_only_fig(processed, extent, cfg, xy_unit, colormap=colormap)
            processed_fig.savefig(save_path, dpi=600, bbox_inches="tight")
            plt.close(processed_fig)
            saved = save_path
        if st.button("💾 Save Processed SVG", use_container_width=True):
            if processed_fig is None:
                processed_fig = make_processed_only_fig(processed, extent, cfg, xy_unit, colormap=colormap)
            processed_fig.savefig(save_path_svg, dpi=600, bbox_inches="tight")
            plt.close(processed_fig)
            saved = save_path_svg

        if saved:
            # Save metadata sidecar
            meta = {
                "source_file":  filepath,
                "save_path":    save_path,
                "pipeline":     json.loads(cfg.to_json()),
                "crop":         crop_meta,
                "scan_size_um": scan_size_um,
                "px_to_nm":     px_to_nm,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            st.success(f"Saved → `{saved}`")
            st.success(f"Metadata → `{meta_path}`")
            with st.expander("View saved metadata"):
                st.json(meta)
