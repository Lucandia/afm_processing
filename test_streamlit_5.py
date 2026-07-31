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
from matplotlib.colors import LinearSegmentedColormap

from AFMReader.jpk import load_jpk
from scar_removal import remove_scars


# ═══════════════════════════════════════════════════════════════════════════════
# COLORMAP
# ═══════════════════════════════════════════════════════════════════════════════

sky_colors = [
    "#000000",
    "#262965",
    "#4B6477",
    "#CA7A3F",
    "#FCD355",
    "#FFFFFF",
]

gwy_sky = LinearSegmentedColormap.from_list("gwy_sky", sky_colors, N=256)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CHANNEL_TRACE = "height_trace"
CHANNEL_RETRACE = "height_retrace"

DEFAULT_ROW_POLY_DEGREE = 1
DEFAULT_BACKGROUND_POLY_DEGREE = 3


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    mask_threshold_nm: float = 2.0

    row_poly_degree: int = DEFAULT_ROW_POLY_DEGREE
    background_poly_degree: int = DEFAULT_BACKGROUND_POLY_DEGREE

    iterations: int = 1
    channel: Literal["trace", "retrace", "average"] = "trace"

    scar_removal: Literal["none", "default", "aggressive"] = "default"

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
    """
    flat = img[np.isfinite(img)]
    med = np.median(flat)
    mad = np.median(np.abs(flat - med))

    if mad == 0:
        return img.copy()

    lo = med - mad_factor * mad
    hi = med + mad_factor * mad

    return np.clip(img, lo, hi)


def subtract_2d_plane(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fit and subtract a 2D plane.

    If mask is provided:
        True pixels are used for fitting.
    If mask is None:
        all finite pixels are used.
    """
    img = np.asarray(img, dtype=np.float64)
    rows, cols = np.indices(img.shape)

    if mask is None:
        valid = np.isfinite(img)
    else:
        mask = np.asarray(mask, dtype=bool)
        valid = mask & np.isfinite(img)

    A = np.column_stack([
        rows[valid],
        cols[valid],
        np.ones(valid.sum()),
    ])

    coeffs, _, _, _ = np.linalg.lstsq(A, img[valid], rcond=None)

    plane = (
        coeffs[0] * rows +
        coeffs[1] * cols +
        coeffs[2]
    )

    return img - plane


def median_align_rows(
    img: np.ndarray,
    preserve_global_median: bool = True,
) -> np.ndarray:
    """
    Median-align rows.

    For each scan row, subtract that row's median. Optionally restore the
    original global median height.
    """
    img = np.asarray(img, dtype=np.float64)
    out = img.copy()

    global_median = np.nanmedian(img)

    row_medians = np.nanmedian(out, axis=1)

    for i in range(out.shape[0]):
        out[i, :] = out[i, :] - row_medians[i]

    if preserve_global_median:
        out += global_median

    return out


def polynomial_terms_2d(
    rows: np.ndarray,
    cols: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Build a 2D polynomial design matrix with terms:

        row^i * col^j where i + j <= degree
    """
    terms = []

    for total_degree in range(degree + 1):
        for i in range(total_degree + 1):
            j = total_degree - i
            terms.append((rows ** i) * (cols ** j))

    return np.column_stack(terms)


def subtract_2d_polynomial_background(
    img: np.ndarray,
    degree: int = 3,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fit and subtract a 2D polynomial background.

    If mask is provided:
        True pixels are used for fitting.
    If mask is None:
        all finite pixels are used.

    degree=1 is equivalent to plane leveling.
    degree=2 removes quadratic bow.
    degree=3 removes cubic background curvature.
    """
    img = np.asarray(img, dtype=np.float64)

    if img.ndim != 2:
        raise ValueError("img must be a 2D array.")

    if degree < 0:
        raise ValueError("degree must be >= 0.")

    ny, nx = img.shape
    rows, cols = np.indices(img.shape)

    # Normalize coordinates for numerical stability.
    r = rows.astype(np.float64)
    c = cols.astype(np.float64)

    r = 2.0 * (r / max(ny - 1, 1)) - 1.0
    c = 2.0 * (c / max(nx - 1, 1)) - 1.0

    if mask is None:
        valid = np.isfinite(img)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != img.shape:
            raise ValueError("mask must have the same shape as img.")
        valid = mask & np.isfinite(img)

    n_terms = (degree + 1) * (degree + 2) // 2

    if np.count_nonzero(valid) < n_terms:
        # Not enough background pixels to fit requested polynomial.
        # Fall back to plane subtraction.
        return subtract_2d_plane(img, mask=mask)

    A = polynomial_terms_2d(r[valid], c[valid], degree)
    coeffs, _, _, _ = np.linalg.lstsq(A, img[valid], rcond=None)

    A_full = polynomial_terms_2d(r.ravel(), c.ravel(), degree)
    background = A_full @ coeffs
    background = background.reshape(img.shape)

    return img - background


def align_rows_polynomial(
    z: np.ndarray,
    degree: int = 1,
    mask: Optional[np.ndarray] = None,
    mask_mode: Literal["exclude", "include"] = "exclude",
    preserve_global_mean: bool = True,
) -> np.ndarray:
    """
    Row-wise polynomial alignment.

    If mask_mode='include':
        True mask pixels are used for fitting.

    If mask_mode='exclude':
        True mask pixels are excluded from fitting.
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
    median_size: int,
) -> np.ndarray:
    """
    Optional final smoothing.
    Gaussian is applied first, then median.
    """
    out = img.copy()

    if use_gaussian:
        out = ndi.gaussian_filter(out, sigma=gaussian_sigma)

    if use_median:
        out = ndi.median_filter(out, size=median_size)

    return out


def rotate_for_crop(
    img: np.ndarray,
    angle_deg: float,
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    """
    Rotate an image about its center for cropping.

    reshape=False keeps the array shape unchanged so that crop-position pixel
    coordinates (computed from the original dimensions) stay valid and the
    pixel-to-nm scale is preserved. Regions rotated in from outside the image
    are filled with `cval` (NaN for height data, 0 for masks). Use order=1
    (bilinear) for continuous height data and order=0 (nearest) for masks.
    """
    if angle_deg == 0:
        return img

    return ndi.rotate(
        img,
        angle_deg,
        reshape=False,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=False,
    )


def full_pipeline(
    image_nm: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    AFM processing pipeline.

    New order:

    Initial preprocessing:
        1. mean plane level subtraction
        2. median align rows
        3. zero-minimum shift
        4. optional MAD outlier clipping

    Iterative processing:
        5. scar removal
        6. background mask
        7. remove polynomial background
        8. background-based row polynomial alignment

    Final:
        9. zero-minimum shift
        10. optional Gaussian/median smoothing
    """
    aligned = image_nm.copy().astype(np.float64)

    # ── Initial preprocessing ──────────────────────────────────────────────────

    # 1. Mean/global plane level subtraction using all finite pixels.
    aligned = subtract_2d_plane(aligned, mask=None)

    # 2. Median align rows.
    aligned = median_align_rows(aligned)

    # 3. Zero-minimum shift.
    aligned -= float(np.nanmin(aligned))

    # 4. Optional MAD outlier clipping.
    if cfg.outlier_mad_factor > 0.0:
        aligned = clamp_outliers(aligned, cfg.outlier_mad_factor)

    # Fallback mask in case iterations somehow do not run.
    mask = np.isfinite(aligned)

    # ── Iterative processing ───────────────────────────────────────────────────

    for _ in range(cfg.iterations):

        # 5. Scar removal.
        if cfg.scar_removal == "none":
            pass
        elif cfg.scar_removal == "default":
            aligned = remove_scars(aligned)
        elif cfg.scar_removal == "aggressive":
            aligned = remove_scars(aligned,
                                removal_iterations=3,
                                threshold_low=0.20,
                                threshold_high=0.50,
                                max_scar_width=10,
                                min_scar_length=12,
                                )


        # 6. Background mask.
        # True = background.
        mask = aligned < cfg.mask_threshold_nm

        # 7. Remove polynomial background using background pixels.
        aligned = subtract_2d_polynomial_background(
            aligned,
            degree=cfg.background_poly_degree,
            mask=mask,
        )

        # 8. Background-based row polynomial alignment.
        aligned = align_rows_polynomial(
            aligned,
            degree=cfg.row_poly_degree,
            mask=mask,
            mask_mode="include",
        )

    # ── Final processing ───────────────────────────────────────────────────────

    # 9. Final zero-minimum shift.
    aligned -= float(np.nanmin(aligned))

    # 10. Optional Gaussian/median smoothing.
    aligned = apply_smoothing(
        aligned,
        cfg.use_gaussian,
        cfg.gaussian_sigma,
        cfg.use_median,
        cfg.median_size,
    )

    return aligned, mask


# ═══════════════════════════════════════════════════════════════════════════════
# FILE I/O
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading file…")
def load_file(
    filepath: str,
    channel: Literal["trace", "retrace", "average"] = "trace",
) -> Tuple[np.ndarray, float, float]:
    """
    Load a JPK file.

    If channel='average' and height_retrace exists, return the pixel-wise
    average of trace and retrace.
    """
    if channel in ("trace", "average"):
        channel_name = CHANNEL_TRACE
    elif channel == "retrace":
        channel_name = CHANNEL_RETRACE
    else:
        raise ValueError("channel must be 'trace', 'retrace', or 'average'.")

    trace, px_to_nm = load_jpk(file_path=filepath, channel=channel_name)
    image_nm = trace.copy().astype(float)

    if channel == "average":
        try:
            retrace, _ = load_jpk(file_path=filepath, channel=CHANNEL_RETRACE)
            retrace_nm = retrace.copy().astype(float)
            image_nm = 0.5 * (image_nm + retrace_nm)
        except Exception:
            st.warning("Retrace channel not found — using trace only.")

    scan_size_um = image_nm.shape[1] * (px_to_nm / 1e3)

    return image_nm, scan_size_um, float(px_to_nm)


@st.cache_data(show_spinner="Processing…")
def run_pipeline(
    image_nm: np.ndarray,
    cfg_json: str,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = PipelineConfig(**json.loads(cfg_json))
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
        1.0,
        xy_unit,
        length_fraction=0.25,
        width_fraction=0.02,
        location=location,
        color=color,
        box_color="black",
        box_alpha=0.0 if not frameon else 0.4,
        frameon=frameon,
        font_properties={"size": 10},
        pad=2,
        dimension="si-length",
    )

    ax.add_artist(sb)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def get_display_extent(scan_size_um: float):
    """
    Choose axis unit based on scan size.
    """
    if scan_size_um >= 1.0:
        return [0, scan_size_um, 0, scan_size_um], "µm"
    else:
        scan_size_nm = scan_size_um * 1e3
        return [0, scan_size_nm, 0, scan_size_nm], "nm"


def _colorbar_ticks(
    fig,
    im,
    ax,
    label: str,
    vmin: float,
    vmax: float,
    set_at_0: bool,
) -> None:
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
    image_nm: np.ndarray,
    aligned: np.ndarray,
    mask: np.ndarray,
    extent: list,
    cfg: PipelineConfig,
    filename: str,
    xy_unit: str = "µm",
    colormap="afmhot",
) -> plt.Figure:
    """
    3-panel diagnostic figure:
        Raw | Background mask | Processed
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(filename, fontsize=10)

    raw_vmin = float(np.nanmin(image_nm))
    raw_vmax = float(np.nanmax(image_nm))

    if colormap == "gwy_sky":
        colormap = gwy_sky

    # Panel 1 — Raw
    im_raw = axes[0].imshow(
        image_nm,
        cmap=colormap,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        vmin=raw_vmin,
        vmax=raw_vmax,
    )

    axes[0].set_title("Raw height (nm)")
    axes[0].set_xlabel(f"X ({xy_unit})")
    axes[0].set_ylabel(f"Y ({xy_unit})")

    _colorbar_ticks(
        fig,
        im_raw,
        axes[0],
        "Height (nm)",
        raw_vmin,
        raw_vmax,
        cfg.set_colorbar_at_0,
    )

    # Panel 2 — Background mask
    axes[1].imshow(
        mask.astype(float),
        cmap="gray",
        origin="upper",
        extent=extent,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        f"Background mask (< {cfg.mask_threshold_nm:.2f} nm)\n"
        "white = background · black = features"
    )

    axes[1].set_xlabel(f"X ({xy_unit})")
    axes[1].set_ylabel(f"Y ({xy_unit})")

    # Panel 3 — Processed
    im_al = axes[2].imshow(
        np.clip(aligned, cfg.zmin, cfg.zmax),
        cmap=colormap,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        vmin=cfg.zmin,
        vmax=cfg.zmax,
    )

    axes[2].set_title("Processed height (nm)")
    axes[2].set_xlabel(f"X ({xy_unit})")
    axes[2].set_ylabel(f"Y ({xy_unit})")

    _colorbar_ticks(
        fig,
        im_al,
        axes[2],
        "Height (nm)",
        cfg.zmin,
        cfg.zmax,
        cfg.set_colorbar_at_0,
    )

    add_scale_bar(
        axes[2],
        xy_unit,
        cfg.scalebar_location,
        cfg.scalebar_color,
        cfg.scalebar_frameon,
    )

    plt.tight_layout()

    return fig


def make_processed_only_fig(
    aligned: np.ndarray,
    extent: list,
    cfg: PipelineConfig,
    xy_unit: str,
    colormap="afmhot",
) -> plt.Figure:
    """
    Clean single-panel publication-style figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Make SVG/PDF background transparent
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    if colormap == "gwy_sky":
        colormap = gwy_sky

    im = ax.imshow(
        np.clip(aligned, cfg.zmin, cfg.zmax),
        cmap=colormap,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        vmin=cfg.zmin,
        vmax=cfg.zmax,
    )

    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Optional: make colorbar background transparent too
    cbar.ax.set_facecolor("none")

    cbar.set_label("Height (nm)")
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()

    if cfg.set_colorbar_at_0:
        ticks = np.linspace(cfg.zmin, cfg.zmax, 5)
        label_ticks = np.linspace(0, cfg.zmax - cfg.zmin, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{v:.2f}" for v in label_ticks])

    add_scale_bar(
        ax,
        xy_unit,
        cfg.scalebar_location,
        cfg.scalebar_color,
        cfg.scalebar_frameon,
    )

    plt.tight_layout()

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if "fold_path" not in st_state:
        st_state.fold_path = ""

    st.set_page_config(
        page_title="AFM Viewer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 AFM JPK Image Viewer")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📂 File selection")

        folder = st.text_input(
            "Folder path",
            value=st_state.fold_path,
            placeholder="/path/to/jpk/files",
        )

        if folder and folder != st_state.fold_path:
            st_state.fold_path = folder
            st.rerun()

        if not folder:
            st.info("Enter a folder path to start")

            folds = [
                f for f in sorted(os.listdir(str(Path.home())), key=lambda x: x.lower())
                if os.path.isdir(str(Path.home() / f)) and not f.startswith(".")
            ]

            picked_folder = st.pills("Or pick a folder", folds)

            if picked_folder:
                st_state.fold_path = str(Path.home() / picked_folder)
                st.rerun()

            st.stop()

        if not os.path.isdir(folder):
            st.error("Path does not exist.")
            st.stop()

        # Collect .jpk files.
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

            folds = [
                f for f in sorted(os.listdir(folder), key=lambda x: x.lower())
                if os.path.isdir(os.path.join(folder, f)) and not f.startswith(".")
            ]

            picked_folder = st.pills("Or pick a folder", folds)

            if picked_folder:
                st_state.fold_path = os.path.join(folder, picked_folder)
                st.rerun()

            st.stop()

        st_state.topo_log_path = os.path.join(folder, "topostats_log.txt")

        selected_ind = st.select_slider(
            "Select file",
            options=range(len(jpk_files)),
            value=0,
        )

        selected_file = jpk_files[selected_ind]
        st.write(f"Selected: `{selected_file}`")

        filepath = os.path.join(folder, selected_file)

        save_stem = os.path.splitext(
            os.path.join(folder, os.path.basename(filepath))
        )[0]

        # ── Processing ────────────────────────────────────────────────────────
        with st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="bottom",
        ):
            st.header("⚙️ Processing")

            channel = st.segmented_control(
                "Height channel",
                ["trace", "retrace", "average"],
                default="average",
                help="Load trace, retrace, or average of both.",
            )

        # Load file.
        image_nm, scan_size_um, px_to_nm = load_file(filepath, channel)

        # ── Crop options ──────────────────────────────────────────────────────
        orig_image_nm = image_nm
        orig_h, orig_w = orig_image_nm.shape

        with st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        ):

            with st.popover("✂️ Crop image"):
                use_crop = st.toggle("Enable crop", value=False)

                # nm per pixel
                nm_per_px = px_to_nm

                # A square crop is limited by the smaller image dimension.
                max_dim_px = min(orig_w, orig_h)
                max_size_nm = max_dim_px * nm_per_px

                with st.container(
                    horizontal=True,
                    horizontal_alignment="distribute",
                    vertical_alignment="center",
                ):
                    crop_unit = st.radio(
                        "Crop size unit",
                        ["nm", "µm"],
                        horizontal=True,
                        disabled=not use_crop,
                    )

                    if crop_unit == "nm":
                        max_size = float(max_size_nm)
                        step = 1.0
                    else:
                        max_size = float(max_size_nm / 1e3)
                        step = 0.01

                    crop_size_phys = st.number_input(
                        f"Crop size ({crop_unit})",
                        min_value=0.0,
                        max_value=max_size,
                        value=max_size,
                        step=step,
                        disabled=not use_crop,
                        help="Side length of the square crop.",
                    )

                # Convert the requested physical size to a square pixel size.
                crop_size_nm = (
                    crop_size_phys if crop_unit == "nm" else crop_size_phys * 1e3
                )
                crop_px = int(round(crop_size_nm / nm_per_px))
                crop_px = max(1, min(crop_px, max_dim_px))

                crop_w = crop_px
                crop_h = crop_px

                rotation_deg = st.slider(
                    "Rotation (°)",
                    min_value=-180.0,
                    max_value=180.0,
                    value=0.0,
                    step=0.5,
                    disabled=not use_crop,
                    help="Rotate the image about its center before cropping. "
                         "Applied after processing when crop timing is "
                         "'after processing'. Keep the crop box away from the "
                         "corners, which become blank after rotation.",
                )

                max_x = max(0, orig_w - crop_px)
                max_y = max(0, orig_h - crop_px)

                with st.container(
                    horizontal=True,
                    horizontal_alignment="distribute",
                    vertical_alignment="center",
                ):
                    crop_x = st.slider(
                        "X position (px)",
                        min_value=0,
                        max_value=max_x + 1,  # epsilon so slider works when max_x=0
                        value=0,
                        step=1,
                        disabled=not use_crop or max_x == 0,
                    )

                    crop_y = st.slider(
                        "Y position (px)",
                        min_value=0,
                        max_value=max_y + 1,  # epsilon so slider works when max_y=0
                        value=0,
                        step=1,
                        disabled=not use_crop or max_y == 0,
                    )

                st.caption(
                    f"Crop: {crop_px}×{crop_px} px "
                    f"({crop_size_nm:.1f} nm) at x={crop_x}, y={crop_y} · "
                    f"rotation={rotation_deg:.1f}°"
                )

                crop_timing = st.radio(
                    "Apply crop",
                    ["after processing", "before processing"],
                    disabled=not use_crop,
                    help="Crop the raw image before running the pipeline, "
                         "or run the pipeline on the full image and crop the result.",
                )

            if use_crop:
                x0 = int(crop_x)
                y0 = int(crop_y)
                x1 = x0 + int(crop_w)
                y1 = y0 + int(crop_h)

                # The final image is cropped in both cases, so the physical
                # scan size reflects the crop regardless of timing.
                scan_size_um = int(crop_w) * (px_to_nm / 1e3)

                if crop_timing == "before processing":
                    # Rotate the raw image, then crop, then process.
                    raw_rot = rotate_for_crop(orig_image_nm, rotation_deg, order=1)
                    image_nm = raw_rot[y0:y1, x0:x1]

                crop_meta = {
                    "enabled": True,
                    "timing": crop_timing,
                    "rotation_deg": float(rotation_deg),
                    "x_px": x0,
                    "y_px": y0,
                    "width_px": int(crop_w),
                    "height_px": int(crop_h),
                    "original_width_px": int(orig_w),
                    "original_height_px": int(orig_h),
                }

            else:
                crop_timing = "before processing"
                crop_meta = {
                    "enabled": False,
                    "timing": crop_timing,
                    "rotation_deg": 0.0,
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
                    use_gaussian = st.toggle("Gaussian", value=False)
                    gaussian_sigma = st.number_input(
                        "σ (px)",
                        0.3,
                        5.0,
                        0.8,
                        0.1,
                        disabled=not use_gaussian,
                    )

                with col2:
                    use_median = st.toggle("Median", value=False)
                    median_size = st.number_input(
                        "Kernel (px)",
                        2,
                        9,
                        3,
                        1,
                        disabled=not use_median,
                    )

            # Statistics used for slider ranges. Base them on the rotated crop
            # region so the Z / mask sliders track what you'll actually see,
            # regardless of crop timing. nanpercentile ignores blank corners.
            if use_crop:
                if crop_timing == "before processing":
                    stat_source = image_nm  # already rotated + cropped
                else:
                    raw_rot = rotate_for_crop(orig_image_nm, rotation_deg, order=1)
                    stat_source = raw_rot[y0:y1, x0:x1]
            else:
                stat_source = image_nm

            p5, p95 = np.nanpercentile(stat_source, [5, 95])
            slider_max = 1.2 * (p95 - p5)

            with st.popover("Processing settings"):
                row_poly_degree = st.number_input(
                    "Row polynomial degree",
                    min_value=0,
                    max_value=5,
                    value=DEFAULT_ROW_POLY_DEGREE,
                    step=1,
                    help="Used during background-based row polynomial alignment.",
                )

                background_poly_degree = st.number_input(
                    "Background polynomial degree",
                    min_value=0,
                    max_value=6,
                    value=DEFAULT_BACKGROUND_POLY_DEGREE,
                    step=1,
                    help="Degree for masked 2D polynomial background subtraction.",
                )

                outlier_mad = st.number_input(
                    "Outlier MAD factor",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    help="Set to 0 to disable MAD clipping.",
                )

                iterations = st.number_input(
                    "Number of processing iterations",
                    min_value=1,
                    value=1,
                    step=1,
                )

                override_slider_max = st.toggle("Override slider max", value=False)

                scar_removal = st.selectbox(
                    "Scar removal",
                    ["none", "default", "aggressive"],
                    index=1,
                    help=(
                        "Apply post-processing scar removal algorithm. "
                    ),)

                if override_slider_max:
                    slider_max = st.number_input(
                        "Slider max (nm)",
                        min_value=0.1,
                        value=float(slider_max),
                        step=0.5,
                    )

        mask_thresh = st.slider(
            "Mask threshold (nm)",
            0.0,
            slider_max+0.5,  # add small epsilon to allow max value when slider_max=0
            slider_max,
            step=0.02,
            help="Pixels below this are background for polynomial fitting.",
        )

        # ── Display range ─────────────────────────────────────────────────────
        st.header("🎨 Display range")

        with st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        ):
            use_fixed = st.toggle("Fixed Z window", value=False)

            if use_fixed:
                window_size = st.number_input(
                    "Window size (nm)",
                    min_value=0.0,
                    value=2.8,
                    step=0.05,
                )

        if use_fixed:
            zmin = st.slider("Z baseline (nm)", -2.0, slider_max, 0.0, 0.05)
            zmax = zmin + window_size
        else:
            zmin, zmax = st.slider(
                "Z range (nm)",
                -2.0,
                slider_max,
                (0.0, slider_max),
                0.05,
            )

        set_colorbar_at_0 = st.checkbox(
            "Show processed colorbar relative to display minimum",
            value=True,
            help="Display processed colorbar as 0 → zmax-zmin.",
        )

        # ── Scale bar options ─────────────────────────────────────────────────
        with st.popover("📏 Scale bar options"):
            sb_location = st.selectbox(
                "Location",
                ["lower right", "lower left", "upper right", "upper left"],
                index=0,
            )

            sb_color = st.color_picker("Bar & text color", "#FFFFFF")
            sb_frame = st.checkbox("Show background box", value=False)

        options = list(mpl.colormaps) + ["gwy_sky"]

        colormap = st.selectbox(
            "Colormap",
            options,
            index=options.index("gwy_sky"),
            help="Colormap for raw and processed images.",
        )

    # ── Validation ────────────────────────────────────────────────────────────
    if zmin >= zmax:
        st.error("Z min must be less than Z max.")
        st.stop()

    # ── Build config and run ──────────────────────────────────────────────────
    cfg = PipelineConfig(
        mask_threshold_nm=mask_thresh,
        row_poly_degree=row_poly_degree,
        background_poly_degree=background_poly_degree,
        iterations=iterations,
        channel=channel,
        use_gaussian=use_gaussian,
        gaussian_sigma=float(gaussian_sigma),
        use_median=use_median,
        scar_removal=scar_removal,
        median_size=int(median_size),
        outlier_mad_factor=outlier_mad,
        zmin=zmin,
        zmax=zmax,
        set_colorbar_at_0=set_colorbar_at_0,
        scalebar_location=sb_location,
        scalebar_color=sb_color,
        scalebar_frameon=sb_frame,
    )

    processed, mask = run_pipeline(image_nm, cfg.to_json())

    # Crop after processing, if requested. The pipeline ran on the full image,
    # so rotate the processed result, the mask, and the raw image about the
    # centre (just before cropping), then slice all three identically.
    if crop_meta["enabled"] and crop_meta["timing"] == "after processing":
        ang = crop_meta["rotation_deg"]

        cy0 = crop_meta["y_px"]
        cx0 = crop_meta["x_px"]
        cy1 = cy0 + crop_meta["height_px"]
        cx1 = cx0 + crop_meta["width_px"]

        # Rotate (height data bilinear + NaN fill, mask nearest + 0 fill).
        processed = rotate_for_crop(processed, ang, order=1)
        image_nm = rotate_for_crop(image_nm, ang, order=1)
        mask = rotate_for_crop(mask.astype(float), ang, order=0, cval=0.0) > 0.5

        processed = processed[cy0:cy1, cx0:cx1]
        image_nm = image_nm[cy0:cy1, cx0:cx1]
        mask = mask[cy0:cy1, cx0:cx1]

    extent, xy_unit = get_display_extent(scan_size_um)

    # ── Version and save path ─────────────────────────────────────────────────
    v_num = st.number_input("Save version number", min_value=0, value=0, step=1)

    save_path = f"{save_stem}_v{v_num}.png"
    save_path_svg = f"{save_stem}_v{v_num}.svg"
    meta_path = f"{save_stem}_v{v_num}_metadata.json"

    if os.path.isfile(save_path):
        st.warning(f"Existing PNG found → `{save_path}`")

        with st.container(
            horizontal=True,
            horizontal_alignment="distribute",
            vertical_alignment="center",
        ):
            redo = st.toggle("Reprocess JPK file", value=False)

        if not redo:
            st.info(f"Displaying saved PNG → `{save_path}`")
            st.image(plt.imread(save_path))

            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    st.json(f.read())

            st.stop()

    # ── Render ────────────────────────────────────────────────────────────────
    processed_fig = None

    show_processed_only = st.toggle("Show processed image only", value=False)

    if show_processed_only:
        processed_fig = make_processed_only_fig(
            processed,
            extent,
            cfg,
            xy_unit,
            colormap=colormap,
        )

        st.pyplot(processed_fig)

    else:
        fig = make_figure(
            image_nm,
            processed,
            mask,
            extent,
            cfg,
            selected_file,
            xy_unit,
            colormap=colormap,
        )

        st.pyplot(fig)
        plt.close(fig)

    # ── Save ──────────────────────────────────────────────────────────────────
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        saved = False

        if st.button("💾 Save Processed PNG", use_container_width=True):
            if processed_fig is None:
                processed_fig = make_processed_only_fig(
                    processed,
                    extent,
                    cfg,
                    xy_unit,
                    colormap=colormap,
                )

            processed_fig.savefig(save_path,
                                  dpi=600,
                                  bbox_inches="tight",
                                  transparent=True,
                                  facecolor="none",
                                  edgecolor="none",
                                  )
            plt.close(processed_fig)

            saved = save_path

        if st.button("💾 Save Processed SVG", use_container_width=True):
            if processed_fig is None:
                processed_fig = make_processed_only_fig(
                    processed,
                    extent,
                    cfg,
                    xy_unit,
                    colormap=colormap,
                )

            processed_fig.savefig(save_path_svg,
                                  dpi=600,
                                  bbox_inches="tight",
                                  transparent=True,
                                  facecolor="none",
                                  edgecolor="none",
                                  )
            plt.close(processed_fig)

            saved = save_path_svg

        if saved:
            meta = {
                "source_file": filepath,
                "save_path": saved,
                "pipeline": json.loads(cfg.to_json()),
                "crop": crop_meta,
                "scan_size_um": scan_size_um,
                "px_to_nm": px_to_nm,
            }

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            st.success(f"Saved → `{saved}`")
            st.success(f"Metadata → `{meta_path}`")

            with st.expander("View saved metadata"):
                st.json(meta)