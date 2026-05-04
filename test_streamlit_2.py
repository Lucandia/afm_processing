import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import streamlit as st
from matplotlib_scalebar.scalebar import ScaleBar

from AFMReader.jpk import load_jpk


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CHANNEL_TRACE   = "height_trace"
CHANNEL_RETRACE = "height_retrace"
POLY_DEGREE     = 3


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    mask_threshold_nm: float = 2.0
    poly_degree: int = POLY_DEGREE
    iterations: int = 1
    average_retrace: bool = False
    use_gaussian: bool = False
    gaussian_sigma: float = 0.8
    use_median: bool = False
    median_size: int = 3
    outlier_mad_factor: float = 5.0
    zmin: float = 0.0
    zmax: float = 5.0
    set_colorbar_at_0: bool = True
    scalebar_location: str = "lower right"
    scalebar_color: str = "white"
    scalebar_frameon: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════
def nanmin_safe(x: np.ndarray) -> float:
    return float(np.nanmin(x))

def nanmax_safe(x: np.ndarray) -> float:
    return float(np.nanmax(x))


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


def row_align_poly(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    degree: int = 3
) -> np.ndarray:
    """
    Polynomial row-by-row alignment.
    If mask provided (True = background), fits only on background pixels.
    Falls back to all finite pixels if too few background points exist.
    """
    aligned = img.copy().astype(float)
    x       = np.arange(aligned.shape[1])

    for i in range(aligned.shape[0]):
        row = aligned[i]
        if mask is not None:
            m     = mask[i] & np.isfinite(row)
            fit_x = x[m]   if m.sum() > degree + 1 else x[np.isfinite(row)]
            fit_y = row[m] if m.sum() > degree + 1 else row[np.isfinite(row)]
        else:
            fit_x = x[np.isfinite(row)]
            fit_y = row[np.isfinite(row)]

        if len(fit_x) < degree + 1:
            continue
        coeffs     = np.polyfit(fit_x, fit_y, deg=degree)
        aligned[i] -= np.polyval(coeffs, x)

    return aligned


def remove_scars(img: np.ndarray, threshold_factor: float = 3.0) -> np.ndarray:
    """
    Replace scan lines whose median deviates from neighbours by more than
    threshold_factor × MAD of inter-row differences with linear interpolation.
    """
    corrected   = img.copy().astype(float)
    row_medians = np.nanmedian(corrected, axis=1)
    diff        = np.abs(np.diff(row_medians))
    mad         = np.nanmedian(diff)

    if not np.isfinite(mad) or mad == 0:
        return corrected

    scar_rows = np.where(diff > threshold_factor * mad)[0] + 1
    scar_set  = set(scar_rows.tolist())

    for r in scar_rows:
        above, below = r - 1, r + 1
        while above in scar_set and above > 0:             above -= 1
        while below in scar_set and below < img.shape[0]:  below += 1

        if above >= 0 and below < img.shape[0]:
            corrected[r] = 0.5 * (corrected[above] + corrected[below])
        elif above >= 0:
            corrected[r] = corrected[above]
        elif below < img.shape[0]:
            corrected[r] = corrected[below]

    return corrected


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
    """
    Full AFM processing pipeline.

    Steps per iteration:
      0. Outlier pixel clamping
      1. Conservative scar removal (pre-pass)
      2. 2D plane subtraction (no mask on first pass, masked on subsequent)
      3. Coarse linear row alignment
      4. Background mask
      5. Scar correction
      6. Masked polynomial row alignment
      7. Floor shift to 0
      8. Final scar pass

    After all iterations: optional Gaussian / median smoothing.

    Returns
    -------
    aligned : processed height array (nm)
    mask    : background boolean mask from last iteration
    coarse  : coarse-aligned array from last iteration (for slider range)
    """
    aligned = image_nm.copy()
    mask    = np.ones(image_nm.shape, dtype=bool)

    for i in range(cfg.iterations):
        # Step 0 — outlier clamping
        aligned = clamp_outliers(aligned, cfg.outlier_mad_factor)

        # Step 1 — conservative scar pre-pass
        aligned = remove_scars(aligned, threshold_factor=5.0)

        # Step 2 — 2D plane subtraction (use mask from previous iteration)
        plane_mask = None if i == 0 else mask
        aligned    = subtract_2d_plane(aligned, mask=plane_mask)

        # Step 3 — coarse linear row alignment
        coarse  = row_align_poly(aligned, mask=None, degree=1)
        coarse -= nanmin_safe(coarse)

        # Step 4 — background mask
        mask = coarse < cfg.mask_threshold_nm

        # Step 5 — scar correction
        scar_corrected = remove_scars(coarse)

        # Step 6 — masked polynomial row alignment
        aligned  = row_align_poly(scar_corrected, mask=mask, degree=cfg.poly_degree)
        aligned -= nanmin_safe(aligned)

        # Step 7 — final scar pass
        aligned = remove_scars(aligned)

    # Post-pipeline smoothing (applied once, after all iterations)
    aligned = apply_smoothing(
        aligned,
        cfg.use_gaussian, cfg.gaussian_sigma,
        cfg.use_median,   cfg.median_size
    )

    return aligned, mask, coarse


# ═══════════════════════════════════════════════════════════════════════════════
# FILE I/O
# ═══════════════════════════════════════════════════════════════════════════════
def _to_nm(image: np.ndarray) -> np.ndarray:
    """Convert from SI metres to nm if the array looks like it's in metres."""
    z_range = np.nanmax(image) - np.nanmin(image)
    return image * 1e9 if z_range < 1e-3 else image.copy().astype(float)


@st.cache_data(show_spinner="Loading file…")
def load_file(
    filepath: str,
    average_retrace: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Load a JPK file.  If average_retrace=True and height_retrace exists,
    return pixel-wise average of trace and retrace (reduces random noise).

    Returns
    -------
    image_nm     : height array in nm
    scan_size_um : physical scan width in µm
    px_to_nm     : nanometres per pixel (used for scale bar)
    """
    trace, px_to_nm = load_jpk(file_path=filepath, channel=CHANNEL_TRACE)
    image_nm        = _to_nm(trace)

    if average_retrace:
        try:
            retrace, _ = load_jpk(file_path=filepath, channel=CHANNEL_RETRACE)
            retrace_nm = _to_nm(retrace)
            image_nm   = 0.5 * (image_nm + retrace_nm)
        except Exception:
            st.warning("Retrace channel not found — using trace only.")

    scan_size_um = image_nm.shape[1] * (px_to_nm * 1e6)
    return image_nm, scan_size_um, float(px_to_nm)


@st.cache_data(show_spinner="Processing…")
def run_pipeline(
    filepath: str,
    cfg_json: str           # serialised PipelineConfig (hashable for cache)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    cfg                       = PipelineConfig(**json.loads(cfg_json))
    image_nm, scan_size_um, px_to_nm = load_file(filepath, cfg.average_retrace)
    aligned, mask, coarse     = full_pipeline(image_nm, cfg)
    return image_nm, aligned, mask, scan_size_um, nanmax_safe(coarse), px_to_nm


# ═══════════════════════════════════════════════════════════════════════════════
# SCALE BAR HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def add_scale_bar(
    ax: plt.Axes,
    px_to_nm: float,
    location: str  = "lower right",
    color: str     = "white",
    frameon: bool  = False
) -> None:
    """
    Add a matplotlib-scalebar to ax.
    px_to_nm : nanometres per pixel → convert to metres for ScaleBar.
    """
    dx_m = px_to_nm * 1e-8          # metres per pixel
    sb   = ScaleBar(
        dx_m, "um",
        length_fraction=0.25,
        location=location,
        color=color,
        box_color="black",
        box_alpha=0.0 if not frameon else 0.4,
        frameon=frameon,
        font_properties={"size": 10},
    )
    ax.add_artist(sb)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
def get_display_extent(scan_size_um: float) -> Tuple[list, str]:
    if scan_size_um >= 1.0:
        return [0, scan_size_um, 0, scan_size_um], "µm"
    else:
        return [0, scan_size_um * 1e3, 0, scan_size_um * 1e3], "nm"


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
    px_to_nm:  float,
    xy_unit:   str = "µm"
) -> plt.Figure:
    """3-panel diagnostic figure: Raw | Mask | Processed."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(filename, fontsize=10)

    raw_vmin = nanmin_safe(image_nm)
    raw_vmax = nanmax_safe(image_nm)

    # Panel 1 — Raw
    im_raw = axes[0].imshow(image_nm, cmap="afmhot", origin="upper",
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
                           cmap="afmhot", origin="upper", extent=extent,
                           interpolation="nearest",
                           vmin=cfg.zmin, vmax=cfg.zmax)
    axes[2].set_title("Processed height (nm)")
    axes[2].set_xlabel(f"X ({xy_unit})")
    axes[2].set_ylabel(f"Y ({xy_unit})")
    _colorbar_ticks(fig, im_al, axes[2], "Height (nm)", cfg.zmin, cfg.zmax,
                    cfg.set_colorbar_at_0)
    add_scale_bar(axes[2], px_to_nm, cfg.scalebar_location,
                  cfg.scalebar_color, cfg.scalebar_frameon)

    plt.tight_layout()
    return fig


def make_processed_only_fig(
    aligned:  np.ndarray,
    extent:   list,
    cfg:      PipelineConfig,
    px_to_nm: float,
    xy_unit:  str = "µm"
) -> plt.Figure:
    """Clean single-panel publication figure with scale bar, no axes."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.imshow(np.clip(aligned, cfg.zmin, cfg.zmax),
                   cmap="afmhot", origin="upper", extent=extent,
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

    add_scale_bar(ax, px_to_nm, cfg.scalebar_location,
                  cfg.scalebar_color, cfg.scalebar_frameon)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="AFM Viewer", layout="wide",
                   initial_sidebar_state="expanded")
st.title("🔬 AFM JPK Image Viewer")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 File selection")
    folder = st.text_input("Folder path", value="",
                           placeholder="/path/to/jpk/files")

    if not folder:
        st.info("Enter a folder path to start.")
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
        st.stop()

    selected_ind  = st.select_slider("Select file",
                                     options=range(len(jpk_files)), value=0)
    selected_file = jpk_files[selected_ind]
    st.write(f"Selected: `{selected_file}`")
    filepath  = os.path.join(folder, selected_file)
    save_stem = os.path.splitext(
                    os.path.join(folder, os.path.basename(filepath))
                )[0]

    # ── Processing ────────────────────────────────────────────────────────────
    st.header("⚙️ Processing")

    average_retrace = st.toggle(
        "Average trace + retrace",
        value=True,
        help="Load height_retrace and average with height_trace. "
             "Reduces random noise before any processing."
    )

    # with st.container(horizontal=True, horizontal_alignment="distribute",
    #                   vertical_alignment="center"):
    #     iters_on   = st.toggle("Iterative processing", value=False)
    #     iterations = st.number_input("Iterations", min_value=1, max_value=10,
    #                                  value=2, step=1,
    #                                  disabled=not iters_on) if iters_on else 1
    iterations = 1

    # Pre-load coarse max for slider ranges (mask=0, no retrace)
    with st.spinner("Loading…"):
        _cfg_probe = PipelineConfig(mask_threshold_nm=0.0,
                                    average_retrace=average_retrace)
        _, _, _, _, coarse_max, _ = run_pipeline(filepath, _cfg_probe.to_json())

    slider_max = round(float(coarse_max), 2) + 1.0

    mask_thresh = st.slider("Mask threshold (nm)", 0.0, slider_max,
                             round(slider_max / 2, 2), 0.05,
                             help="Pixels below this are background for poly fitting.")

    # poly_degree = st.select_slider("Polynomial degree",
    #                                options=[1, 2, 3, 4, 5], value=POLY_DEGREE)
    poly_degree = 1

    # st.subheader("Outlier clamping")
    # outlier_mad = st.slider("MAD factor", 2.0, 10.0, 5.0, 0.5,
    #                          help="Clip values beyond median ± N×MAD before processing.")
    outlier_mad = 10.0

    st.subheader("Smoothing (post-alignment)")
    col1, col2 = st.columns(2)
    with col1:
        use_gaussian    = st.toggle("Gaussian filter", value=False)
        gaussian_sigma  = st.number_input("σ (px)", 0.3, 5.0, 0.8, 0.1,
                                          disabled=not use_gaussian)
    with col2:
        use_median  = st.toggle("Median filter", value=False)
        median_size = st.number_input("Kernel (px)", 2, 9, 3, 1,
                                      disabled=not use_median)

    # ── Display range ─────────────────────────────────────────────────────────
    st.header("🎨 Display range")

    with st.container(horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center"):
        use_fixed = st.toggle("Fixed Z window", value=False)
        if use_fixed:
            window_size = st.number_input("Window size (nm)", 1.0,
                                          slider_max, 3.0, 0.05)

    if use_fixed:
        zmin = st.slider("Z baseline (nm)", 0.0, slider_max, 0.0, 0.05)
        zmax = zmin + window_size
    else:
        zmin, zmax = st.slider("Z range (nm)", 0.0, slider_max,
                               (0.0, slider_max), 0.05)

    set_colorbar_at_0 = st.checkbox(
        "Show processed colorbar relative to display minimum",
        value=True,
        help="Display the processed colorbar as 0 → (zmax-zmin) without changing the underlying data."
    )

    # ── Scale bar options ─────────────────────────────────────────────────────
    st.header("📏 Scale bar")
    with st.expander("Scale bar options", expanded=False):
        sb_location = st.selectbox(
            "Location",
            ["lower right", "lower left", "upper right", "upper left", "center"],
            index=0
        )
        sb_color  = st.color_picker("Bar & text color", "#FFFFFF")
        sb_frame  = st.checkbox("Show background box", value=False)

# ── Validation ────────────────────────────────────────────────────────────────
if zmin >= zmax:
    st.error("Z min must be less than Z max.")
    st.stop()

# ── Build config & run ────────────────────────────────────────────────────────
cfg = PipelineConfig(
    mask_threshold_nm = mask_thresh,
    poly_degree       = poly_degree,
    iterations        = iterations,
    average_retrace   = average_retrace,
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

image_nm, aligned, mask, scan_size_um, _, px_to_nm = run_pipeline(
    filepath, cfg.to_json()
)
extent, xy_unit = get_display_extent(scan_size_um)

# ── Version & save path ───────────────────────────────────────────────────────
v_num     = st.number_input("Save version number", min_value=0, value=0, step=1)
save_path = f"{save_stem}_v{v_num}.png"
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
    processed_fig = make_processed_only_fig(aligned, extent, cfg, px_to_nm, xy_unit)
    st.pyplot(processed_fig)
else:
    fig = make_figure(image_nm, aligned, mask, extent, cfg,
                      selected_file, px_to_nm, xy_unit)
    st.pyplot(fig)
    plt.close(fig)

# ── Save ──────────────────────────────────────────────────────────────────────
if st.button("💾 Save Processed PNG", use_container_width=True):
    if processed_fig is None:
        processed_fig = make_processed_only_fig(aligned, extent, cfg,
                                                px_to_nm, xy_unit)
    processed_fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(processed_fig)

    # Save metadata sidecar
    meta = {
        "source_file":  filepath,
        "save_path":    save_path,
        "pipeline":     json.loads(cfg.to_json()),
        "scan_size_um": scan_size_um,
        "px_to_nm":     px_to_nm,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    st.success(f"Saved → `{save_path}`")
    st.success(f"Metadata → `{meta_path}`")
    with st.expander("View saved metadata"):
        st.json(meta)
