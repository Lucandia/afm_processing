import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from AFMReader.jpk import load_jpk  # your existing loader


# =============================================================================
# CONFIG
# =============================================================================
CHANNEL = "height_trace"


# =============================================================================
# SAFE NUMERIC HELPERS
# =============================================================================
def nanmin_safe(x: np.ndarray) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.min()) if vals.size else 0.0


def nanmax_safe(x: np.ndarray) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else 0.0


def nanpercentile_safe(x: np.ndarray, q: float) -> float:
    vals = np.asarray(x, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, q)) if vals.size else 0.0


def _interp_nan_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    idx = np.arange(y.size)
    good = np.isfinite(y)

    if good.sum() == 0:
        return np.zeros_like(y)
    if good.sum() == 1:
        return np.full_like(y, y[good][0])

    out = y.copy()
    out[~good] = np.interp(idx[~good], idx[good], y[good])
    return out


# =============================================================================
# PROCESSING (safer, more "Gwyddion-ish")
# =============================================================================
def plane_level_2d(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Subtract best-fit plane z = ax + by + c (global 2D leveling).
    Much safer than fitting high-order polynomials per scan line.
    """
    z = np.asarray(img, dtype=float)
    yy, xx = np.indices(z.shape)

    valid = np.isfinite(z)
    if mask is not None:
        valid &= mask

    if valid.sum() < 3:
        return z.copy()

    A = np.column_stack([xx[valid], yy[valid], np.ones(valid.sum())])
    coeffs, *_ = np.linalg.lstsq(A, z[valid], rcond=None)

    plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
    out = z - plane
    out[~np.isfinite(z)] = np.nan
    return out


def make_background_mask(
    img: np.ndarray,
    threshold_nm: float | None = None,
    percentile: float = 35.0,
    min_points_per_row: int = 16,
) -> tuple[np.ndarray, float]:
    """
    Background mask:
      - if threshold_nm is set: background = pixels <= threshold_nm
      - else: background = lowest `percentile`% pixels (globally), with row fallback
    """
    z = np.asarray(img, dtype=float)
    valid = np.isfinite(z)
    if not valid.any():
        return np.zeros_like(z, dtype=bool), 0.0

    threshold = float(threshold_nm) if threshold_nm is not None else nanpercentile_safe(z, percentile)
    mask = valid & (z <= threshold)

    min_points = min(min_points_per_row, max(1, z.shape[1] // 20))
    for i, row in enumerate(z):
        if mask[i].sum() >= min_points:
            continue
        row_valid = np.isfinite(row)
        if not row_valid.any():
            continue
        row_thr = float(np.percentile(row[row_valid], percentile))
        mask[i] = row_valid & (row <= row_thr)

    return mask, threshold


def align_rows(img: np.ndarray, mask: np.ndarray | None = None, degree: int = 0) -> np.ndarray:
    """
    Row alignment:
      degree = 0 -> subtract row offset (recommended default)
      degree = 1 -> subtract row linear trend
      degree > 1 -> forced to 1 intentionally (per-row higher order is too destructive)
    """
    z = np.asarray(img, dtype=float).copy()
    x = np.arange(z.shape[1], dtype=float)
    x0 = x - x.mean()

    use_linear = degree >= 1

    for i, row in enumerate(z):
        valid = np.isfinite(row)
        if mask is not None:
            valid &= mask[i]

        # fall back to all finite pixels if mask gives too few points
        if valid.sum() < (2 if use_linear else 1):
            valid = np.isfinite(row)

        if valid.sum() == 0:
            continue

        if use_linear and valid.sum() >= 2:
            coeffs = np.polyfit(x0[valid], row[valid], deg=1)
            baseline = np.polyval(coeffs, x0)
        else:
            baseline = np.nanmedian(row[valid])

        z[i] = row - baseline

    return z


def correct_scar_rows(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    threshold_factor: float = 5.0,
) -> np.ndarray:
    """
    Detect anomalous row baselines and correct only the row offset (NOT overwriting full rows).
    Safer than replacing whole scan lines.
    """
    z = np.asarray(img, dtype=float).copy()

    row_base = np.full(z.shape[0], np.nan)
    min_points = max(8, z.shape[1] // 40)

    for i, row in enumerate(z):
        valid = np.isfinite(row)
        if mask is not None:
            valid &= mask[i]
        if valid.sum() < min_points:
            valid = np.isfinite(row)
        if valid.any():
            row_base[i] = np.nanmedian(row[valid])

    row_base = _interp_nan_1d(row_base)
    d = np.diff(row_base)
    if d.size == 0:
        return z

    med = np.median(d)
    mad = np.median(np.abs(d - med))
    if not np.isfinite(mad) or mad == 0:
        return z

    sigma = 1.4826 * mad
    bad_rows = np.where(np.abs(d - med) > threshold_factor * sigma)[0] + 1
    if bad_rows.size == 0:
        return z

    corrected_base = row_base.copy()
    good = np.ones_like(corrected_base, dtype=bool)
    good[bad_rows] = False
    idx = np.arange(corrected_base.size)

    if good.sum() >= 2:
        corrected_base[~good] = np.interp(idx[~good], idx[good], corrected_base[good])
    else:
        corrected_base[:] = corrected_base[good][0] if good.sum() == 1 else corrected_base[0]

    # Only shift row offsets
    delta = row_base - corrected_base
    z -= delta[:, None]
    return z


def full_pipeline(
    image_nm: np.ndarray,
    mask_threshold_nm: float | None = None,
    mask_percentile: float = 35.0,
    row_degree: int = 0,
    iterations: int = 1,
):
    """
    Safer AFM pipeline:
      1) global 2D plane level
      2) build background mask
      3) plane level again on background only
      4) row alignment (offset or linear)
      5) scar row correction (offset-only)
      6) shift min to 0 (display convenience)

    Tip: iterations > 3 often starts flattening real signal.
    """
    z = np.asarray(image_nm, dtype=float).copy()
    if z.ndim != 2:
        raise ValueError("Expected a 2-D AFM height image.")

    # initial coarse leveling (for stable masking)
    z = plane_level_2d(z)
    z -= nanmin_safe(z)
    coarse = z.copy()

    iterations = max(1, int(iterations))
    row_degree = 1 if row_degree >= 1 else 0

    last_threshold = 0.0
    mask = np.zeros_like(z, dtype=bool)

    for _ in range(iterations):
        mask, last_threshold = make_background_mask(
            z,
            threshold_nm=mask_threshold_nm,
            percentile=mask_percentile,
        )

        z = plane_level_2d(z, mask=mask)
        z = align_rows(z, mask=mask, degree=row_degree)
        z = correct_scar_rows(z, mask=mask, threshold_factor=5.0)
        z -= nanmin_safe(z)

    return z, mask, coarse, last_threshold


# =============================================================================
# FILE I/O + UNITS
# =============================================================================
def _infer_px_size_m(px_size_raw: float) -> float:
    """
    Heuristic conversion of pixel size to meters/pixel.

    Cases we try to handle:
      - px_size already meters/pixel  ~ 1e-11..1e-6
      - px_size nm/pixel             ~ 0.05..500
      - px_size µm/pixel             ~ 1e-4..1e2 (rare, but possible)

    If this guess is wrong for your reader, set it explicitly here.
    """
    v = float(px_size_raw)

    # typical meters/pixel for AFM: ~1e-11 to 1e-6
    if 0 < v < 1e-5:
        return v  # meters/pixel

    # typical nm/pixel: ~0.05 to 500
    if 0.01 <= v <= 1e4:
        return v * 1e-9  # nm/pixel -> m/pixel

    # fallback: assume meters/pixel
    return v


@st.cache_data(show_spinner=False)
def load_file(filepath: str):
    """
    Load a JPK AFM image and convert height to nanometres.

    Height:
      - assume meters if z-range < 1e-3 (typical AFM stored in SI)
      - else assume already nm

    Pixel size:
      - inferred to meters/pixel via heuristic (see _infer_px_size_m)
    """
    image, px_size_raw = load_jpk(file_path=filepath, channel=CHANNEL)
    image = np.asarray(image, dtype=float)

    z_range = nanmax_safe(image) - nanmin_safe(image)
    image_nm = image * 1e9 if z_range < 1e-3 else image.copy()

    px_size_m = _infer_px_size_m(px_size_raw)
    scan_size_um = image.shape[1] * px_size_m * 1e6
    return image_nm, scan_size_um


def get_display_extent(scan_size_um: float):
    """
    If scan width >= 1 µm -> axes in µm
    else -> axes in nm
    """
    if scan_size_um >= 1.0:
        return [0.0, scan_size_um, 0.0, scan_size_um], "µm"
    scan_size_nm = scan_size_um * 1000.0
    return [0.0, scan_size_nm, 0.0, scan_size_nm], "nm"


@st.cache_data(show_spinner=False)
def get_coarse_max_for_sliders(filepath: str) -> float:
    image_nm, _ = load_file(filepath)
    coarse = plane_level_2d(image_nm)
    coarse -= nanmin_safe(coarse)
    # robust max so one outlier pixel doesn't ruin slider range
    return max(1.0, nanpercentile_safe(coarse, 99.5))


@st.cache_data(show_spinner=False)
def run_pipeline(
    filepath: str,
    mask_mode: str,
    mask_threshold_nm: float,
    mask_percentile: float,
    row_degree: int,
    iterations: int,
):
    image_nm, scan_size_um = load_file(filepath)

    thr = mask_threshold_nm if mask_mode == "fixed" else None

    aligned, mask, coarse, used_threshold = full_pipeline(
        image_nm,
        mask_threshold_nm=thr,
        mask_percentile=mask_percentile,
        row_degree=row_degree,
        iterations=iterations,
    )

    coarse_max = nanpercentile_safe(coarse, 99.5)
    return image_nm, aligned, mask, scan_size_um, coarse_max, used_threshold


# =============================================================================
# PLOTTING
# =============================================================================
def _prepare_processed_display(
    aligned: np.ndarray,
    zmin: float,
    zmax: float,
    relative_to_display_min: bool = False,
):
    disp = np.clip(aligned, zmin, zmax)
    if relative_to_display_min:
        return disp - zmin, 0.0, zmax - zmin, "Relative height (nm)"
    return disp, zmin, zmax, "Height (nm)"


def make_figure(
    image_nm,
    aligned,
    mask,
    extent,
    zmin,
    zmax,
    mask_label_value,
    filename,
    relative_to_display_min=False,
    xy_unit="µm",
    cmap="afmhot",
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(filename, fontsize=10)

    # Panel 1: Raw (robust display)
    raw_vmin = nanpercentile_safe(image_nm, 1.0)
    raw_vmax = nanpercentile_safe(image_nm, 99.0)

    im_raw = axes[0].imshow(
        image_nm, cmap=cmap, origin="lower", extent=extent,
        interpolation="nearest", vmin=raw_vmin, vmax=raw_vmax
    )
    axes[0].set_title("Raw height (robust display)")
    axes[0].set_xlabel(f"X ({xy_unit})")
    axes[0].set_ylabel(f"Y ({xy_unit})")
    cbar_raw = fig.colorbar(im_raw, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_raw.set_label("Height (nm)")
    cbar_raw.formatter.set_scientific(False)
    cbar_raw.formatter.set_useOffset(False)
    cbar_raw.update_ticks()

    # Panel 2: Mask
    axes[1].imshow(
        mask.astype(float), cmap="gray", origin="lower", extent=extent,
        interpolation="nearest", vmin=0, vmax=1
    )
    axes[1].set_title(
        f"Background mask (threshold ~ {mask_label_value:.2f} nm)\nwhite = background · black = features"
    )
    axes[1].set_xlabel(f"X ({xy_unit})")
    axes[1].set_ylabel(f"Y ({xy_unit})")

    # Panel 3: Processed
    disp, vmin_disp, vmax_disp, cbar_label = _prepare_processed_display(
        aligned, zmin, zmax, relative_to_display_min=relative_to_display_min
    )
    im_al = axes[2].imshow(
        disp, cmap=cmap, origin="lower", extent=extent,
        interpolation="nearest", vmin=vmin_disp, vmax=vmax_disp
    )
    axes[2].set_title("Processed height")
    axes[2].set_xlabel(f"X ({xy_unit})")
    axes[2].set_ylabel(f"Y ({xy_unit})")
    cbar_al = fig.colorbar(im_al, ax=axes[2], fraction=0.046, pad=0.04)
    cbar_al.set_label(cbar_label)
    cbar_al.formatter.set_scientific(False)
    cbar_al.formatter.set_useOffset(False)
    cbar_al.update_ticks()

    plt.tight_layout()
    return fig


def make_processed_only_fig(
    aligned,
    extent,
    zmin,
    zmax,
    relative_to_display_min=False,
    xy_unit="µm",
    cmap="afmhot",
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    disp, vmin_disp, vmax_disp, cbar_label = _prepare_processed_display(
        aligned, zmin, zmax, relative_to_display_min=relative_to_display_min
    )

    im = ax.imshow(
        disp, cmap=cmap, origin="lower", extent=extent,
        interpolation="nearest", vmin=vmin_disp, vmax=vmax_disp
    )
    ax.set_xlabel(f"X ({xy_unit})")
    ax.set_ylabel(f"Y ({xy_unit})")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.formatter.set_scientific(False)
    cbar.formatter.set_useOffset(False)
    cbar.update_ticks()

    plt.tight_layout()
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================
st.set_page_config(page_title="AFM Viewer", layout="wide", initial_sidebar_state="expanded")
st.title("🔬 AFM JPK Image Viewer (safer processing)")

with st.sidebar:
    st.header("📂 File selection")
    folder = st.text_input("Folder path", value="", placeholder="/path/to/jpk/files")

    if not folder:
        st.info("Enter a folder path to start.")
        st.stop()

    if not os.path.isdir(folder):
        st.error("Path does not exist.")
        st.stop()

    # Collect .jpk and imaging-*/*.jpk
    jpk_files = []
    all_f = sorted(os.listdir(folder))

    for f in all_f:
        fp = os.path.join(folder, f)
        if f.endswith(".jpk") and os.path.isfile(fp):
            jpk_files.append(f)
        elif os.path.isdir(fp) and f.startswith("imaging-"):
            for subf in sorted(os.listdir(fp)):
                if subf.endswith(".jpk"):
                    jpk_files.append(os.path.join(f, subf))

    if not jpk_files:
        st.warning("No .jpk files found in this folder.")
        st.stop()

    jpk_files = sorted(jpk_files)
    # selected_file = st.selectbox("Select file", options=jpk_files, index=0)
    selected_file = st.select_slider("Select file", options=jpk_files,
                                     value=jpk_files[0])
    st.write(f"Selected file: `{selected_file}`")
    filepath = os.path.join(folder, selected_file)

    # output path
    base_out = os.path.splitext(os.path.join(folder, os.path.basename(filepath)))[0]

    # Determine slider max from coarse plane-leveled data
    with st.spinner("Reading file & estimating ranges…"):
        slider_max_val = round(float(get_coarse_max_for_sliders(filepath)), 2) + 1.0

    st.divider()
    st.header("⚙️ Processing")

    # Mask controls
    mask_mode = st.radio(
        "Background mask mode",
        options=["auto (percentile)", "fixed threshold"],
        index=0,
        help="Auto chooses background as the lowest percentile of heights (with row fallback). "
             "Fixed uses a nm threshold after initial leveling."
    )
    if mask_mode == "fixed threshold":
        mask_threshold_nm = st.slider(
            "Mask threshold (nm)",
            min_value=0.0, max_value=slider_max_val,
            value=min(2.0, slider_max_val),
            step=0.05
        )
        mask_percentile = 35.0
        mask_mode_key = "fixed"
    else:
        mask_percentile = st.slider(
            "Background percentile (%)",
            min_value=0.0, max_value=100.0,
            value=35.0, step=1.0,
            help="Background = lowest X% heights (after coarse leveling)."
        )
        mask_threshold_nm = 0.0
        mask_mode_key = "auto"

    # Row alignment strength
    row_degree = st.radio(
        "Row alignment",
        options=["offset only (recommended)", "linear per row (stronger)"],
        index=0,
        help="Offset only is safest. Linear can help gentle bow per line, but can flatten real signal if abused."
    )
    row_degree_val = 0 if row_degree.startswith("offset") else 1

    # Iterations
    iters_on = st.toggle(
        "Iterative refinement (1–3 recommended)",
        value=False,
        help="Repeats mask+level+row align. More than 3 iterations often starts over-flattening."
    )
    iterations = 1
    if iters_on:
        iterations = st.number_input("Iterations", min_value=1, max_value=5, value=2, step=1)

    st.divider()
    st.header("🎨 Display range")

    use_fixed_range = st.toggle("Use fixed Z range", value=False)
    if use_fixed_range:
        window_size = st.number_input(
            "Color window size (nm)",
            min_value=0.2, max_value=slider_max_val,
            value=min(3.0, slider_max_val),
            step=0.05,
        )
        zmin = st.slider(
            "Baseline (nm)",
            min_value=0.0, max_value=slider_max_val,
            value=0.0, step=0.05
        )
        zmax = zmin + float(window_size)
    else:
        zmin, zmax = st.slider(
            "Z range (nm)",
            min_value=0.0, max_value=slider_max_val,
            value=(0.0, slider_max_val),
            step=0.05
        )

    relative_to_display_min = st.checkbox(
        "Show processed colorbar relative to display minimum",
        value=True,
        help="Displays processed colorbar as 0 → (zmax-zmin) without modifying data."
    )

    # cmap = st.selectbox("Colormap", options=["afmhot", "viridis", "gray", "inferno", "magma"], index=0)
    cmap = "afmhot"

    st.divider()
    st.header("💾 Export")
    v_num = st.number_input("Save version number", min_value=0, value=0, step=1)
    save_path = f"{base_out}_v{v_num}.png"
    dpi = st.number_input("Export DPI (PNG)", min_value=150, max_value=1200, value=600, step=50)

# Validation
if zmin >= zmax:
    st.error("Z min must be less than Z max.")
    st.stop()

# If saved image exists: optionally reuse
if os.path.isfile(save_path):
    st.warning(f"Found existing PNG → `{save_path}`")
    redo = st.toggle("Reprocess JPK file anyway", value=False)
    if not redo:
        img = plt.imread(save_path)
        st.image(img, caption=os.path.basename(save_path))
        st.stop()

# Run pipeline
with st.spinner("Processing…"):
    image_nm, aligned, mask, scan_size_um, _, used_threshold = run_pipeline(
        filepath=filepath,
        mask_mode=mask_mode_key,
        mask_threshold_nm=float(mask_threshold_nm),
        mask_percentile=float(mask_percentile),
        row_degree=int(row_degree_val),
        iterations=int(iterations),
    )

extent, xy_unit = get_display_extent(scan_size_um)

# Render
show_processed_only = st.toggle("Show processed image only", value=False)

if show_processed_only:
    fig = make_processed_only_fig(
        aligned, extent, zmin, zmax,
        relative_to_display_min=relative_to_display_min,
        xy_unit=xy_unit, cmap=cmap
    )
    st.pyplot(fig)
else:
    fig = make_figure(
        image_nm, aligned, mask, extent,
        zmin, zmax, used_threshold,
        os.path.basename(filepath),
        relative_to_display_min=relative_to_display_min,
        xy_unit=xy_unit, cmap=cmap
    )
    st.pyplot(fig)

plt.close(fig)

# Save
if st.button("💾 Save PNG", use_container_width=True):
    fig_out = make_processed_only_fig(
        aligned, extent, zmin, zmax,
        relative_to_display_min=relative_to_display_min,
        xy_unit=xy_unit, cmap=cmap
    )
    fig_out.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig_out)
    st.success(f"Saved → `{save_path}`")
    st.image(plt.imread(save_path), caption=os.path.basename(save_path))