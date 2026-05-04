import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import trim_mean

def fit_plane_subtract(img, mask=None):
    """Fit z = ax + by + c (least squares) and subtract.
    
    Args:
        img: Input image array (H, W)
        mask: Optional boolean mask (H, W), True=use for fitting
    
    Returns:
        corrected: Image with plane subtracted
        plane: Fitted plane
    """
    H, W = img.shape
    y, x = np.mgrid[0:H, 0:W]
    
    if mask is not None:
        mask = mask.astype(bool)
        if mask.shape != img.shape:
            raise ValueError("mask must have same shape as image")
        
        # Filter coordinates and values using mask
        x_masked = x[mask]
        y_masked = y[mask]
        z_masked = img[mask]
        
        if x_masked.size == 0:
            # If no valid pixels, return original image and zero plane
            return img, np.zeros_like(img)
        
        A = np.c_[x_masked, y_masked, np.ones(x_masked.size)]
        Z = z_masked
    else:
        A = np.c_[x.reshape(-1), y.reshape(-1), np.ones(H*W)]
        Z = img.reshape(-1)
    
    coeff, *_ = np.linalg.lstsq(A, Z, rcond=None)
    plane = (coeff[0]*x + coeff[1]*y + coeff[2])
    return img - plane, plane

def _row_mode_estimate(row, bins=256, mask=None):
    if mask is not None:
        row = row[mask]
    if row.size == 0:
        return 0.0
    hist, edges = np.histogram(row, bins=bins)
    i = np.argmax(hist)
    return 0.5 * (edges[i] + edges[i+1])

def _row_poly_fit(x, y, deg, mask=None):
    if mask is not None:
        x = x[mask]; y = y[mask]
    if x.size == 0:
        # fall back to 0
        return Polynomial([0.0])
    return Polynomial.fit(x, y, deg).convert()

def _trimmed_mean(a, p, mask=None):
    if mask is not None:
        a = a[mask]
    if a.size == 0:
        return 0.0
    # p is fraction to trim at each tail in [0, 0.5]; SciPy trim_mean uses total proportion
    return trim_mean(a, 2*p)

def _trimmed_mean_of_differences(a, b, p, mask=None):
    # mean of (a-b) with trimming; unlike medians, mean(d) == mean(a)-mean(b)
    if mask is not None:
        m = mask
        a = a[m]; b = b[m]
    if a.size == 0:
        return 0.0
    d = a - b
    return trim_mean(d, 2*p)

def align_rows(
    img,
    method="median",                 # "median","modus","polynomial","median_diff","matching","facet_tilt","trimmed_mean","trimmed_mean_diff"
    poly_degree=1,                   # for "polynomial"
    trimmed_fraction=0.1,            # for trimmed_* methods (0..0.5)
    matching_flat_weight=0.5,        # 0..1, weight flat areas more in "matching"
    mask=None                        # optional (H,W) boolean; True=use for estimation
):
    """
    Gwyddion-style Align Rows: correct per-row height offsets/slopes (z), not lateral shifts.
    Returns:
        corrected : (H,W) array
        background: (H,W) array of per-row values subtracted (constant or polynomial per row)
        deltas    : (H,) array of row constants (for methods that yield pure offsets)
    """
    I = np.asarray(img, float)
    H, W = I.shape
    x = np.arange(W, dtype=float)

    if mask is not None:
        mask = mask.astype(bool)
        if mask.shape != I.shape:
            raise ValueError("mask must have same shape as image")

    corrected = I.copy()
    background = np.zeros_like(I)
    deltas = np.zeros(H, float)

    # helper to get per-row estimation mask
    def row_mask(r):
        return None if mask is None else mask[r]

    if method.lower() in ("median", "modus", "polynomial", "trimmed_mean"):
        for r in range(H):
            row = I[r]
            mrow = row_mask(r)
            if method == "median":
                c = np.median(row[mrow]) if mrow is not None else np.median(row)
                bg = np.full(W, c)
                deltas[r] = c
            elif method == "modus":
                c = _row_mode_estimate(row, bins=256, mask=mrow)
                bg = np.full(W, c)
                deltas[r] = c
            elif method == "trimmed_mean":
                p = np.clip(trimmed_fraction, 0.0, 0.5)
                c = _trimmed_mean(row, p, mask=mrow)
                bg = np.full(W, c)
                deltas[r] = c
            else:  # polynomial
                pfit = _row_poly_fit(x, row, deg=poly_degree, mask=mrow)
                bg = pfit(x)
            background[r] = bg
            corrected[r] = row - bg

    elif method.lower() in ("median_diff", "trimmed_mean_diff", "matching", "facet_tilt"):
        # iterative row-to-row corrections, preserving large features better (per docs)
        # start with first row unchanged
        corrected[0] = I[0]
        background[0] = 0.0
        deltas[0] = 0.0

        for r in range(1, H):
            row = I[r]
            prev = corrected[r-1]

            mrow = row_mask(r)
            mprev = row_mask(r-1)
            m12 = None
            if mrow is not None and mprev is not None:
                m12 = mrow & mprev

            if method == "median_diff":
                # subtract median of vertical neighbour differences so median(row-prev)==0
                if m12 is not None and np.any(m12):
                    d = row[m12] - prev[m12]
                else:
                    d = row - prev
                delta = np.median(d)
                bg = np.full(W, delta)
                deltas[r] = delta
                background[r] = bg
                corrected[r] = row - delta

            elif method == "trimmed_mean_diff":
                p = np.clip(trimmed_fraction, 0.0, 0.5)
                delta = _trimmed_mean_of_differences(row, prev, p, mask=m12)
                bg = np.full(W, delta)
                deltas[r] = delta
                background[r] = bg
                corrected[r] = row - delta

            elif method == "matching":
                # Weight flat areas more: w = 1 / (1 + |grad|), compute weighted mean diff
                if m12 is not None and np.any(m12):
                    rr, pp = row[m12], prev[m12]
                else:
                    rr, pp = row, prev
                # gradient magnitude along x to downweight sloped regions
                g = np.abs(np.gradient(pp))
                if m12 is not None and np.any(m12):
                    g = g[m12]
                w = 1.0 / (1.0 + g)
                w /= (w.sum() + 1e-12)
                delta = float(np.sum(w * (rr - pp)))
                bg = np.full(W, delta)
                deltas[r] = delta
                background[r] = bg
                corrected[r] = row - delta

            else:  # facet_tilt
                # Approximate: remove per-row slope so prevalent normals are vertical (docs: tilt only)
                # Robust linear fit of row vs x (weighted by inverse gradient to favor flat facets)
                rr = row.copy()
                if mrow is not None:
                    mm = mrow
                else:
                    mm = np.ones_like(rr, dtype=bool)
                g = np.abs(np.gradient(rr))
                w = 1.0 / (1.0 + g)
                w = w * mm
                w[w < (0.1 * np.max(w))] = 0  # ignore steep edges
                if np.count_nonzero(w) < 4:
                    pfit = Polynomial.fit(x[mm], rr[mm], 1).convert()
                else:
                    # weighted least squares for slope/intercept
                    X = np.c_[x, np.ones_like(x)]
                    W = np.diag(w)
                    beta = np.linalg.lstsq(W @ X, W @ rr, rcond=None)[0]  # [a,b]
                    pfit = Polynomial([beta[1], beta[0]])
                bg = pfit(x)          # tilt only; no offset alignment
                background[r] = bg
                corrected[r] = row - bg
                deltas[r] = float(np.mean(bg))  # informational

    else:
        raise ValueError("Unknown method")

    return corrected, background, deltas
