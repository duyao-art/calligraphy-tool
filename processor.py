import os
import cv2
import numpy as np
from PIL import Image


def load_manuscript(path):
    """Load image(s) from file. Handles PDF and common image formats."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        try:
            import fitz  # PyMuPDF — no external binary needed
            doc = fitz.open(path)
            images = []
            for page in doc:
                # Render at 150 dpi — good quality, ~44% less memory than 200 dpi
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append(img)
            doc.close()
            return images
        except ImportError:
            raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")
    else:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")
        return [img]


def smooth_projection(proj, kernel_size):
    """Gaussian-like smoothing via convolution."""
    k = max(3, kernel_size | 1)
    gauss = np.exp(-0.5 * np.linspace(-3, 3, k) ** 2)
    gauss /= gauss.sum()
    return np.convolve(proj, gauss, mode='same')


def content_span(proj, threshold_ratio=0.008):
    """Return (start, end) indices of the content span (first/last ink row/col)."""
    threshold = proj.max() * threshold_ratio
    above = proj > threshold
    if not above.any():
        return 0, len(proj)
    start = int(np.argmax(above))
    end   = int(len(above) - 1 - np.argmax(above[::-1]))
    return start, end


def find_valley_boundaries(proj, min_spacing, prominence_thresh=0.25):
    """
    Detect inter-character gap positions using prominence-based valley detection.

    Strategy:
      1. Find all local minima (with coarse stride to avoid redundancy).
      2. For each minimum, measure prominence = how far it drops relative to
         the nearest peaks on each side — works even when the absolute value
         is high (dense calligraphy).
      3. Greedy NMS: pick valleys from most prominent down, keeping only those
         that are ≥ min_spacing away from already-selected ones.

    Returns sorted list of valley indices into `proj`.
    """
    n = len(proj)
    if n < 2 * min_spacing:
        return []

    # Small smoothing — just enough to kill single-pixel scan noise.
    # Crucially, do NOT use large kernels: inter-character gaps can be
    # only 30–50 px, and a k=40 kernel would blur them away entirely.
    smooth = smooth_projection(proj, kernel_size=7)

    # --- Step 1: collect candidate local minima ---
    # Check every pixel (no stride) to avoid stepping over narrow valleys.
    # Use a ±5 comparison window so that flat-bottomed valleys are collapsed
    # to a single candidate (the leftmost minimum of a flat stretch).
    candidates = []
    cmp_w = 5
    for i in range(min_spacing, n - min_spacing):
        lo, hi = max(0, i - cmp_w), min(n, i + cmp_w + 1)
        # Strict minimum over the left half AND right half separately
        if (smooth[i] < smooth[lo:i].min() and
                smooth[i] < smooth[i + 1:hi].min()):
            candidates.append(i)

    if not candidates:
        return []

    # --- Step 2: compute prominence for each candidate ---
    scored = []
    search_r = min_spacing * 4  # look this far on each side for neighboring peaks
    for idx in candidates:
        lw = max(0,   idx - search_r)
        rw = min(n-1, idx + search_r)
        left_peak  = smooth[lw:idx].max()   if idx > lw   else smooth[idx]
        right_peak = smooth[idx+1:rw].max() if idx+1 < rw else smooth[idx]
        ref = min(left_peak, right_peak)
        prominence = (ref - smooth[idx]) / ref if ref > 0 else 0
        if prominence >= prominence_thresh:
            scored.append((idx, prominence))

    if not scored:
        return []

    # --- Step 3: greedy NMS — select in order of prominence ---
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for idx, prom in scored:
        if all(abs(idx - s) >= min_spacing for s in selected):
            selected.append(idx)

    return sorted(selected)


def snap_boundaries(proj, boundaries, radius=15):
    """
    Refine interior boundary lines to the exact local ink minimum within ±radius px.

    Valley detection places lines near the gap centre, but the true minimum
    (ink-free pixel) may be a few pixels away.  Snapping ensures the line sits
    at the deepest part of the gap so it never cuts through a stroke edge.

    The first and last boundaries (content edges) are left unchanged.
    """
    if len(boundaries) < 3:
        return boundaries
    result = [boundaries[0]]
    for b in boundaries[1:-1]:
        lo = max(0, b - radius)
        hi = min(len(proj), b + radius + 1)
        best = int(lo + np.argmin(proj[lo:hi]))
        result.append(best)
    result.append(boundaries[-1])
    return sorted(set(result))


def detect_grid(img, manual_rows=None, manual_cols=None):
    """
    Detect character grid using projection-profile valley detection.

    Lines are placed at the actual gaps between characters — no uniform
    spacing is forced. Works for both tight and loose calligraphy layouts.

    Returns (row_lines, col_lines) as sorted lists of pixel positions.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    h, w = gray.shape

    # Binarize: characters dark on light → invert so ink = white
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    morph_k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph_k)

    # Full-image projection for content-span detection
    row_proj_full = np.sum(binary, axis=1).astype(float)
    col_proj_full = np.sum(binary, axis=0).astype(float)

    top,  bottom = content_span(row_proj_full)
    left, right  = content_span(col_proj_full)

    content_h = bottom - top
    content_w = right  - left

    if content_h < 10 or content_w < 10:
        r = manual_rows or 10
        c = manual_cols or 5
        return (list(np.linspace(0, h, r + 1, dtype=int)),
                list(np.linspace(0, w, c + 1, dtype=int)))

    # Projection on content region only
    h_proj = np.sum(binary[top:bottom, :], axis=1).astype(float)
    v_proj = np.sum(binary[:, left:right], axis=0).astype(float)

    # Full-image projections for snapping (snap uses absolute coordinates)
    h_proj_full = row_proj_full
    v_proj_full = col_proj_full

    # min_spacing: minimum expected gap between consecutive character boundaries.
    # Tuned so the NMS discards sub-character dips while keeping all real row/col gaps.
    # Assumes no more than 30 rows or 7 columns on a single page.
    min_row_sp = max(100, content_h // 30)
    min_col_sp = max(100, content_w // 7)

    if manual_rows:
        row_lines = list(np.linspace(top, bottom, manual_rows + 1, dtype=int))
    else:
        valleys = find_valley_boundaries(h_proj, min_row_sp)
        row_lines = [top] + [top + v for v in valleys] + [bottom]
        row_lines = sorted(set(row_lines))
        # Snap each interior line to the exact ink minimum within ±15px
        row_lines = snap_boundaries(h_proj_full, row_lines, radius=15)

    if manual_cols:
        col_lines = list(np.linspace(left, right, manual_cols + 1, dtype=int))
    else:
        valleys = find_valley_boundaries(v_proj, min_col_sp)
        col_lines = [left] + [left + v for v in valleys] + [right]
        col_lines = sorted(set(col_lines))
        col_lines = snap_boundaries(v_proj_full, col_lines, radius=15)

    return [int(x) for x in row_lines], [int(x) for x in col_lines]


# ── Drawing helpers ────────────────────────────────────────────────────────────

def dashed_hline(img, y, x1, x2, color, thickness=1, dash=12, gap=6):
    x = x1
    while x < x2:
        cv2.line(img, (x, y), (min(x + dash, x2), y), color, thickness)
        x += dash + gap


def dashed_vline(img, x, y1, y2, color, thickness=1, dash=12, gap=6):
    y = y1
    while y < y2:
        cv2.line(img, (x, y), (x, min(y + dash, y2)), color, thickness)
        y += dash + gap


def dashed_line(img, pt1, pt2, color, thickness=1, dash=10, gap=6):
    x1, y1 = pt1
    x2, y2 = pt2
    length = np.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    pos, draw = 0.0, True
    while pos < length:
        seg = dash if draw else gap
        end = min(pos + seg, length)
        if draw:
            p1 = (int(x1 + pos * dx), int(y1 + pos * dy))
            p2 = (int(x1 + end * dx), int(y1 + end * dy))
            cv2.line(img, p1, p2, color, thickness)
        pos = end
        draw = not draw


def draw_grid_overlay(img, row_lines, col_lines, grid_style='tian'):
    """Render the grid overlay on a copy of img for preview."""
    out = img.copy()
    h, w = out.shape[:2]

    BORDER_COLOR = (30, 30, 200)     # red-ish (BGR)
    GUIDE_COLOR  = (30, 140, 220)    # orange (BGR)

    for y in row_lines:
        dashed_hline(out, y, 0, w, BORDER_COLOR, thickness=2)
    for x in col_lines:
        dashed_vline(out, x, 0, h, BORDER_COLOR, thickness=2)

    if grid_style in ('tian', 'mi', 'jiu'):
        for i in range(len(row_lines) - 1):
            for j in range(len(col_lines) - 1):
                y1, y2 = row_lines[i], row_lines[i + 1]
                x1, x2 = col_lines[j], col_lines[j + 1]

                if grid_style == 'jiu':
                    # 九宫格: 2 horizontal + 2 vertical lines → 3×3 = 9 squares
                    for k in (1, 2):
                        yg = y1 + (y2 - y1) * k // 3
                        xg = x1 + (x2 - x1) * k // 3
                        dashed_hline(out, yg, x1, x2, GUIDE_COLOR, thickness=1, dash=7, gap=5)
                        dashed_vline(out, xg, y1, y2, GUIDE_COLOR, thickness=1, dash=7, gap=5)
                else:
                    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
                    dashed_hline(out, cy, x1, x2, GUIDE_COLOR, thickness=1, dash=7, gap=5)
                    dashed_vline(out, cx, y1, y2, GUIDE_COLOR, thickness=1, dash=7, gap=5)
                    if grid_style == 'mi':
                        dashed_line(out, (x1, y1), (x2, y2), GUIDE_COLOR, thickness=1, dash=7)
                        dashed_line(out, (x2, y1), (x1, y2), GUIDE_COLOR, thickness=1, dash=7)
    return out


# ── Main pipeline ──────────────────────────────────────────────────────────────

def process_manuscript(input_path, output_pdf_path, preview_path, params=None):
    if params is None:
        params = {}
    grid_style   = params.get('grid_style', 'tian')
    manual_rows  = params.get('manual_rows')
    manual_cols  = params.get('manual_cols')

    images = load_manuscript(input_path)

    # Cap maximum image dimension to avoid OOM on cloud free tier (512 MB RAM).
    # 3000 px on the long side is plenty for both detection and PDF quality.
    MAX_DIM = 3000
    capped = []
    for img in images:
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        capped.append(img)
    images = capped

    pages_data = []
    for img in images:
        row_lines, col_lines = detect_grid(img, manual_rows, manual_cols)
        pages_data.append((img, row_lines, col_lines))

    # Preview: first page — downscale to max 1200px for fast browser loading
    first_img, first_rows, first_cols = pages_data[0]
    preview = draw_grid_overlay(first_img, first_rows, first_cols, grid_style)
    ph, pw = preview.shape[:2]
    if max(ph, pw) > 1200:
        ps = 1200 / max(ph, pw)
        preview = cv2.resize(preview, (int(pw * ps), int(ph * ps)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(preview_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 85])

    from pdf_generator import generate_practice_pdf
    generate_practice_pdf(pages_data, output_pdf_path, grid_style)

    return {
        'status': 'success',
        'pages': len(images),
        'rows': len(first_rows) - 1,
        'cols': len(first_cols) - 1,
    }
