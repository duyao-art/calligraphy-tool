import os
import tempfile
import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


def img_to_tempfile(img_bgr):
    """Save OpenCV BGR image to a temp JPEG file; caller must unlink."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    pil.save(tmp.name, 'JPEG', quality=92)
    tmp.close()
    return tmp.name


def mask_outside_grid(img_bgr, row_lines, col_lines):
    """
    Return a copy of img with everything outside the detected grid boundary
    painted white.  This removes stray marks / partial text in the margins
    that fall outside any character cell.
    """
    masked = img_bgr.copy()
    h, w = masked.shape[:2]
    r0, r1 = row_lines[0],  row_lines[-1]
    c0, c1 = col_lines[0],  col_lines[-1]
    white = (255, 255, 255)

    # Top margin
    if r0 > 0:
        masked[:r0, :] = white
    # Bottom margin
    if r1 < h:
        masked[r1:, :] = white
    # Left margin
    if c0 > 0:
        masked[:, :c0] = white
    # Right margin
    if c1 < w:
        masked[:, c1:] = white

    return masked


def generate_practice_pdf(pages_data, output_path, grid_style='tian'):
    """
    Generate a printable practice PDF.

    pages_data : list of (img_bgr, row_lines, col_lines)
    grid_style : 'tian' | 'mi' | 'plain'
    """
    c = canvas.Canvas(output_path, pagesize=A4)
    a4_w, a4_h = A4
    margin = 12 * mm

    for page_idx, (img, row_lines, col_lines) in enumerate(pages_data):
        if page_idx > 0:
            c.showPage()

        img_h, img_w = img.shape[:2]

        # Scale image to fit within margins while preserving aspect ratio
        avail_w = a4_w - 2 * margin
        avail_h = a4_h - 2 * margin
        scale   = min(avail_w / img_w, avail_h / img_h)
        draw_w  = img_w * scale
        draw_h  = img_h * scale
        x_off   = margin + (avail_w - draw_w) / 2
        y_off   = margin + (avail_h - draw_h) / 2

        # Mask content outside the grid before drawing
        img_clean = mask_outside_grid(img, row_lines, col_lines)
        tmp_path = img_to_tempfile(img_clean)
        try:
            c.drawImage(tmp_path, x_off, y_off, width=draw_w, height=draw_h,
                        preserveAspectRatio=True)
        finally:
            os.unlink(tmp_path)

        # Coordinate helper: image pixels → PDF points
        # Image: (0,0) top-left, y↓
        # PDF:   (0,0) bottom-left, y↑
        def to_pdf_x(x_img):
            return x_off + x_img * scale

        def to_pdf_y(y_img):
            return y_off + (img_h - y_img) * scale

        # ── Outer grid boundaries (dashed red) ──────────────────────────────
        c.setStrokeColorRGB(0.85, 0.10, 0.10)
        c.setLineWidth(0.8)
        c.setDash(5, 3)

        for y_img in row_lines:
            y = to_pdf_y(y_img)
            c.line(x_off, y, x_off + draw_w, y)

        for x_img in col_lines:
            x = to_pdf_x(x_img)
            c.line(x, y_off, x, y_off + draw_h)

        # ── Inner cell guides (dashed, lighter) ─────────────────────────────
        if grid_style in ('tian', 'mi', 'jiu'):
            c.setStrokeColorRGB(0.90, 0.50, 0.10)
            c.setLineWidth(0.4)
            c.setDash(3, 4)

            for i in range(len(row_lines) - 1):
                for j in range(len(col_lines) - 1):
                    y1_i, y2_i = row_lines[i], row_lines[i + 1]
                    x1_i, x2_i = col_lines[j], col_lines[j + 1]

                    if grid_style == 'jiu':
                        # 九宫格: divide cell into 3×3 with 2 h-lines + 2 v-lines
                        for k in (1, 2):
                            yg = y1_i + (y2_i - y1_i) * k / 3
                            xg = x1_i + (x2_i - x1_i) * k / 3
                            c.line(to_pdf_x(x1_i), to_pdf_y(yg),
                                   to_pdf_x(x2_i), to_pdf_y(yg))
                            c.line(to_pdf_x(xg), to_pdf_y(y1_i),
                                   to_pdf_x(xg), to_pdf_y(y2_i))
                    else:
                        cy_i = (y1_i + y2_i) / 2
                        cx_i = (x1_i + x2_i) / 2

                        c.line(to_pdf_x(x1_i), to_pdf_y(cy_i),
                               to_pdf_x(x2_i), to_pdf_y(cy_i))
                        c.line(to_pdf_x(cx_i), to_pdf_y(y1_i),
                               to_pdf_x(cx_i), to_pdf_y(y2_i))

                        if grid_style == 'mi':
                            c.line(to_pdf_x(x1_i), to_pdf_y(y1_i),
                                   to_pdf_x(x2_i), to_pdf_y(y2_i))
                            c.line(to_pdf_x(x2_i), to_pdf_y(y1_i),
                                   to_pdf_x(x1_i), to_pdf_y(y2_i))

        c.setDash([])  # reset

    c.save()
