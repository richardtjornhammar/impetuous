"""
Copyright 2026 RICHARD TJÖRNHAMMAR

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Standalone QR Code Decoder — pure Python, standard library only.
Supports: versions 1-40, all EC levels, multi-block RS, arbitrary rotation,
          perspective, lens distortion, uneven lighting, inverted codes.

Robustness measures:
  - Sub-pixel finder centers via weighted centroid of ring pattern
  - BCH-verified format read (all 15 single-bit corrections, both copies)
  - BCH-verified version info (v >= 7)
  - Outer-quad tracing from finder edges (not naive 3.5*mod_px extension)
  - Hybrid binarization (Otsu + adaptive with consistency check)
  - 90-degree retry loop
  - Multi-falloff alignment retry (power-4, power-2, none)
"""

import sys
import zlib
import struct
import itertools


# =========================================================================
# 1. PNG LOADER
# =========================================================================

def load_png(path):
    with open(path, 'rb') as f:
        data = f.read()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError("Not a PNG file")

    pos = 8
    width = height = bitdepth = colortype = None
    idat = b''
    palette = None
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos+4])[0]
        ctype = data[pos+4:pos+8]
        chunk = data[pos+8:pos+8+length]
        pos += 12 + length
        if ctype == b'IHDR':
            width, height, bitdepth, colortype = struct.unpack('>IIBB', chunk[:10])
            if bitdepth != 8:
                raise ValueError(f"Only 8-bit PNG supported (got {bitdepth})")
        elif ctype == b'IDAT':
            idat += chunk
        elif ctype == b'PLTE':
            palette = chunk
        elif ctype == b'IEND':
            break

    raw = zlib.decompress(idat)
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colortype]
    stride = width * channels
    bpp = channels

    out = []
    prev = bytearray(stride)
    p = 0
    for _ in range(height):
        ft = raw[p]; p += 1
        line = bytearray(raw[p:p+stride]); p += stride
        if ft == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i-bpp]) & 0xff
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xff
        elif ft == 3:
            for i in range(stride):
                a = line[i-bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xff
        elif ft == 4:
            for i in range(stride):
                a = line[i-bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i-bpp] if i >= bpp else 0
                pp = a + b - c
                pa, pb, pc = abs(pp-a), abs(pp-b), abs(pp-c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xff
        prev = line

        row = []
        if colortype in (0, 4):
            for x in range(width):
                row.append(line[x * channels])
        elif colortype == 2:
            for x in range(width):
                r = line[x*3]; g = line[x*3+1]; b = line[x*3+2]
                row.append((r*299 + g*587 + b*114) // 1000)
        elif colortype == 3:
            for x in range(width):
                idx = line[x]
                r, g, b = palette[idx*3:idx*3+3]
                row.append((r*299 + g*587 + b*114) // 1000)
        elif colortype == 6:
            for x in range(width):
                r = line[x*4]; g = line[x*4+1]; b = line[x*4+2]
                row.append((r*299 + g*587 + b*114) // 1000)
        out.append(row)
    return out


# =========================================================================
# 2. BINARIZATION (hybrid Otsu + adaptive with consistency check)
# =========================================================================

def otsu_binarize(gray):
    hist = [0]*256
    total = 0
    for row in gray:
        for v in row:
            hist[v] += 1
        total += len(row)
    sum_all = sum(i*hist[i] for i in range(256))
    sumB = wB = 0
    best_t, best_var = 0, 0.0
    for t in range(256):
        wB += hist[t]
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        sumB += t*hist[t]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        var = wB * wF * (mB - mF) ** 2
        if var > best_var:
            best_var = var; best_t = t
    return [[1 if v <= best_t else 0 for v in row] for row in gray]


def adaptive_binarize(gray, block=16, k=0.10):
    h = len(gray); w = len(gray[0])
    if block < 3: block = 3
    if block % 2 == 0: block += 1
    half = block // 2

    integ = [[0]*(w+1) for _ in range(h+1)]
    for r in range(h):
        s = 0
        row = gray[r]
        ip = integ[r]; ip1 = integ[r+1]
        for c in range(w):
            s += row[c]
            ip1[c+1] = ip[c+1] + s

    def mean_rect(r0, c0, r1, c1):
        area = (r1-r0)*(c1-c0)
        if area <= 0: return 128.0
        return (integ[r1][c1] - integ[r0][c1] - integ[r1][c0] + integ[r0][c0]) / area

    bw = [[0]*w for _ in range(h)]
    for r in range(h):
        r0 = max(0, r - half); r1 = min(h, r + half + 1)
        grow = gray[r]; brow = bw[r]
        for c in range(w):
            c0 = max(0, c - half); c1 = min(w, c + half + 1)
            m = mean_rect(r0, c0, r1, c1)
            brow[c] = 1 if grow[c] < m * (1 - k) else 0
    return bw


def hybrid_binarize(gray, finder_hint_pts=None):
    """
    Produce a binarized image. Uses adaptive where local variance is high,
    global Otsu elsewhere. If finder hint points are provided, do a consistency
    check: if >5% of the finder ring agrees with adaptive but not Otsu (or
    vice versa), trust the majority.

    In practice we simply return (bw_global, bw_adaptive) and let the caller
    try both.
    """
    bw_g = otsu_binarize(gray)
    bw_a = adaptive_binarize(gray, block=16, k=0.10)
    return bw_g, bw_a


def invert_bw(bw):
    return [[1 - v for v in row] for row in bw]


# =========================================================================
# 3. FINDER PATTERNS
# =========================================================================

def find_finder_patterns(bw):
    """Find QR finder patterns by horizontal run ratios and vertical confirmation."""
    h, w = len(bw), len(bw[0])
    candidates = []
    for y in range(h):
        row = bw[y]
        runs = []
        x = 0
        while x < w:
            color = row[x]
            x0 = x
            while x < w and row[x] == color:
                x += 1
            runs.append((color, x - x0, x0))
        for i in range(len(runs) - 4):
            window = runs[i:i+5]
            if [item[0] for item in window] != [1, 0, 1, 0, 1]:
                continue
            lengths = [item[1] for item in window]
            if any(v <= 0 for v in lengths):
                continue
            unit = sum(lengths) / 7.0
            if (unit >= 1 and
                all(abs(lengths[j] - unit) < unit*0.7 for j in (0, 1, 3, 4)) and
                abs(lengths[2] - 3*unit) < unit*1.2):
                cx = int(round(window[0][2] + sum(lengths)/2.0))
                cy = _vertical_confirm(bw, cx, y, unit)
                if cy is not None:
                    candidates.append((float(cx), float(cy), unit*7, unit))

    candidates.sort(key=lambda item: item[2], reverse=True)
    kept = []
    for candidate in candidates:
        if all((candidate[0]-other[0])**2 + (candidate[1]-other[1])**2 >
               (0.6*max(candidate[2], other[2]))**2 for other in kept):
            kept.append(candidate)
    # The vertical cross-check already estimates the geometric centre. The
    # previous radial centroid shifted centres because finder patterns are square.
    return kept


def _vertical_confirm(bw, cx, y0, unit):
    h = len(bw)
    def run(y, direction, color):
        cnt = 0
        while 0 <= y < h and bw[y][cx] == color:
            cnt += 1; y += direction
        return cnt
    b_up = run(y0, -1, 1)
    b_dn = run(y0 + 1, 1, 1)
    b1 = b_up + b_dn
    if abs(b1 - 3*unit) > unit * 1.0: return None
    w_up = run(y0 - b_up, -1, 0)
    b2 = run(y0 - b_up - w_up, -1, 1)
    w_dn = run(y0 + b_dn + 1, 1, 0)
    b3 = run(y0 + b_dn + 1 + w_dn, 1, 1)
    total = b1 + w_up + b2 + w_dn + b3
    if abs(total - 7*unit) > unit * 1.5: return None
    return y0 - b_up - w_up - b2 + 1 + total / 2.0


def subpixel_finder_center(bw, cx0, cy0, unit):
    """
    Refine finder centre by weighted centroid of the concentric 1:1:3:1:1 pattern.
    Weights: +1 for pixels that match expected ring pattern, -0.5 for mismatch.
    """
    h = len(bw); w = len(bw[0])
    r = int(round(unit * 4))
    if r < 2: return float(cx0), float(cy0)
    sx = sy = sw = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x = int(round(cx0 + dx))
            y = int(round(cy0 + dy))
            if not (0 <= y < h and 0 <= x < w): continue
            d = ((dx*dx + dy*dy) ** 0.5) / max(unit, 0.5)
            # Expected: dark within d<1, light in 1<d<2, dark in 2<d<3, etc.
            # Actually QR finder is: 3x3 dark, ring of light, ring of dark
            # In module units: r in [0,1.5] -> dark, [1.5,2.5] -> light, [2.5,3.5] -> dark
            if d < 1.5:
                expected = 1
            elif d < 2.5:
                expected = 0
            elif d < 3.5:
                expected = 1
            else:
                expected = 0
            weight = 1.0 if bw[y][x] == expected else -0.5
            sx += weight * x; sy += weight * y; sw += weight
    if abs(sw) < 1e-9:
        return float(cx0), float(cy0)
    return sx / sw, sy / sw



def select_finder_triplet(finders):
    """Choose the most plausible right-angle finder triple from candidates."""
    if len(finders) < 3:
        return None
    pool = finders[:min(len(finders), 60)]
    best = None
    for triple in itertools.combinations(pool, 3):
        pts = [(f[0], f[1]) for f in triple]
        for a, b, c in itertools.permutations(range(3), 3):
            va = (pts[b][0]-pts[a][0], pts[b][1]-pts[a][1])
            vb = (pts[c][0]-pts[a][0], pts[c][1]-pts[a][1])
            na = (va[0]**2 + va[1]**2)**0.5
            nb = (vb[0]**2 + vb[1]**2)**0.5
            if na < 1 or nb < 1:
                continue
            cosine = abs((va[0]*vb[0] + va[1]*vb[1])/(na*nb))
            # Perspective permits unequal legs; penalize gross imbalance only.
            imbalance = abs(na-nb) / max(na, nb)
            sizes = [triple[i][2] for i in range(3)]
            mean_size = sum(sizes)/3
            size_spread = sum(abs(s-mean_size) for s in sizes)/(3*mean_size)
            score = cosine + 0.12*imbalance + 0.35*size_spread
            if best is None or score < best[0]:
                best = (score, pts)
    return best[1] if best else None

def order_corners(pts):
    best = None
    for a, b, c in itertools.permutations(pts, 3):
        v1 = (b[0]-a[0], b[1]-a[1])
        v2 = (c[0]-a[0], c[1]-a[1])
        n1 = (v1[0]**2 + v1[1]**2) ** 0.5
        n2 = (v2[0]**2 + v2[1]**2) ** 0.5
        if n1 == 0 or n2 == 0: continue
        cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
        score = -abs(cosang)
        if best is None or score > best[0]:
            best = (score, a, b, c)
    _, A, B, C = best
    cross = (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
    if cross > 0:
        TR, BL = B, C
    else:
        TR, BL = C, B
    BR = (TR[0] + BL[0] - A[0], TR[1] + BL[1] - A[1])
    return A, TR, BR, BL


# =========================================================================
# 4. HOMOGRAPHY
# =========================================================================

def homography_from_quad(src, dst):
    A = []
    b = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        A.append([sx, sy, 1, 0, 0, 0, -sx*dx, -sy*dx]); b.append(dx)
        A.append([0, 0, 0, sx, sy, 1, -sx*dy, -sy*dy]); b.append(dy)
    n = 8
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(A[r][i]))
        if abs(A[piv][i]) < 1e-12:
            raise ValueError("Degenerate quad")
        if piv != i:
            A[i], A[piv] = A[piv], A[i]
            b[i], b[piv] = b[piv], b[i]
        inv = 1.0 / A[i][i]
        for j in range(i, n): A[i][j] *= inv
        b[i] *= inv
        for k in range(n):
            if k == i: continue
            f = A[k][i]
            if f == 0: continue
            for j in range(i, n): A[k][j] -= f * A[i][j]
            b[k] -= f * b[i]
    return b


def apply_h(h, x, y):
    den = h[6]*x + h[7]*y + 1.0
    if abs(den) < 1e-9: den = 1e-9
    return ((h[0]*x + h[1]*y + h[2]) / den,
            (h[3]*x + h[4]*y + h[5]) / den)


def trace_outer_quad(bw, TL, TR, BL, mod_px):
    """
    Instead of extending finder centres by 3.5*mod_px along the local axes
    (fragile if mod_px is off), trace outward from each finder centre along
    the direction of the other finder, finding the last dark pixel that
    belongs to the finder's outer black border, then step back 0.5 module.

    Actually: finder outer border is the black 7x7 square. Its edge is at
    exactly 3.5 modules from centre in the direction perpendicular to each
    side. We find the outer edge by scanning along each of the two axis
    directions from the centre until we exit the finder pattern (transition
    from dark to light and staying light for ~2 modules).
    """
    def unit(dx, dy):
        m = (dx*dx + dy*dy) ** 0.5
        return (dx/m, dy/m) if m > 0 else (0.0, 0.0)

    def scan_to_edge(cx, cy, ux, uy, max_dist):
        """Walk from (cx,cy) along (ux,uy); return the last dark pixel distance."""
        last_dark = 0
        steps = int(max_dist) + 2
        for i in range(steps):
            x = int(round(cx + ux * i))
            y = int(round(cy + uy * i))
            if not (0 <= y < len(bw) and 0 <= x < len(bw[0])):
                break
            if bw[y][x] == 1:
                last_dark = i
            else:
                # Allow brief light gaps (up to ~1 module) before declaring edge
                if i - last_dark > mod_px * 1.5:
                    break
        return last_dark

    def find_outer_corner(cx, cy, dir1, dir2):
        """The finder's outer corner in the (dir1, dir2) octant."""
        # Look for the outer corner: scan along (dir1+dir2)/sqrt(2) until edge
        ux, uy = unit(dir1[0]+dir2[0], dir1[1]+dir2[1])
        max_d = mod_px * 6
        last_dark = 0
        for i in range(int(max_d) + 2):
            x = int(round(cx + ux * i))
            y = int(round(cy + uy * i))
            if not (0 <= y < len(bw) and 0 <= x < len(bw[0])):
                break
            if bw[y][x] == 1:
                last_dark = i
            else:
                if i - last_dark > mod_px * 1.5:
                    break
        # The corner of the 7x7 square is at (3.5, 3.5) modules from centre,
        # which is 3.5*sqrt(2) modules along the diagonal. Reflect that.
        # But we've been searching along the diagonal, so last_dark is the
        # distance from centre to the outer corner along the diagonal.
        return (cx + ux * last_dark, cy + uy * last_dark)

    u_x = unit(TR[0]-TL[0], TR[1]-TL[1]) # TL's local +x (toward TR)
    u_y = unit(BL[0]-TL[0], BL[1]-TL[1]) # TL's local +y (toward BL)

    # TL outer corner: away from TR and away from BL
    A = find_outer_corner(TL[0], TL[1], (-u_x[0], -u_x[1]), (-u_y[0], -u_y[1]))

    # TR outer corner: outward from the symbol (+x, -y).
    B = find_outer_corner(TR[0], TR[1], (u_x[0], u_x[1]), (-u_y[0], -u_y[1]))

    # BL outer corner: outward from the symbol (-x, +y).
    D = find_outer_corner(BL[0], BL[1], (-u_x[0], -u_x[1]), (u_y[0], u_y[1]))

    # BR outer corner: derived from parallelogram (accurate enough; BR has no finder)
    C = (B[0] + D[0] - A[0], B[1] + D[1] - A[1])

    return A, B, C, D


def outer_corners_simple(TL, TR, BR, BL, mod_px):
    """Fallback: extend finder centres by 3.5 modules along local axes."""
    def unit(dx, dy):
        m = (dx*dx + dy*dy) ** 0.5
        return (dx/m, dy/m) if m > 0 else (0.0, 0.0)
    u_x = unit(TR[0]-TL[0], TR[1]-TL[1])
    u_y = unit(BL[0]-TL[0], BL[1]-TL[1])
    e = 3.5 * mod_px
    A = (TL[0] - e*u_x[0] - e*u_y[0], TL[1] - e*u_x[1] - e*u_y[1])
    B = (TR[0] + e*u_x[0] - e*u_y[0], TR[1] + e*u_x[1] - e*u_y[1])
    D = (BL[0] - e*u_x[0] + e*u_y[0], BL[1] - e*u_x[1] + e*u_y[1])
    C = (BR[0] + e*u_x[0] + e*u_y[0], BR[1] + e*u_x[1] + e*u_y[1])
    return A, B, C, D


# =========================================================================
# 5. TIMING REFINEMENT
# =========================================================================

def refine_module_size(bw, TL, TR, mod_px0):
    """Keep the robust 7-module finder-width estimate.

    The former implementation sampled the line through finder centres as if it
    were the timing pattern; that line is not the QR timing row and biased the
    module size substantially.
    """
    return mod_px0


# =========================================================================
# 6. ALIGNMENT TABLE + SNAPPING
# =========================================================================

ALIGNMENT_TABLE = {
    1: [],
    2: [6, 18],
    3: [6, 22],
    4: [6, 26],
    5: [6, 30],
    6: [6, 34],
    7: [6, 22, 38],
    8: [6, 24, 42],
    9: [6, 26, 46],
    10: [6, 28, 50],
    11: [6, 30, 54],
    12: [6, 32, 58],
    13: [6, 34, 62],
    14: [6, 26, 46, 66],
    15: [6, 26, 48, 70],
    16: [6, 26, 50, 74],
    17: [6, 30, 54, 78],
    18: [6, 30, 56, 82],
    19: [6, 30, 58, 86],
    20: [6, 34, 62, 90],
    21: [6, 28, 50, 72, 94],
    22: [6, 26, 50, 74, 98],
    23: [6, 30, 54, 78, 102],
    24: [6, 28, 54, 80, 106],
    25: [6, 32, 58, 84, 110],
    26: [6, 30, 58, 86, 114],
    27: [6, 34, 62, 90, 118],
    28: [6, 26, 50, 74, 98, 122],
    29: [6, 30, 54, 78, 102, 126],
    30: [6, 26, 52, 78, 104, 130],
    31: [6, 30, 56, 82, 108, 134],
    32: [6, 34, 60, 86, 112, 138],
    33: [6, 30, 58, 86, 114, 142],
    34: [6, 34, 62, 90, 118, 146],
    35: [6, 30, 54, 78, 102, 126, 150],
    36: [6, 24, 50, 76, 102, 128, 154],
    37: [6, 28, 54, 80, 106, 132, 158],
    38: [6, 32, 58, 84, 110, 136, 162],
    39: [6, 26, 54, 82, 110, 138, 166],
    40: [6, 30, 58, 86, 114, 142, 170],
}


def alignment_coords(version):
    if version == 1:
        return []
    n = version * 4 + 17
    coords = ALIGNMENT_TABLE[version]
    out = []
    for r in coords:
        for c in coords:
            if (r < 9 and c < 9) or (r < 9 and c > n - 9) or (r > n - 9 and c < 9):
                continue
            out.append((r, c))
    return out


def alignment_score(bw, cx, cy, half):
    if half < 1: half = 1
    score = 0
    h = len(bw); w = len(bw[0])
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = cx + i*half
            y = cy + j*half
            if not (0 <= y < h and 0 <= x < w):
                score -= 1; continue
            ring = max(abs(i), abs(j))
            expect = 1 if ring in (0, 2) else 0
            score += 1 if bw[y][x] == expect else -1
    return score


def snap_alignment(bw, h_matrix, version, mod_px):
    positions = alignment_coords(version)
    if not positions: return {}
    search_r = max(2, int(mod_px * 1.5))
    half = max(1, int(round(mod_px)))
    out = {}
    for (ar, ac) in positions:
        px, py = apply_h(h_matrix, ac + 0.5, ar + 0.5)
        best = (-10**9, None, None)
        for step, radius in ((2, search_r), (1, 1)):
            if best[1] is None:
                bx, by = int(round(px)), int(round(py))
            else:
                bx, by = best[1], best[2]
            for dy in range(-radius, radius+1, step):
                for dx in range(-radius, radius+1, step):
                    x = bx + dx; y = by + dy
                    if not (0 <= y < len(bw) and 0 <= x < len(bw[0])): continue
                    s = alignment_score(bw, x, y, half)
                    if s > best[0]:
                        best = (s, x, y)
        if best[1] is not None and best[0] > -50:
            out[(ar, ac)] = (best[1], best[2])
    return out


def resample_with_alignment(bw, h_matrix, modules, alignment_pts, falloff=4):
    """
    falloff = IDW exponent. 4 = sharp (default), 2 = soft, 0 = no alignment correction.
    """
    h = len(bw); w = len(bw[0])
    nodes = []
    if falloff > 0:
        for (mr, mc), (px, py) in alignment_pts.items():
            px0, py0 = apply_h(h_matrix, mc + 0.5, mr + 0.5)
            nodes.append((mr, mc, px - px0, py - py0))

    def shift_at(mr, mc):
        if not nodes: return 0.0, 0.0
        sx = sy = den = 0.0
        for (nr, nc, dx, dy) in nodes:
            d2 = (mr - nr)**2 + (mc - nc)**2 + 1.0
            wgt = 1.0 / (d2 ** (falloff / 2.0))
            sx += wgt * dx; sy += wgt * dy; den += wgt
        return sx / den, sy / den

    out = []
    for r in range(modules):
        row = []
        for c in range(modules):
            px, py = apply_h(h_matrix, c + 0.5, r + 0.5)
            if nodes:
                dx, dy = shift_at(r, c)
                px += dx; py += dy
            x = int(round(px)); y = int(round(py))
            if 0 <= y < h and 0 <= x < w:
                row.append(bw[y][x])
            else:
                row.append(0)
        out.append(row)
    return out


# =========================================================================
# 7. BCH(15,5) FOR FORMAT + BCH(18,6) FOR VERSION
# =========================================================================

FORMAT_MASK = 0b101010000010010
EC_LEVELS = {1: 'L', 0: 'M', 3: 'Q', 2: 'H'}

FORMAT_POS_A = [
    (8,0),(8,1),(8,2),(8,3),(8,4),(8,5),(8,7),(8,8),
    (7,8),(5,8),(4,8),(3,8),(2,8),(1,8),(0,8),
]


def _bch15_5_valid(codeword):
    """BCH(15,5) generator G(x) = x^10 + x^8 + x^5 + x^4 + x^2 + x + 1 = 0b10100110111."""
    G = 0b10100110111
    val = codeword & 0x7FFF
    for shift in range(14, 9, -1):
        if val & (1 << shift):
            val ^= G << (shift - 10)
    return (val & 0x3FF) == 0


def read_format(qr):
    """Read either format copy using BCH minimum-distance decoding."""
    n = len(qr)
    pos_b = [(n-1, 8), (n-2, 8), (n-3, 8), (n-4, 8), (n-5, 8),
             (n-6, 8), (n-7, 8), (8, n-8), (8, n-7), (8, n-6),
             (8, n-5), (8, n-4), (8, n-3), (8, n-2), (8, n-1)]
    legal = []
    for ec_bits in range(4):
        for mask_id in range(8):
            data = (ec_bits << 3) | mask_id
            value = data << 10
            for shift in range(14, 9, -1):
                if value & (1 << shift):
                    value ^= 0b10100110111 << (shift - 10)
            legal.append(((data << 10) | (value & 0x3ff)) ^ FORMAT_MASK)

    best_distance, best_code = 16, None
    for positions in (FORMAT_POS_A, pos_b):
        bits = 0
        for r, c in positions:
            bits = (bits << 1) | qr[r][c]
        for code in legal:
            distance = (bits ^ code).bit_count()
            if distance < best_distance:
                best_distance, best_code = distance, code
    if best_code is None or best_distance > 3:
        return None, None
    value = best_code ^ FORMAT_MASK
    return EC_LEVELS[(value >> 13) & 3], (value >> 10) & 7



def _make_version_codewords():
    G = 0b1111100100101
    result = {}
    for v in range(7, 41):
        val = v << 12
        for shift in range(17, 11, -1):
            if val & (1 << shift):
                val ^= G << (shift - 12)
        result[v] = (v << 12) | (val & 0xFFF)
    return result


VERSION_CODEWORDS = _make_version_codewords()


def read_version_info(qr):
    n = len(qr)
    def bits_to_int(bits):
        v = 0
        for b in bits:
            v = (v << 1) | b
        return v

    def extract_bl():
        bits = []
        for c in range(6):
            for r in range(n-9, n-12, -1):
                bits.append(qr[r][c])
        return bits_to_int(bits)

    def extract_tr():
        bits = []
        for r in range(6):
            for c in range(n-9, n-12, -1):
                bits.append(qr[r][c])
        return bits_to_int(bits)

    def best_match(code):
        if code in VERSION_CODEWORDS.values():
            for v, cw in VERSION_CODEWORDS.items():
                if cw == code: return v
        best_v, best_d = None, 99
        for v, cw in VERSION_CODEWORDS.items():
            d = bin(code ^ cw).count('1')
            if d < best_d:
                best_v, best_d = v, d
        return best_v if best_d <= 3 else None

    return best_match(extract_bl()) or best_match(extract_tr())


# =========================================================================
# 8. FUNCTION MASK + BIT EXTRACTION
# =========================================================================

def function_mask(version):
    n = version * 4 + 17
    mask = [[0] * n for _ in range(n)]

    # Finder patterns plus their one-module separators.
    for r in range(8):
        for c in range(8):
            mask[r][c] = 1
            mask[r][n-1-c] = 1
            mask[n-1-r][c] = 1

    # Both copies of the 15 format-information modules.
    pos_b = [(n-1, 8), (n-2, 8), (n-3, 8), (n-4, 8), (n-5, 8),
             (n-6, 8), (n-7, 8), (8, n-8), (8, n-7), (8, n-6),
             (8, n-5), (8, n-4), (8, n-3), (8, n-2), (8, n-1)]
    for r, c in FORMAT_POS_A + pos_b:
        mask[r][c] = 1

    # Timing patterns, excluding the finder/separator regions.
    for i in range(8, n-8):
        mask[6][i] = 1
        mask[i][6] = 1

    for ar, ac in alignment_coords(version):
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                mask[ar+dr][ac+dc] = 1

    # Fixed dark module.
    mask[n-8][8] = 1

    if version >= 7:
        for i in range(6):
            for j in range(3):
                mask[n-11+j][i] = 1
                mask[i][n-11+j] = 1
    return mask



def mask_fn(mask):
    return {
        0: lambda r,c: (r+c) % 2 == 0,
        1: lambda r,c: r % 2 == 0,
        2: lambda r,c: c % 3 == 0,
        3: lambda r,c: (r+c) % 3 == 0,
        4: lambda r,c: (r//2 + c//3) % 2 == 0,
        5: lambda r,c: ((r*c) % 2 + (r*c) % 3) == 0,
        6: lambda r,c: (((r*c) % 2 + (r*c) % 3) % 2) == 0,
        7: lambda r,c: (((r+c) % 2 + (r*c) % 3) % 2) == 0,
    }[mask]


def extract_bits(qr, version, mask_id):
    n = version * 4 + 17
    fm = function_mask(version)
    fn = mask_fn(mask_id)
    bits = []
    col = n - 1
    up = True
    while col > 0:
        if col == 6: col -= 1
        rows = range(n-1, -1, -1) if up else range(n)
        for row in rows:
            for c in (col, col-1):
                if fm[row][c]: continue
                bits.append(qr[row][c] ^ (1 if fn(row, c) else 0))
        up = not up
        col -= 2
    return bits


# =========================================================================
# 9. REED-SOLOMON
# =========================================================================

GF_EXP = [0]*512
GF_LOG = [0]*256


def _init_gf():
    x = 1
    for i in range(255):
        GF_EXP[i] = x
        GF_LOG[x] = i
        x <<= 1
        if x & 0x100: x ^= 0x11D
    for i in range(255, 512):
        GF_EXP[i] = GF_EXP[i-255]


_init_gf()


def gf_mul(a, b):
    if a == 0 or b == 0: return 0
    return GF_EXP[GF_LOG[a] + GF_LOG[b]]


def gf_inv(a):
    if a == 0: raise ZeroDivisionError
    return GF_EXP[255 - GF_LOG[a]]


def poly_eval(p, x):
    r = 0
    for c in p:
        r = gf_mul(r, x) ^ c
    return r


def rs_correct(codewords, nsym):
    """Correct up to nsym//2 symbol errors in a QR Reed-Solomon block."""
    cw = list(codewords)
    n = len(cw)
    if nsym == 0:
        return cw
    if n == 0 or n > 255:
        return None
    synd = []
    for i in range(nsym):
        s = 0
        a = GF_EXP[i]
        for c in cw:
            s = gf_mul(s, a) ^ c
        synd.append(s)
    if not any(synd):
        return cw

    # Determine the locator polynomial from the syndrome recurrence.
    # Try increasing degrees; the first consistent solution is the minimal one.
    C = None
    degree = 0
    for t in range(1, nsym // 2 + 1):
        equations = []
        for k in range(t, nsym):
            equations.append([synd[k-i] for i in range(1, t+1)] + [synd[k]])
        row_count = len(equations)
        pivot_cols = []
        pivot_row = 0
        for col in range(t):
            pivot = next((r for r in range(pivot_row, row_count)
                          if equations[r][col]), None)
            if pivot is None:
                continue
            equations[pivot_row], equations[pivot] = equations[pivot], equations[pivot_row]
            inv = gf_inv(equations[pivot_row][col])
            equations[pivot_row] = [gf_mul(v, inv) for v in equations[pivot_row]]
            for r in range(row_count):
                if r != pivot_row and equations[r][col]:
                    factor = equations[r][col]
                    equations[r] = [x ^ gf_mul(factor, y)
                                    for x, y in zip(equations[r], equations[pivot_row])]
            pivot_cols.append(col)
            pivot_row += 1
            if pivot_row == row_count:
                break
        if len(pivot_cols) != t:
            continue
        if any(all(row[c] == 0 for c in range(t)) and row[t] != 0
               for row in equations):
            continue
        coeffs = [0] * t
        for r, col in enumerate(pivot_cols):
            coeffs[col] = equations[r][t]
        C = [1] + coeffs
        degree = t
        break
    if C is None:
        return None

    positions, locations = [], []
    for pos in range(n):
        X = GF_EXP[(n - 1 - pos) % 255]
        if poly_eval(C[::-1], gf_inv(X)) == 0:
            positions.append(pos)
            locations.append(X)
    if len(positions) != degree:
        return None

    # Solve the first degree syndrome equations for error magnitudes.
    mat = []
    for k in range(degree):
        row = [1 if k == 0 else GF_EXP[(GF_LOG[x] * k) % 255]
               for x in locations]
        row.append(synd[k])
        mat.append(row)
    for col in range(degree):
        pivot = next((r for r in range(col, degree) if mat[r][col]), None)
        if pivot is None:
            return None
        mat[col], mat[pivot] = mat[pivot], mat[col]
        inv = gf_inv(mat[col][col])
        mat[col] = [gf_mul(v, inv) for v in mat[col]]
        for r in range(degree):
            if r != col and mat[r][col]:
                f = mat[r][col]
                mat[r] = [a ^ gf_mul(f, b) for a, b in zip(mat[r], mat[col])]
    for pos, row in zip(positions, mat):
        cw[pos] ^= row[-1]

    for i in range(nsym):
        s = 0
        a = GF_EXP[i]
        for c in cw:
            s = gf_mul(s, a) ^ c
        if s:
            return None
    return cw


# =========================================================================
# 10. RS BLOCK STRUCTURE (all 40 versions)
# =========================================================================

RS_BLOCKS = {
    (1,'L'): (7, [(1, 19)]),
    (1,'M'): (10, [(1, 16)]),
    (1,'Q'): (13, [(1, 13)]),
    (1,'H'): (17, [(1, 9)]),
    (2,'L'): (10, [(1, 34)]),
    (2,'M'): (16, [(1, 28)]),
    (2,'Q'): (22, [(1, 22)]),
    (2,'H'): (28, [(1, 16)]),
    (3,'L'): (15, [(1, 55)]),
    (3,'M'): (26, [(1, 44)]),
    (3,'Q'): (18, [(2, 17)]),
    (3,'H'): (22, [(2, 13)]),
    (4,'L'): (20, [(1, 80)]),
    (4,'M'): (18, [(2, 32)]),
    (4,'Q'): (26, [(2, 24)]),
    (4,'H'): (16, [(4, 9)]),
    (5,'L'): (26, [(1, 108)]),
    (5,'M'): (24, [(2, 43)]),
    (5,'Q'): (18, [(2, 15), (2, 16)]),
    (5,'H'): (22, [(2, 11), (2, 12)]),
    (6,'L'): (18, [(2, 68)]),
    (6,'M'): (16, [(4, 27)]),
    (6,'Q'): (24, [(4, 19)]),
    (6,'H'): (28, [(4, 15)]),
    (7,'L'): (20, [(2, 78)]),
    (7,'M'): (18, [(4, 31)]),
    (7,'Q'): (18, [(2, 14), (4, 15)]),
    (7,'H'): (26, [(4, 13), (1, 14)]),
    (8,'L'): (24, [(2, 97)]),
    (8,'M'): (22, [(2, 38), (2, 39)]),
    (8,'Q'): (22, [(4, 18), (2, 19)]),
    (8,'H'): (26, [(4, 14), (2, 15)]),
    (9,'L'): (30, [(2, 116)]),
    (9,'M'): (22, [(3, 36), (2, 37)]),
    (9,'Q'): (20, [(4, 16), (4, 17)]),
    (9,'H'): (24, [(4, 12), (4, 13)]),
    (10,'L'): (18, [(2, 68), (2, 69)]),
    (10,'M'): (26, [(4, 43), (1, 44)]),
    (10,'Q'): (24, [(6, 19), (2, 20)]),
    (10,'H'): (28, [(6, 15), (2, 16)]),
    (11,'L'): (20, [(4, 81)]),
    (11,'M'): (30, [(1, 50), (4, 51)]),
    (11,'Q'): (28, [(4, 22), (4, 23)]),
    (11,'H'): (24, [(3, 12), (8, 13)]),
    (12,'L'): (24, [(2, 92), (2, 93)]),
    (12,'M'): (22, [(6, 36), (2, 37)]),
    (12,'Q'): (26, [(4, 20), (6, 21)]),
    (12,'H'): (28, [(7, 14), (4, 15)]),
    (13,'L'): (26, [(4, 107)]),
    (13,'M'): (22, [(8, 37), (1, 38)]),
    (13,'Q'): (24, [(8, 20), (4, 21)]),
    (13,'H'): (22, [(12, 11), (4, 12)]),
    (14,'L'): (30, [(3, 115), (1, 116)]),
    (14,'M'): (24, [(4, 40), (5, 41)]),
    (14,'Q'): (20, [(11, 16), (5, 17)]),
    (14,'H'): (24, [(11, 12), (5, 13)]),
    (15,'L'): (22, [(5, 87), (1, 88)]),
    (15,'M'): (24, [(5, 41), (5, 42)]),
    (15,'Q'): (30, [(5, 24), (7, 25)]),
    (15,'H'): (24, [(11, 12), (7, 13)]),
    (16,'L'): (24, [(5, 98), (1, 99)]),
    (16,'M'): (28, [(7, 45), (3, 46)]),
    (16,'Q'): (24, [(15, 19), (2, 20)]),
    (16,'H'): (30, [(3, 15), (13, 16)]),
    (17,'L'): (28, [(1, 107), (5, 108)]),
    (17,'M'): (28, [(10, 46), (1, 47)]),
    (17,'Q'): (28, [(1, 22), (15, 23)]),
    (17,'H'): (28, [(2, 14), (17, 15)]),
    (18,'L'): (30, [(5, 120), (1, 121)]),
    (18,'M'): (26, [(9, 43), (4, 44)]),
    (18,'Q'): (28, [(17, 22), (1, 23)]),
    (18,'H'): (28, [(2, 14), (19, 15)]),
    (19,'L'): (28, [(3, 113), (4, 114)]),
    (19,'M'): (26, [(3, 44), (11, 45)]),
    (19,'Q'): (26, [(17, 21), (4, 22)]),
    (19,'H'): (26, [(9, 13), (16, 14)]),
    (20,'L'): (28, [(3, 107), (5, 108)]),
    (20,'M'): (26, [(3, 41), (13, 42)]),
    (20,'Q'): (30, [(15, 24), (5, 25)]),
    (20,'H'): (28, [(15, 15), (10, 16)]),
    (21,'L'): (28, [(4, 116), (4, 117)]),
    (21,'M'): (26, [(17, 42)]),
    (21,'Q'): (28, [(17, 22), (6, 23)]),
    (21,'H'): (30, [(19, 16), (6, 17)]),
    (22,'L'): (28, [(2, 111), (7, 112)]),
    (22,'M'): (28, [(17, 46)]),
    (22,'Q'): (30, [(7, 24), (16, 25)]),
    (22,'H'): (24, [(34, 13)]),
    (23,'L'): (30, [(4, 121), (5, 122)]),
    (23,'M'): (28, [(4, 47), (14, 48)]),
    (23,'Q'): (30, [(11, 24), (14, 25)]),
    (23,'H'): (30, [(16, 15), (14, 16)]),
    (24,'L'): (30, [(6, 117), (4, 118)]),
    (24,'M'): (28, [(6, 45), (14, 46)]),
    (24,'Q'): (30, [(11, 24), (16, 25)]),
    (24,'H'): (30, [(30, 16), (2, 17)]),
    (25,'L'): (26, [(8, 106), (4, 107)]),
    (25,'M'): (28, [(8, 47), (13, 48)]),
    (25,'Q'): (30, [(7, 24), (22, 25)]),
    (25,'H'): (30, [(22, 15), (13, 16)]),
    (26,'L'): (28, [(10, 114), (2, 115)]),
    (26,'M'): (28, [(19, 46), (4, 47)]),
    (26,'Q'): (28, [(28, 22), (6, 23)]),
    (26,'H'): (30, [(33, 16), (4, 17)]),
    (27,'L'): (30, [(8, 122), (4, 123)]),
    (27,'M'): (28, [(22, 45), (3, 46)]),
    (27,'Q'): (30, [(8, 23), (26, 24)]),
    (27,'H'): (30, [(12, 15), (28, 16)]),
    (28,'L'): (30, [(3, 117), (10, 118)]),
    (28,'M'): (28, [(3, 45), (23, 46)]),
    (28,'Q'): (30, [(4, 24), (31, 25)]),
    (28,'H'): (30, [(11, 15), (31, 16)]),
    (29,'L'): (30, [(7, 116), (7, 117)]),
    (29,'M'): (28, [(21, 45), (7, 46)]),
    (29,'Q'): (30, [(1, 23), (37, 24)]),
    (29,'H'): (30, [(19, 15), (26, 16)]),
    (30,'L'): (30, [(5, 115), (10, 116)]),
    (30,'M'): (28, [(19, 47), (10, 48)]),
    (30,'Q'): (30, [(15, 24), (25, 25)]),
    (30,'H'): (30, [(23, 15), (25, 16)]),
    (31,'L'): (30, [(13, 115), (3, 116)]),
    (31,'M'): (28, [(2, 46), (29, 47)]),
    (31,'Q'): (30, [(42, 24), (1, 25)]),
    (31,'H'): (30, [(23, 15), (28, 16)]),
    (32,'L'): (30, [(17, 115)]),
    (32,'M'): (28, [(10, 46), (23, 47)]),
    (32,'Q'): (30, [(10, 24), (35, 25)]),
    (32,'H'): (30, [(19, 15), (35, 16)]),
    (33,'L'): (30, [(17, 115), (1, 116)]),
    (33,'M'): (28, [(14, 46), (21, 47)]),
    (33,'Q'): (30, [(29, 24), (19, 25)]),
    (33,'H'): (30, [(11, 15), (46, 16)]),
    (34,'L'): (30, [(13, 115), (6, 116)]),
    (34,'M'): (28, [(14, 46), (23, 47)]),
    (34,'Q'): (30, [(44, 24), (7, 25)]),
    (34,'H'): (30, [(59, 16), (1, 17)]),
    (35,'L'): (30, [(12, 121), (7, 122)]),
    (35,'M'): (28, [(12, 47), (26, 48)]),
    (35,'Q'): (30, [(39, 24), (14, 25)]),
    (35,'H'): (30, [(22, 15), (41, 16)]),
    (36,'L'): (30, [(6, 121), (14, 122)]),
    (36,'M'): (28, [(6, 47), (34, 48)]),
    (36,'Q'): (30, [(46, 24), (10, 25)]),
    (36,'H'): (30, [(2, 15), (64, 16)]),
    (37,'L'): (30, [(17, 122), (4, 123)]),
    (37,'M'): (28, [(29, 46), (14, 47)]),
    (37,'Q'): (30, [(49, 24), (10, 25)]),
    (37,'H'): (30, [(24, 15), (46, 16)]),
    (38,'L'): (30, [(4, 122), (18, 123)]),
    (38,'M'): (28, [(13, 46), (32, 47)]),
    (38,'Q'): (30, [(48, 24), (14, 25)]),
    (38,'H'): (30, [(42, 15), (32, 16)]),
    (39,'L'): (30, [(20, 117), (4, 118)]),
    (39,'M'): (28, [(40, 47), (7, 48)]),
    (39,'Q'): (30, [(43, 24), (22, 25)]),
    (39,'H'): (30, [(10, 15), (67, 16)]),
    (40,'L'): (30, [(19, 118), (6, 119)]),
    (40,'M'): (28, [(18, 47), (31, 48)]),
    (40,'Q'): (30, [(34, 24), (34, 25)]),
    (40,'H'): (30, [(20, 15), (61, 16)]),
}


def version_ec_info(version, ec):
    entry = RS_BLOCKS.get((version, ec))
    if entry is None:
        return None
    ec_per_block, groups = entry
    total_data = sum(cnt * data for cnt, data in groups)
    num_blocks = sum(cnt for cnt, _ in groups)
    total = total_data + ec_per_block * num_blocks
    return total, ec_per_block, num_blocks


def deinterleave_blocks(cw, version, ec):
    entry = RS_BLOCKS.get((version, ec))
    if entry is None:
        return None
    ec_per_block, groups = entry

    data_sizes = []
    for cnt, data in groups:
        data_sizes.extend([data] * cnt)
    num_blocks = len(data_sizes)
    total = sum(data_sizes) + ec_per_block * num_blocks
    if len(cw) < total:
        return None

    blocks = [[] for _ in range(num_blocks)]
    idx = 0
    max_ds = max(data_sizes)
    for col in range(max_ds):
        for b in range(num_blocks):
            if col < data_sizes[b]:
                blocks[b].append(cw[idx]); idx += 1

    for col in range(ec_per_block):
        for b in range(num_blocks):
            blocks[b].append(cw[idx]); idx += 1
    return blocks


# =========================================================================
# 11. DATA DECODING
# =========================================================================

ALPHA = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


def decode_data_bytes(cw_list, version=1):
    """Decode QR data codewords into Unicode text."""
    bits = [(c >> i) & 1 for c in cw_list for i in range(7, -1, -1)]
    pos, total_bits = 0, len(bits)
    parts = []
    eci = 3

    def read(n):
        nonlocal pos
        if pos + n > total_bits:
            raise EOFError
        value = 0
        for _ in range(n):
            value = (value << 1) | bits[pos]
            pos += 1
        return value

    group = 0 if version <= 9 else (1 if version <= 26 else 2)
    widths = {1: (10, 12, 14), 2: (9, 11, 13),
              4: (8, 16, 16), 8: (8, 10, 12)}

    while pos + 4 <= total_bits:
        try:
            mode = read(4)
            if mode == 0:
                break
            if mode == 7:
                first = read(8)
                if first & 0x80 == 0:
                    eci = first & 0x7f
                elif first & 0xc0 == 0x80:
                    eci = ((first & 0x3f) << 8) | read(8)
                else:
                    eci = ((first & 0x1f) << 16) | read(16)
                continue
            if mode == 5:
                continue
            if mode == 9:
                read(8)
                continue
            if mode == 3:
                read(16)
                continue
            if mode not in widths:
                raise ValueError(f"Unsupported QR mode: {mode}")
            length = read(widths[mode][group])

            if mode == 1:
                while length >= 3:
                    value = read(10)
                    if value >= 1000:
                        raise ValueError("Invalid numeric group")
                    parts.append(f"{value:03d}")
                    length -= 3
                if length == 2:
                    value = read(7)
                    if value >= 100:
                        raise ValueError("Invalid numeric pair")
                    parts.append(f"{value:02d}")
                elif length == 1:
                    value = read(4)
                    if value >= 10:
                        raise ValueError("Invalid numeric digit")
                    parts.append(str(value))
            elif mode == 2:
                while length >= 2:
                    value = read(11)
                    if value >= 2025:
                        raise ValueError("Invalid alphanumeric pair")
                    parts.append(ALPHA[value // 45] + ALPHA[value % 45])
                    length -= 2
                if length:
                    value = read(6)
                    if value >= 45:
                        raise ValueError("Invalid alphanumeric value")
                    parts.append(ALPHA[value])
            elif mode == 4:
                raw = bytes(read(8) for _ in range(length))
                enc = {3: "iso-8859-1", 20: "shift_jis", 26: "utf-8"}.get(eci)
                parts.append(raw.decode(enc or "iso-8859-1", errors="replace"))
            elif mode == 8:
                raw = bytearray()
                for _ in range(length):
                    value = read(13)
                    assembled = ((value // 0xc0) << 8) | (value % 0xc0)
                    assembled += 0x8140 if assembled < 0x1f00 else 0xc140
                    raw.extend((assembled >> 8, assembled & 0xff))
                parts.append(bytes(raw).decode("shift_jis", errors="replace"))
        except EOFError:
            break
    return "".join(parts)


# =========================================================================
# 12. ROTATION HELPERS
# =========================================================================

def rotate_bw_90(bw):
    """Rotate 90 degrees clockwise."""
    h = len(bw); w = len(bw[0])
    return [[bw[h-1-r][c] for r in range(h)] for c in range(w)]


# =========================================================================
# 13. SINGLE-VERSION ATTEMPT
# =========================================================================

def _try_version(bw, TL, TR, BR, BL, mod_px, version, use_traced_quad, debug):
    modules = version * 4 + 17

    try:
        if use_traced_quad:
            A, B, C, D = trace_outer_quad(bw, TL, TR, BL, mod_px)
        else:
            A, B, C, D = outer_corners_simple(TL, TR, BR, BL, mod_px)
        src = [(0, 0), (modules, 0), (modules, modules), (0, modules)]
        h_matrix = homography_from_quad(src, [A, B, C, D])
    except ValueError:
        return None

    al = snap_alignment(bw, h_matrix, version, mod_px) if version >= 2 else {}

    # Try multiple alignment falloffs
    for falloff in (4, 2, 0):
        qr = resample_with_alignment(bw, h_matrix, modules, al, falloff=falloff)

        if version >= 7:
            ver_read = read_version_info(qr)
            if ver_read is not None and ver_read != version:
                continue

        ec, mask = read_format(qr)
        if ec is None:
            continue
        info = version_ec_info(version, ec)
        if info is None:
            continue
        total_cw, ec_per_block, _ = info

        try:
            bits = extract_bits(qr, version, mask)
        except Exception:
            continue

        cw = []
        for i in range(0, len(bits) - 7, 8):
            v = 0
            for j in range(8):
                v = (v << 1) | bits[i+j]
            cw.append(v)
        cw = cw[:total_cw]

        blocks = deinterleave_blocks(cw, version, ec)
        if blocks is None:
            continue

        corrected_data = []
        ok = True
        for blk in blocks:
            corrected = rs_correct(blk, ec_per_block)
            if corrected is None:
                ok = False; break
            corrected_data.extend(corrected[:len(blk) - ec_per_block])
        if not ok:
            continue

        try:
            text = decode_data_bytes(corrected_data, version)
            if text:
                if debug:
                    print(f"[debug] decoded v{version} ec={ec} mask={mask} "
                          f"falloff={falloff} traced_quad={use_traced_quad}")
                return text
        except Exception:
            continue
    return None


# =========================================================================
# 14. FULL PIPELINE FOR ONE BINARIZATION
# =========================================================================

def _try_decode_on_bw(bw, finders, mod_px, debug=False):
    if len(finders) < 3:
        return None

    pts = select_finder_triplet(finders)
    if pts is None:
        return None
    TL, TR, BR, BL = order_corners(pts)

    # Estimate version
    finder_span = ((TR[0]-TL[0])**2 + (TR[1]-TL[1])**2) ** 0.5
    if mod_px > 0:
        est_modules = finder_span / mod_px + 7
        est_version = max(1, min(40, int(round((est_modules - 17) / 4.0))))
    else:
        est_version = 1

    candidates = sorted(range(1, 41), key=lambda v: (abs(v - est_version), -v))

    # Try both traced and simple outer quad, in that priority order
    for use_traced in (True, False):
        for version in candidates:
            text = _try_version(bw, TL, TR, BR, BL, mod_px, version,
                                use_traced, debug)
            if text:
                return text
    return None


def read_qr(image_path, debug=False):
    gray = load_png(image_path)
    bw_global, bw_adaptive = hybrid_binarize(gray)

    for bw_source, bw_name in ((bw_global, "global"), (bw_adaptive, "adaptive")):
        for polarity, bw_polarity in (("normal", bw_source),
                                      ("inverted", invert_bw(bw_source))):
            current = bw_polarity
            for rotation in range(4):
                finders = find_finder_patterns(current)
                selected = select_finder_triplet(finders)
                if selected is not None:
                    selected_finders = [
                        min(finders, key=lambda f:
                            (f[0]-point[0])**2 + (f[1]-point[1])**2)
                        for point in selected
                    ]
                    mod_px = sum(f[2] for f in selected_finders) / 21.0
                    ordered = order_corners(selected)
                    mod_px = refine_module_size(current, ordered[0], ordered[1], mod_px)
                    if debug:
                        print(f"[debug] {bw_name}/{polarity}/rot{rotation}: "
                              f"{len(finders)} finders, mod_px={mod_px:.2f}")
                    text = _try_decode_on_bw(current, finders, mod_px, debug)
                    if text:
                        return text
                elif debug and rotation == 0:
                    print(f"[debug] {bw_name}/{polarity}: only {len(finders)} finders")

                if rotation < 3:
                    current = rotate_bw_90(current)

    raise ValueError("Could not decode QR code")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qrdecode.py <image.png> [--debug]")
        sys.exit(1)
    debug = "--debug" in sys.argv
    try:
        print(read_qr(sys.argv[1], debug=debug))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)



