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
Pure-Python QR Code generator (no external dependencies).
Supports versions 1-40, all 4 encoding modes (numeric, alphanumeric, byte, kanji),
all 4 EC levels, PNG + SVG + ASCII output.

Chinese characters: use byte mode (default for any non-ASCII text) — UTF-8 encoded.
"""

import struct
import zlib


# ===========================================================================
# Galois Field arithmetic
# ===========================================================================
EXP = [0] * 512
LOG = [0] * 256

def _init_galois():
    x = 1
    for i in range(255):
        EXP[i] = x
        LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    for i in range(255, 512):
        EXP[i] = EXP[i - 255]

_init_galois()


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


def rs_generator_poly(degree):
    poly = [1]
    for i in range(degree):
        new = [0] * (len(poly) + 1)
        for j, coef in enumerate(poly):
            new[j] ^= coef
            new[j + 1] ^= gf_mul(coef, EXP[i])
        poly = new
    return poly


def rs_encode(data, num_ecc):
    gen = rs_generator_poly(num_ecc)
    msg = list(data) + [0] * num_ecc
    for i in range(len(data)):
        coef = msg[i]
        if coef != 0:
            for j in range(len(gen)):
                msg[i + j] ^= gf_mul(gen[j], coef)
    return msg[len(data):]


# ===========================================================================
# Capacity tables for versions 1-40
# Format: version -> [(total_cw, ec_per_block, g1_blocks, g1_data, g2_blocks, g2_data)] x [L,M,Q,H]
# ===========================================================================
CAPACITY = {
    1: [(26, 7, 1, 19, 0, 0), (26, 10, 1, 16, 0, 0), (26, 13, 1, 13, 0, 0), (26, 17, 1, 9, 0, 0)],
    2: [(44, 10, 1, 34, 0, 0), (44, 16, 1, 28, 0, 0), (44, 22, 1, 22, 0, 0), (44, 28, 1, 16, 0, 0)],
    3: [(70, 15, 1, 55, 0, 0), (70, 26, 1, 44, 0, 0), (70, 18, 2, 17, 0, 0), (70, 22, 2, 13, 0, 0)],
    4: [(100, 20, 1, 80, 0, 0), (100, 18, 2, 32, 0, 0), (100, 26, 2, 24, 0, 0), (100, 16, 4, 9, 0, 0)],
    5: [(134, 26, 1, 108, 0, 0), (134, 24, 2, 43, 0, 0), (134, 18, 2, 15, 2, 16), (134, 22, 2, 11, 2, 12)],
    6: [(172, 18, 2, 68, 0, 0), (172, 16, 4, 27, 0, 0), (172, 24, 4, 19, 0, 0), (172, 28, 4, 15, 0, 0)],
    7: [(196, 20, 2, 78, 0, 0), (196, 18, 4, 31, 0, 0), (196, 18, 2, 14, 4, 15), (196, 26, 4, 13, 1, 14)],
    8: [(242, 24, 2, 97, 0, 0), (242, 22, 2, 38, 2, 39), (242, 22, 4, 18, 2, 19), (242, 26, 4, 14, 2, 15)],
    9: [(292, 30, 2, 116, 0, 0), (292, 22, 3, 36, 2, 37), (292, 20, 4, 16, 4, 17), (292, 24, 4, 12, 4, 13)],
    10: [(346, 18, 2, 68, 2, 69), (346, 26, 4, 43, 1, 44), (346, 24, 6, 19, 2, 20), (346, 28, 6, 15, 2, 16)],
    11: [(404, 20, 4, 81, 0, 0), (404, 30, 1, 50, 4, 51), (404, 28, 4, 22, 4, 23), (404, 24, 3, 12, 8, 13)],
    12: [(466, 24, 2, 92, 2, 93), (466, 22, 6, 36, 2, 37), (466, 26, 4, 20, 6, 21), (466, 28, 7, 14, 4, 15)],
    13: [(532, 26, 4, 107, 0, 0), (532, 22, 8, 37, 1, 38), (532, 24, 8, 20, 4, 21), (532, 22, 12, 11, 4, 12)],
    14: [(581, 30, 3, 115, 1, 116), (581, 24, 4, 40, 5, 41), (581, 20, 11, 16, 5, 17), (581, 24, 11, 12, 5, 13)],
    15: [(655, 22, 5, 87, 1, 88), (655, 24, 5, 41, 5, 42), (655, 30, 5, 24, 7, 25), (655, 24, 11, 12, 7, 13)],
    16: [(733, 24, 5, 98, 1, 99), (733, 28, 7, 45, 3, 46), (733, 24, 15, 19, 2, 20), (733, 30, 3, 15, 13, 16)],
    17: [(815, 28, 1, 107, 5, 108), (815, 28, 10, 46, 1, 47), (815, 28, 1, 22, 15, 23), (815, 28, 2, 14, 17, 15)],
    18: [(901, 30, 5, 120, 1, 121), (901, 26, 9, 43, 4, 44), (901, 28, 17, 22, 1, 23), (901, 28, 2, 14, 19, 15)],
    19: [(991, 28, 3, 113, 4, 114), (991, 26, 3, 44, 11, 45), (991, 26, 17, 21, 4, 22), (991, 26, 9, 13, 16, 14)],
    20: [(1085, 28, 3, 107, 5, 108), (1085, 26, 3, 41, 13, 42), (1085, 30, 15, 24, 5, 25), (1085, 28, 15, 15, 10, 16)],
    21: [(1156, 28, 4, 116, 4, 117), (1156, 26, 17, 42, 0, 0), (1156, 28, 17, 22, 6, 23), (1156, 30, 19, 16, 6, 17)],
    22: [(1258, 28, 2, 111, 7, 112), (1258, 28, 17, 46, 0, 0), (1258, 30, 7, 24, 16, 25), (1258, 24, 34, 13, 0, 0)],
    23: [(1364, 30, 4, 121, 5, 122), (1364, 28, 4, 47, 14, 48), (1364, 30, 11, 24, 14, 25), (1364, 30, 16, 15, 14, 16)],
    24: [(1474, 30, 6, 117, 4, 118), (1474, 28, 6, 45, 14, 46), (1474, 30, 11, 24, 16, 25), (1474, 30, 30, 16, 2, 17)],
    25: [(1588, 26, 8, 106, 4, 107), (1588, 28, 8, 47, 13, 48), (1588, 30, 7, 24, 22, 25), (1588, 30, 22, 15, 13, 16)],
    26: [(1706, 28, 10, 114, 2, 115), (1706, 28, 19, 46, 4, 47), (1706, 28, 28, 22, 6, 23), (1706, 30, 33, 16, 4, 17)],
    27: [(1828, 30, 8, 122, 4, 123), (1828, 28, 22, 45, 3, 46), (1828, 30, 8, 23, 26, 24), (1828, 30, 12, 15, 28, 16)],
    28: [(1921, 30, 3, 117, 10, 118), (1921, 28, 3, 45, 23, 46), (1921, 30, 4, 24, 31, 25), (1921, 30, 11, 15, 31, 16)],
    29: [(2051, 30, 7, 116, 7, 117), (2051, 28, 21, 45, 7, 46), (2051, 30, 1, 23, 37, 24), (2051, 30, 19, 15, 26, 16)],
    30: [(2185, 30, 5, 115, 10, 116), (2185, 28, 19, 47, 10, 48), (2185, 30, 15, 24, 25, 25), (2185, 30, 23, 15, 25, 16)],
    31: [(2323, 30, 13, 115, 3, 116), (2323, 28, 2, 46, 29, 47), (2323, 30, 42, 24, 1, 25), (2323, 30, 23, 15, 28, 16)],
    32: [(2465, 30, 17, 115, 0, 0), (2465, 28, 10, 46, 23, 47), (2465, 30, 10, 24, 35, 25), (2465, 30, 19, 15, 35, 16)],
    33: [(2611, 30, 17, 115, 1, 116), (2611, 28, 14, 46, 21, 47), (2611, 30, 29, 24, 19, 25), (2611, 30, 11, 15, 46, 16)],
    34: [(2761, 30, 13, 115, 6, 116), (2761, 28, 14, 46, 23, 47), (2761, 30, 44, 24, 7, 25), (2761, 30, 59, 16, 1, 17)],
    35: [(2876, 30, 12, 121, 7, 122), (2876, 28, 12, 47, 26, 48), (2876, 30, 39, 24, 14, 25), (2876, 30, 22, 15, 41, 16)],
    36: [(3034, 30, 6, 121, 14, 122), (3034, 28, 6, 47, 34, 48), (3034, 30, 46, 24, 10, 25), (3034, 30, 2, 15, 64, 16)],
    37: [(3196, 30, 17, 122, 4, 123), (3196, 28, 29, 46, 14, 47), (3196, 30, 49, 24, 10, 25), (3196, 30, 24, 15, 46, 16)],
    38: [(3362, 30, 4, 122, 18, 123), (3362, 28, 13, 46, 32, 47), (3362, 30, 48, 24, 14, 25), (3362, 30, 42, 15, 32, 16)],
    39: [(3532, 30, 20, 117, 4, 118), (3532, 28, 40, 47, 7, 48), (3532, 30, 43, 24, 22, 25), (3532, 30, 10, 15, 67, 16)],
    40: [(3706, 30, 19, 118, 6, 119), (3706, 28, 18, 47, 31, 48), (3706, 30, 34, 24, 34, 25), (3706, 30, 20, 15, 61, 16)],
}

# Alignment pattern center positions per version
ALIGNMENT = {
    1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
    7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
    11: [6, 30, 54], 12: [6, 32, 58], 13: [6, 34, 62], 14: [6, 26, 46, 66],
    15: [6, 26, 48, 70], 16: [6, 26, 50, 74], 17: [6, 30, 54, 78],
    18: [6, 30, 56, 82], 19: [6, 30, 58, 86], 20: [6, 34, 62, 90],
    21: [6, 28, 50, 72, 94], 22: [6, 26, 50, 74, 98], 23: [6, 30, 54, 78, 102],
    24: [6, 28, 54, 80, 106], 25: [6, 32, 58, 84, 110], 26: [6, 30, 58, 86, 114],
    27: [6, 34, 62, 90, 118], 28: [6, 26, 50, 74, 98, 122],
    29: [6, 30, 54, 78, 102, 126], 30: [6, 26, 52, 78, 104, 130],
    31: [6, 30, 56, 82, 108, 134], 32: [6, 34, 60, 86, 112, 138],
    33: [6, 30, 58, 86, 114, 142], 34: [6, 34, 62, 90, 118, 146],
    35: [6, 30, 54, 78, 102, 126, 150], 36: [6, 24, 50, 76, 102, 128, 154],
    37: [6, 28, 54, 80, 106, 132, 158], 38: [6, 32, 58, 84, 110, 136, 162],
    39: [6, 26, 54, 82, 110, 138, 166], 40: [6, 30, 58, 86, 114, 142, 170],
}

EC_BITS = {0: 1, 1: 0, 2: 3, 3: 2}
EC_LEVEL_MAP = {'L': 0, 'M': 1, 'Q': 2, 'H': 3}
EC_LEVEL_NAMES = ['L', 'M', 'Q', 'H']


# ===========================================================================
# Character count indicator lengths (bits) per mode per version group
# mode: 'num', 'alnum', 'byte', 'kanji'
# ===========================================================================
CC_BITS = {
    'num': [10, 12, 14],
    'alnum': [9, 11, 13],
    'byte': [8, 16, 16],
    'kanji': [8, 10, 12],
}

MODE_INDICATOR = {'num': 0b0001, 'alnum': 0b0010, 'byte': 0b0100, 'kanji': 0b1000}

ALNUM_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


def _version_group(version):
    if version <= 9:
        return 0
    if version <= 26:
        return 1
    return 2


# ===========================================================================
# Data encoders
# ===========================================================================
def _encode_numeric(text):
    bits = []
    for i in range(0, len(text), 3):
        chunk = text[i:i+3]
        val = int(chunk)
        bits_count = {1: 4, 2: 7, 3: 10}[len(chunk)]
        for j in range(bits_count - 1, -1, -1):
            bits.append((val >> j) & 1)
    return bits


def _encode_alnum(text):
    bits = []
    for i in range(0, len(text), 2):
        chunk = text[i:i+2]
        if len(chunk) == 2:
            val = ALNUM_CHARS.index(chunk[0]) * 45 + ALNUM_CHARS.index(chunk[1])
            for j in range(10, -1, -1):
                bits.append((val >> j) & 1)
        else:
            val = ALNUM_CHARS.index(chunk[0])
            for j in range(5, -1, -1):
                bits.append((val >> j) & 1)
    return bits


def _encode_byte(text):
    bits = []
    for b in text.encode('utf-8'):
        for j in range(7, -1, -1):
            bits.append((b >> j) & 1)
    return bits


def _encode_kanji(text):
    """Kanji mode uses Shift-JIS. Works for many CJK chars including most
    traditional Chinese and common Japanese kanji. Simplified Chinese
    characters generally are NOT in Shift-JIS — use byte mode for those."""
    try:
        sjis = text.encode('shift_jis')
    except UnicodeEncodeError:
        raise ValueError("Text contains characters not representable in Shift-JIS")
    if len(sjis) % 2 != 0:
        raise ValueError("Kanji mode requires pairs of Shift-JIS bytes")
    bits = []
    for i in range(0, len(sjis), 2):
        code = (sjis[i] << 8) | sjis[i+1]
        if 0x8140 <= code <= 0x9FFC:
            code -= 0x8140
        elif 0xE040 <= code <= 0xEBBF:
            code -= 0xC140
        else:
            raise ValueError(f"Invalid Shift-JIS code: {code:04X}")
        val = (code >> 8) * 0xC0 + (code & 0xFF)
        for j in range(12, -1, -1):
            bits.append((val >> j) & 1)
    return bits


def _choose_mode(text):
    """Pick the most compact mode for the given text."""
    if all(c.isdigit() for c in text):
        return 'num'
    if all(c in ALNUM_CHARS for c in text):
        return 'alnum'
    # Try Kanji mode if all chars fit Shift-JIS
    try:
        text.encode('shift_jis')
        if all(ord(c) > 0x7F for c in text):
            return 'kanji'
    except UnicodeEncodeError:
        pass
    return 'byte'


def _count_chars(text, mode):
    if mode == 'kanji':
        return len(text)
    return len(text) if mode != 'byte' else len(text.encode('utf-8'))


def _data_bit_length(text, mode):
    """Approximate raw bits needed for data portion (excludes header)."""
    if mode == 'num':
        n = len(text)
        return (n // 3) * 10 + {0: 0, 1: 4, 2: 7}[n % 3]
    if mode == 'alnum':
        n = len(text)
        return (n // 2) * 11 + (6 if n % 2 else 0)
    if mode == 'byte':
        return len(text.encode('utf-8')) * 8
    if mode == 'kanji':
        return len(text) * 13
    return 0


def _encode_data(text, version, level_idx, mode):
    bits = []
    def put(val, n):
        for i in range(n - 1, -1, -1):
            bits.append((val >> i) & 1)

    put(MODE_INDICATOR[mode], 4)
    cc = _count_chars(text, mode)
    put(cc, CC_BITS[mode][_version_group(version)])

    if mode == 'num':
        bits.extend(_encode_numeric(text))
    elif mode == 'alnum':
        bits.extend(_encode_alnum(text))
    elif mode == 'byte':
        bits.extend(_encode_byte(text))
    elif mode == 'kanji':
        bits.extend(_encode_kanji(text))

    _, ec_cw, b1, d1, b2, d2 = CAPACITY[version][level_idx]
    data_cw = b1 * d1 + b2 * d2
    capacity_bits = data_cw * 8

    for _ in range(min(4, capacity_bits - len(bits))):
        bits.append(0)
    while len(bits) % 8 != 0:
        bits.append(0)
    pad_bytes = [0xEC, 0x11]
    i = 0
    while len(bits) < capacity_bits:
        put(pad_bytes[i % 2], 8)
        i += 1
    return bits


def _bits_to_bytes(bits):
    return bytes(int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, len(bits), 8))


def _interleave(data_bytes, version, level_idx):
    _, ec_cw, b1, d1, b2, d2 = CAPACITY[version][level_idx]
    blocks = []
    offset = 0
    for _ in range(b1):
        chunk = data_bytes[offset:offset + d1]
        offset += d1
        blocks.append((list(chunk), rs_encode(chunk, ec_cw)))
    for _ in range(b2):
        chunk = data_bytes[offset:offset + d2]
        offset += d2
        blocks.append((list(chunk), rs_encode(chunk, ec_cw)))

    result = []
    max_data = max(len(b[0]) for b in blocks)
    for i in range(max_data):
        for data, _ in blocks:
            if i < len(data):
                result.append(data[i])
    for i in range(ec_cw):
        for _, ecc in blocks:
            result.append(ecc[i])
    return bytes(result)


# ===========================================================================
# Matrix construction
# ===========================================================================
def _make_matrix(version):
    size = version * 4 + 17
    matrix = [[None] * size for _ in range(size)]
    reserved = [[False] * size for _ in range(size)]

    def place_finder(r, c):
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size:
                    if 0 <= dr <= 6 and 0 <= dc <= 6:
                        val = 1 if (dr in (0, 6) or dc in (0, 6) or (2 <= dr <= 4 and 2 <= dc <= 4)) else 0
                    else:
                        val = 0
                    matrix[rr][cc] = val
                    reserved[rr][cc] = True

    place_finder(0, 0)
    place_finder(0, size - 7)
    place_finder(size - 7, 0)

    for i in range(8, size - 8):
        val = 1 if i % 2 == 0 else 0
        if not reserved[6][i]:
            matrix[6][i] = val
            reserved[6][i] = True
        if not reserved[i][6]:
            matrix[i][6] = val
            reserved[i][6] = True

    pos = ALIGNMENT[version]
    for r in pos:
        for c in pos:
            if (r < 8 and c < 8) or (r < 8 and c > size - 9) or (r > size - 9 and c < 8):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    val = 1 if (max(abs(dr), abs(dc)) != 1) else 0
                    matrix[r + dr][c + dc] = val
                    reserved[r + dr][c + dc] = True

    matrix[size - 8][8] = 1
    reserved[size - 8][8] = True

    for i in range(9):
        reserved[8][i] = True
        reserved[i][8] = True
    for i in range(8):
        reserved[8][size - 1 - i] = True
        reserved[size - 1 - i][8] = True

    if version >= 7:
        for i in range(6):
            for j in range(3):
                reserved[size - 11 + j][i] = True
                reserved[i][size - 11 + j] = True

    return matrix, reserved


def _place_data(matrix, reserved, data_bits):
    size = len(matrix)
    idx = 0
    col = size - 1
    upward = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for row in rows:
            for c in (col, col - 1):
                if not reserved[row][c]:
                    bit = data_bits[idx] if idx < len(data_bits) else 0
                    matrix[row][c] = bit
                    idx += 1
        upward = not upward
        col -= 2


def _apply_mask(matrix, reserved, mask_id):
    size = len(matrix)
    for r in range(size):
        for c in range(size):
            if reserved[r][c]:
                continue
            if mask_id == 0:
                m = (r + c) % 2 == 0
            elif mask_id == 1:
                m = r % 2 == 0
            elif mask_id == 2:
                m = c % 3 == 0
            elif mask_id == 3:
                m = (r + c) % 3 == 0
            elif mask_id == 4:
                m = (r // 2 + c // 3) % 2 == 0
            elif mask_id == 5:
                m = (r * c) % 2 + (r * c) % 3 == 0
            elif mask_id == 6:
                m = ((r * c) % 2 + (r * c) % 3) % 2 == 0
            else:
                m = ((r + c) % 2 + (r * c) % 3) % 2 == 0
            if m:
                matrix[r][c] ^= 1


def _place_format_info(matrix, level_idx, mask_id):
    size = len(matrix)
    fmt = (EC_BITS[level_idx] << 3) | mask_id
    rem = fmt
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    fmt_bits = ((fmt << 10) | rem) ^ 0x5412

    def bit(n):
        return (fmt_bits >> n) & 1

    for i in range(6):
        matrix[8][i] = bit(i)
    matrix[8][7] = bit(6)
    matrix[8][8] = bit(7)
    matrix[7][8] = bit(8)
    for i in range(9, 15):
        matrix[14 - i][8] = bit(i)
    for i in range(8):
        matrix[8][size - 1 - i] = bit(i)
    for i in range(8, 15):
        matrix[size - 15 + i][8] = bit(i)
    matrix[size - 8][8] = 1


def _place_version_info(matrix, version):
    if version < 7:
        return
    size = len(matrix)
    rem = version
    for _ in range(12):
        rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
    bits = (version << 12) | rem
    for i in range(18):
        b = (bits >> i) & 1
        r, c = i // 3, i % 3
        matrix[size - 11 + c][r] = b
        matrix[r][size - 11 + c] = b


def _mask_penalty(matrix):
    size = len(matrix)
    penalty = 0
    for row in matrix:
        run = 1
        for i in range(1, size):
            if row[i] == row[i-1]:
                run += 1
            else:
                if run >= 5:
                    penalty += run - 2
                run = 1
        if run >= 5:
            penalty += run - 2
    for c in range(size):
        run = 1
        for r in range(1, size):
            if matrix[r][c] == matrix[r-1][c]:
                run += 1
            else:
                if run >= 5:
                    penalty += run - 2
                run = 1
        if run >= 5:
            penalty += run - 2
    for r in range(size - 1):
        for c in range(size - 1):
            if matrix[r][c] == matrix[r][c+1] == matrix[r+1][c] == matrix[r+1][c+1]:
                penalty += 3
    p1 = [1,0,1,1,1,0,1,0,0,0,0]
    p2 = [0,0,0,0,1,0,1,1,1,0,1]
    for r in range(size):
        for c in range(size - 10):
            seg = [matrix[r][c+i] for i in range(11)]
            if seg == p1 or seg == p2:
                penalty += 40
    for c in range(size):
        for r in range(size - 10):
            seg = [matrix[r+i][c] for i in range(11)]
            if seg == p1 or seg == p2:
                penalty += 40
    dark = sum(sum(row) for row in matrix)
    pct = dark * 100 // (size * size)
    penalty += abs(pct - 50) // 5 * 10
    return penalty


# ===========================================================================
# Public API
# ===========================================================================
def generate_qr_matrix(text, ec_level='M', mode=None):
    """Generate a QR matrix for the given text.

    mode: None (auto), 'num', 'alnum', 'byte', 'kanji'
    Chinese text uses byte mode automatically (UTF-8).
    """
    level_idx = EC_LEVEL_MAP[ec_level]
    if mode is None:
        mode = _choose_mode(text)

    # Find smallest fitting version
    version = None
    for v in range(1, 41):
        _, ec_cw, b1, d1, b2, d2 = CAPACITY[v][level_idx]
        data_cw = b1 * d1 + b2 * d2
        cc = _count_chars(text, mode)
        header = 4 + CC_BITS[mode][_version_group(v)]
        needed = header + _data_bit_length(text, mode)
        if needed <= data_cw * 8:
            version = v
            break
    if version is None:
        raise ValueError("Text too long for a QR code at this EC level")

    bits = _encode_data(text, version, level_idx, mode)
    data_bytes = _bits_to_bytes(bits)
    final_bytes = _interleave(data_bytes, version, level_idx)
    data_bits = []
    for b in final_bytes:
        for i in range(7, -1, -1):
            data_bits.append((b >> i) & 1)

    # Build the best-masked matrix
    best = None
    best_penalty = None
    for mask_id in range(8):
        matrix, reserved = _make_matrix(version)
        _place_data(matrix, reserved, data_bits)
        _apply_mask(matrix, reserved, mask_id)
        _place_format_info(matrix, level_idx, mask_id)
        _place_version_info(matrix, version)
        p = _mask_penalty(matrix)
        if best_penalty is None or p < best_penalty:
            best_penalty = p
            best = matrix
    return best


# ===========================================================================
# Output writers
# ===========================================================================
def save_png(matrix, filename, scale=8, border=4,
             dark=(0, 0, 0), light=(255, 255, 255)):
    """Save QR matrix as a PNG using only zlib + struct."""
    size = len(matrix)
    img_size = (size + border * 2) * scale

    raw = bytearray()
    for y in range(img_size):
        raw.append(0)  # filter type: none
        my = y // scale - border
        for x in range(img_size):
            mx = x // scale - border
            if 0 <= my < size and 0 <= mx < size and matrix[my][mx]:
                raw.extend(dark)
            else:
                raw.extend(light)

    compressed = zlib.compress(bytes(raw), 9)

    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        c += struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)
        return c

    png = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', img_size, img_size, 8, 2, 0, 0, 0)
    png += chunk(b'IHDR', ihdr)
    png += chunk(b'IDAT', compressed)
    png += chunk(b'IEND', b'')

    with open(filename, 'wb') as f:
        f.write(png)


def save_svg(matrix, filename, scale=8, border=4,
             dark='#000000', light='#ffffff'):
    """Save QR matrix as an SVG file."""
    size = len(matrix)
    full = size + border * 2
    dim = full * scale

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{dim}" height="{dim}" viewBox="0 0 {full} {full}" '
        f'shape-rendering="crispEdges">',
        f'<rect width="{full}" height="{full}" fill="{light}"/>',
        f'<path fill="{dark}" d="',
    ]
    # Merge horizontal runs of dark modules into path segments
    for r in range(size):
        c = 0
        while c < size:
            if matrix[r][c]:
                start = c
                while c < size and matrix[r][c]:
                    c += 1
                parts.append(f'M{start + border},{r + border}h{c - start}v1h-{c - start}z')
            else:
                c += 1
    parts.append('"/>')
    parts.append('</svg>')

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(''.join(parts))


def to_ascii(matrix, border=2):
    """Return the QR code as a string using block characters."""
    size = len(matrix)
    full = size + border * 2
    lines = []
    for y in range(full):
        row = []
        for x in range(full):
            my, mx = y - border, x - border
            dark = 0 <= my < size and 0 <= mx < size and matrix[my][mx]
            row.append('##' if dark else '  ')
        lines.append(''.join(row))
    return '\n'.join(lines)


# ===========================================================================
# Convenience wrapper + CLI
# ===========================================================================
def make_qr(text, filename='qrcode.png', ec_level='M', scale=8,
            border=4, mode=None, fmt=None):
    """Generate a QR code and save it. Format inferred from extension
    unless fmt is given ('png', 'svg', 'ascii')."""
    matrix = generate_qr_matrix(text, ec_level=ec_level, mode=mode)

    if fmt is None:
        lower = filename.lower()
        if lower.endswith('.svg'):
            fmt = 'svg'
        elif lower.endswith('.png'):
            fmt = 'png'
        else:
            fmt = 'png'

    if fmt == 'svg':
        save_svg(matrix, filename, scale=scale, border=border)
    elif fmt == 'ascii':
        text_out = to_ascii(matrix, border=border)
        if filename == '-':
            print(text_out)
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(text_out)
    else:
        save_png(matrix, filename, scale=scale, border=border)

    return matrix


def make_blocky_qr(text, filename, ec_level='M', target_pixels=800):
    """Render the smallest version that fits, scaled up to ~target_pixels."""
    matrix = generate_qr_matrix(text, ec_level=ec_level)
    size = len(matrix) # modules per side
    border = 4
    # Pick scale so the final image ≈ target_pixels
    scale = max(1, target_pixels // (size + border * 2))
    save_png(matrix, filename, scale=scale, border=border)
    return scale


if __name__ == '__main__':
    text = input("Enter text to encode: ").strip()
    if not text:
        print("No text entered.")
    else:
        fn = input("Filename (default: qrcode.png, .svg or .txt for other formats): ").strip() or 'qrcode.png'
        level = (input("Error correction L/M/Q/H (default: M): ").strip() or 'M').upper()
        if level not in ('L', 'M', 'Q', 'H'):
            level = 'M'
        matrix = make_qr(text, fn, ec_level=level)
        print(f"QR code saved as '{fn}'")
        if fn.lower().endswith('.png'):
            pass  # optionally print ASCII preview

    # Usage
    make_blocky_qr(text, "big.png", target_pixels=1000)
    # scale auto-computed, always fills ~1000px regardless of version
