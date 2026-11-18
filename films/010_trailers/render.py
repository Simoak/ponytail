"""ORRIKAN ARCHIVE #010 v2 — "475M": four columns of light to exact scale (1 unit = 100M views).
Usage: render.py still t1 t2 ... | render.py seg i (5 s lossless segment) | render.py check"""
import numpy as np, cv2, sys, os, subprocess, time
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, T = 1080, 1920, 60, 44.0
NF = int(T * FPS)
HERE = os.path.dirname(os.path.abspath(__file__))
FD = '/usr/share/fonts/opentype/inter/'
TAU = 2 * np.pi

# ---------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0): return min(max(x, a), b)
def smooth(x): x = clamp(x); return x * x * x * (x * (x * 6 - 15) + 10)
def bez(x1, y1, x2, y2):
    s = np.linspace(0, 1, 2001)
    bx = 3*(1-s)**2*s*x1 + 3*(1-s)*s*s*x2 + s**3; by = 3*(1-s)**2*s*y1 + 3*(1-s)*s*s*y2 + s**3
    return lambda x: float(np.interp(clamp(x), bx, by))
E_CAM = bez(.65, 0, .35, 1); E_IN = bez(.16, 1, .3, 1); E_OUT = bez(.7, 0, .84, 0)
def seg(t, a, b): return clamp((t - a) / (b - a))

# ---------------------------------------------------------------- data (BRIEF facts table), 1 unit = 100M views
#        name          scope            value  x
COLS = [('GTA VI · T1', 'YouTube', 90.4, -1.95),
        ('GTA VI · T2', 'all platforms', 475.0, -0.65),
        ('DEADPOOL', 'all platforms', 365.0, 0.65),
        ('SPIDER-MAN', 'all platforms', 718.6, 1.95)]
CW = 0.62  # column width

def height(i, t):
    """column height in units at time t (views/100M); the story lives here"""
    if i == 0:  # Trailer 1, YouTube
        v = 90.4 * smooth(seg(t, 5.0, 7.8))
    elif i == 1:  # Trailer 2: full in the hook, rewinds, YouTube part, every platform, payoff burst
        if t < 4.0: v = 475
        elif t < 4.8: v = 475 * (1 - smooth(seg(t, 4.0, 4.8)))
        elif t < 17.0: v = 0
        elif t < 19.6: v = 83.6 * E_IN(seg(t, 17.0, 19.6))  # shape only: the exact YouTube figure is never displayed
        elif t < 24.3: v = 83.6
        elif t < 28.0: v = 83.6 + (330 - 83.6) * smooth(seg(t, 24.3, 28.0))
        elif t < 32.0: v = 330 + 22 * E_IN(seg(t, 28.0, 31.75))
        elif t < 32.8: v = 352 + 123 * E_IN(seg(t, 32.0, 32.8))
        else: v = 475
    elif i == 2: v = 365 * smooth(seg(t, 28.2, 29.6))
    else: v = 718.6 * smooth(seg(t, 36.3, 38.3))
    if i != 1: v *= 1 - smooth(seg(t, 42.5, 43.6))
    return v / 100.0
def growing(i, t): return 0.02 < abs(height(i, t + 1 / FPS) - height(i, t)) * FPS

# ---------------------------------------------------------------- geometry (built once)
rng = np.random.default_rng(10)
def column_pts(x0, hmax):
    """lattice of light on 4 faces + bright vertical edges; y in [0, hmax]"""
    ys = np.arange(0.008, hmax + 0.03, 0.016); u = np.linspace(-CW / 2, CW / 2, 15)
    P, E = [], []
    for side in range(4):
        U, Y = np.meshgrid(u, ys)
        if side == 0: X, Z = x0 + U, np.full_like(U, CW / 2)
        elif side == 1: X, Z = x0 + U, np.full_like(U, -CW / 2)
        elif side == 2: X, Z = np.full_like(U, x0 - CW / 2), U
        else: X, Z = np.full_like(U, x0 + CW / 2), U
        p = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        edge = (np.abs(np.abs(U.ravel()) - CW / 2) < 1e-6)
        P.append(p); E.append(edge)
    P = np.concatenate(P); E = np.concatenate(E)
    P += rng.normal(0, 0.003, P.shape)  # break the perfect lattice
    return P, E
COLP = [column_pts(x, v / 100 + 0.05) for _, _, v, x in COLS]
CAPU = np.stack(np.meshgrid(np.linspace(-CW / 2, CW / 2, 15), np.linspace(-CW / 2, CW / 2, 15)), -1).reshape(-1, 2)
COLPH = [rng.uniform(0, TAU, len(p)) for p, _ in COLP]
# floor: soft dust + base line
FL = np.stack([rng.uniform(-3.8, 3.8, 2600), np.zeros(2600), rng.uniform(-2.5, 3.0, 2600)], 1)
FLB = 0.04 + 0.08 * rng.random(2600)
BASE = np.stack([np.linspace(-2.6, 2.6, 700), np.zeros(700), np.full(700, CW / 2 + 0.05)], 1)
def dashed(y, x0, x1, n=420):
    x = np.linspace(x0, x1, n); keep = (np.arange(n) // 6) % 2 == 0
    return np.stack([x[keep], np.full(keep.sum(), y), np.full(keep.sum(), CW / 2 + 0.06)], 1)
REC1 = dashed(0.904, -2.45, 2.45)   # Trailer 1's YouTube record
REC2 = dashed(3.65, -2.45, 2.45)    # movie-trailer record of the time
# scale: a faint line every 100M behind the chart, revealed as the story climbs
SCALE = [dashed(k, -2.7, 2.7, 300) * np.array([1, 1, 0]) + np.array([0, 0, -0.9]) for k in range(1, 8)]
def scale_on(k, t):
    peak = max(height(i, t) for i in range(4))
    return smooth((peak + 0.6 - k) / 0.5) if k <= 7 else 0
# giant background number in the empty top band (5-25 s): Trailer 1's 90.4M, the number to beat
def text_points(txt, n=6000):
    F = ImageFont.truetype(FD + 'InterDisplay-Bold.otf', 300); w = int(F.getlength(txt)) + 20
    im = Image.new('L', (w, 380), 0); ImageDraw.Draw(im).text((10, 10), txt, font=F, fill=255)
    ys, xs = np.nonzero(np.asarray(im) > 128); r = np.random.default_rng(7)
    idx = r.choice(len(xs), n, replace=len(xs) < n); xs, ys = xs[idx] + r.random(n), ys[idx] + r.random(n)
    sc = 860 / w  # fills the text column width
    return np.stack([540 + (xs - w / 2) * sc, 375 + (ys - (ys.min() + ys.max()) / 2) * sc], 1)
BIGP = text_points('90.4M'); _r = np.random.default_rng(9)
BIG0 = np.stack([_r.uniform(-200, 1280, len(BIGP)), _r.uniform(-300, 1100, len(BIGP))], 1)   # scattered start
BIGUP = BIGP + np.stack([_r.normal(0, 120, len(BIGP)), -_r.uniform(250, 700, len(BIGP))], 1)  # scatter upward
BIGPH = _r.uniform(0, TAU, len(BIGP))
def big_number(t):
    """screen-space particles: gather 4.6-5.8, hold, pulse when T2 falls short (20.2), scatter up 24.3-25.6"""
    if not 4.6 <= t < 25.6: return None
    g, u = smooth(seg(t, 4.6, 5.8)), smooth(seg(t, 24.3, 25.6))
    P = BIG0 * (1 - g) + BIGP * g; P = P * (1 - u) + BIGUP * u
    P = P + np.array([3 * np.sin(TAU * 2 * t / T), 2 * np.cos(TAU * 3 * t / T)])  # micro-drift on the hold
    b = 0.11 * (0.4 + 0.6 * g) * (1 - u) * (1 + 0.2 * np.cos(TAU * 22 * t / T + BIGPH))
    b = b * (1 + 0.5 * np.exp(-((t - 20.35) / 0.18) ** 2))
    return P, b
def line_draw(t, a, b): return smooth(seg(t, a, b)) * (1 - smooth(seg(t, 42.5, 43.3)))

# ---------------------------------------------------------------- camera
P = {  # pos, target, mm, fstop, focus point
 'H':  ((-0.07, 0.34, 13.55), (-0.65, 2.25, 0), 35, 2.8, (-0.65, 2.4, 0)),
 'H2': ((-0.07, 0.30, 12.52), (-0.65, 2.25, 0), 35, 2.8, (-0.65, 2.4, 0)),
 'A':  ((-0.96, 1.18, 6.27), (-1.30, 0.70, 0), 40, 2.8, (-1.3, 0.5, 0)),
 'A2': ((-0.42, 1.30, 6.06), (-1.30, 0.70, 0), 40, 2.8, (-1.3, 0.5, 0)),
 'B':  ((-0.90, 0.92, 6.27), (-1.30, 0.45, 0), 40, 2.8, (-0.65, 0.5, 0)),
 'C':  ((-0.31, 2.64, 12.99), (-0.65, 2.30, 0), 35, 4.0, (-0.3, 2.0, 0)),
 'C2': ((-0.37, 2.72, 12.72), (-0.65, 2.30, 0), 35, 4.0, (0.0, 2.0, 0)),
 'D':  ((0.14, 3.99, 19.01), (0.00, 3.50, 0), 35, 5.6, (0.0, 3.0, 0)),
 'D2': ((1.01, 3.85, 19.09), (0.00, 3.50, 0), 35, 5.6, (0.0, 3.0, 0)),
}
MOVES = [(0.0, 3.6, 'H', 'H2'), (4.3, 7.6, 'H2', 'A'), (12.3, 15.5, 'A', 'A2'), (16.3, 18.8, 'A2', 'B'),
         (24.3, 28.6, 'B', 'C'), (29.0, 31.7, 'C', 'C2'), (33.0, 37.0, 'C2', 'D'), (38.8, 42.2, 'D', 'D2'), (42.5, 44.0, 'D2', 'H')]
def lerp(a, b, k): return np.asarray(a, float) * (1 - k) + np.asarray(b, float) * k
def camera(t):
    cur = MOVES[-1][3]; pose = None
    for a, b, p0, p1 in MOVES:
        if t < a: pose = P[cur]; break
        if t <= b:
            k = E_CAM(seg(t, a, b)); A, B = P[p0], P[p1]
            pose = (lerp(A[0], B[0], k), lerp(A[1], B[1], k), A[2] + (B[2] - A[2]) * k, A[3] + (B[3] - A[3]) * k, lerp(A[4], B[4], k)); break
        cur = p1
    if pose is None: pose = P[cur]
    drift = 0.01 * np.array([np.sin(TAU * 2 * t / T), np.sin(TAU * 3 * t / T + 1), 0])  # micro-drift, loop periodic
    return np.asarray(pose[0], float) + drift, np.asarray(pose[1], float) + drift * 0.5, pose[2], pose[3], np.asarray(pose[4], float)
CY = 0.38 * H  # lens shift: the chart sits above the caption band
def project(X, pos, tgt, mm):
    f = np.asarray(tgt) - pos; f = f / np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    d = X - pos; z = d @ f; fp = H * mm / 36.0
    return W / 2 + fp * (d @ r) / np.maximum(z, 1e-3), CY - fp * (d @ u) / np.maximum(z, 1e-3), z

# ---------------------------------------------------------------- splatting with true depth of field
CTH = (0.5, 1.6, 3.0, 5.0); GAIN = (1.0, 1.1, 1.5, 2.6, 4.0)
def disc(r):
    k = int(np.ceil(r)) + 1; y, x = np.mgrid[-k:k+1, -k:k+1].astype(np.float32)
    d = np.sqrt((x / 0.78) ** 2 + y ** 2); m = np.clip(r + 0.5 - d, 0, 1); return m / m.sum()
K3, K4 = disc(4.0), disc(6.5)
def splat(bufs, x, y, w, coc):
    lvl = np.searchsorted(CTH, coc)
    for L in range(5):
        m = lvl == L
        if not m.any(): continue
        s = 1 if L < 3 else 2; b = bufs[L]; hh, ww = b.shape
        xs, ys, ws = x[m] / s, y[m] / s, w[m] * GAIN[L]
        x0 = np.floor(xs).astype(np.int64); y0 = np.floor(ys).astype(np.int64); fx, fy = xs - x0, ys - y0
        I, Wt = [], []
        for dx, dy, wt in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)), (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
            xi, yi = x0 + dx, y0 + dy; ok = (xi >= 0) & (xi < ww) & (yi >= 0) & (yi < hh)
            I.append(yi[ok] * ww + xi[ok]); Wt.append(ws[ok] * wt[ok])
        b += np.bincount(np.concatenate(I), np.concatenate(Wt), hh * ww).astype(np.float32).reshape(hh, ww)
def new_bufs(): return [np.zeros((H, W), np.float32) for _ in range(3)] + [np.zeros((H // 2, W // 2), np.float32) for _ in range(2)]

def scene_points(t):
    out = []
    for i, (P_, E_) in enumerate(COLP):
        h = height(i, t)
        if h <= 0.002: continue
        y = P_[:, 1]; vis = y <= h
        if not vis.any(): continue
        p, e, ph = P_[vis], E_[vis], COLPH[i][vis]
        b = 0.30 + 0.55 * e                                                   # lattice + bright edges
        b = b * (1 + 0.12 * np.cos(TAU * 11 * t / T + ph))                     # shimmer (loop periodic)
        b += 1.4 * np.exp(-((h - p[:, 1]) / 0.06) ** 2)                        # bright growth front / top
        scan = (t % 1.0) * h; env = np.sin(np.pi * (t % 1.0)) ** 2            # light scan up the column, once per second
        b += 0.6 * env * np.exp(-((p[:, 1] - scan) / 0.05) ** 2)
        out.append((p, b)); out.append((p, b * 0.14, 'soft'))  # soft volume glow: reads as a solid bar of light
        cap = np.stack([COLS[i][3] + CAPU[:, 0], np.full(len(CAPU), h), CAPU[:, 1]], 1)
        out.append((cap, np.full(len(cap), 1.3)))
        refl = p * np.array([1, -1, 1]); out.append((refl, b * 0.10 * np.exp(-p[:, 1] / 0.35), 'soft'))  # floor reflection
    out.append((FL, FLB))
    for k, L in enumerate(SCALE, 1):
        a = scale_on(k, t)
        if a > 0.01: out.append((L, np.full(len(L), 0.16 * a)))
    out.append((BASE, np.full(len(BASE), 0.22)))
    for L, a, b in ((REC1, 12.3, 13.3), (REC2, 29.4, 30.2)):
        k = line_draw(t, a, b)
        if k > 0:
            s = (L[:, 0] - L[0, 0]) / (L[-1, 0] - L[0, 0])
            out.append((L, (s <= k) * (0.75 + 1.5 * np.exp(-((s - k) / 0.03) ** 2) * (k < 1))))
    return out

def accumulate(t, groups, wscale, shake=(0, 0)):
    pos, tgt, mm, fs, fpt = camera(t)
    zf = float(np.dot(fpt - pos, (tgt - pos) / np.linalg.norm(tgt - pos)))
    A = 3.2 * (mm / 50) ** 2 * (2.8 / fs)
    for g in scene_points(t):
        X, b = g[0], g[1]
        x, y, z = project(X, pos, tgt, mm)
        ok = (z > 0.2) & (b > 0.004) & (x > -20) & (x < W + 20) & (y > -20) & (y < H + 20)
        if not ok.any(): continue
        coc = A * np.abs(z[ok] - zf) / z[ok] + (3.5 if len(g) > 2 else 0)
        w = 3.0 * b[ok] * wscale * np.clip((6.0 / z[ok]) ** 1.0, 0.2, 2.0)
        groups.append((x[ok] + shake[0], y[ok] + shake[1], w, coc))

def cam_speed(t):
    pr = np.concatenate([p[::300] for p, _ in COLP])
    a = project(pr, *camera(t)[:2], camera(t)[2]); b = project(pr, *camera(t + 1 / FPS)[:2], camera(t + 1 / FPS)[2])
    ok = (a[2] > 1) & (b[2] > 1) & (a[0] > 0) & (a[0] < W) & (a[1] > 0) & (a[1] < H)
    return float(np.percentile(np.hypot(a[0] - b[0], a[1] - b[1])[ok], 90)) if ok.sum() > 5 else 0

# ---------------------------------------------------------------- post: DOF levels, gentle bloom, tone map
def dim(t): return 1 - 0.4 * smooth(seg(t, 32.3, 33.0)) + 0.4 * smooth(seg(t, 42.4, 43.0)) * (t >= 42.4)
TOP = (1 - 0.72 * np.clip(1 - np.arange(H) / 320, 0, 1) ** 1.5)  # label legibility
BOT = 1 - 0.8 * np.clip((np.arange(H) - 1120) / 140, 0, 1)        # caption band legibility
VIG = (TOP * BOT)[:, None].astype(np.float32)
def post(bufs, t):
    img = cv2.GaussianBlur(bufs[0], (0, 0), 0.6) + cv2.GaussianBlur(bufs[1], (0, 0), 1.1) + cv2.GaussianBlur(bufs[2], (0, 0), 2.2)
    half = cv2.filter2D(bufs[3], -1, K3) + cv2.filter2D(bufs[4], -1, K4)
    img += cv2.resize(half, (W, H), interpolation=cv2.INTER_LINEAR) * 0.25
    q = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bloom = 0.12 * cv2.GaussianBlur(q, (0, 0), 2) + 0.10 * cv2.GaussianBlur(q, (0, 0), 7) + 0.10 * cv2.GaussianBlur(q, (0, 0), 24)
    img += cv2.resize(bloom, (W, H), interpolation=cv2.INTER_LINEAR)
    img = (1 - np.exp(-1.6 * img)) * dim(t)
    return img * VIG

# ---------------------------------------------------------------- text
_fc, _wc = {}, {}
def font(name, size):
    if (name, size) not in _fc: _fc[name, size] = ImageFont.truetype(FD + name, size)
    return _fc[name, size]
def word_img(text, name, size, track):
    key = (text, name, size, track)
    if key in _wc: return _wc[key]
    F = font(name, size); wdt = int(F.getlength(text) + abs(track) * len(text) * size + 8)
    im = Image.new('L', (wdt, int(size * 1.35)), 0); dr = ImageDraw.Draw(im); x = 2.0
    for ch in text: dr.text((x, size * 0.1), ch, font=F, fill=255); x += F.getlength(ch) + track * size
    _wc[key] = (np.asarray(im, np.float32) / 255, x); return _wc[key]
def layout(text, name, size, track, x0, y0, maxw=822, lh=1.12):
    words, x, y, out = text.split(' '), x0, y0, []; sp = font(name, size).getlength(' ')
    for w in words:
        a, adv = word_img(w, name, size, track)
        if x > x0 and x + adv - x0 > maxw: x, y = x0, y + size * lh
        out.append((a, x, y)); x += adv + sp
    return out, y + size * lh
DIM, SEC, WHT = 0.706, 0.541, 1.0
def block(lines, x0, y0, gap=10):
    out, y = [], y0
    for text, name, size, track, col in lines:
        ws, y = layout(text, name, size, track, x0, y); out += [(a, x, yy, col) for a, x, yy in ws]; y += gap
    return out, y
HOOK = [('One video game trailer.', 'InterDisplay-Medium.otf', 56, -0.02, DIM),
        ('475M views in 24 hours.', 'InterDisplay-SemiBold.otf', 72, -0.02, WHT),
        ('· per Rockstar', 'Inter-Medium.otf', 30, 0, SEC)]
CARDS = [(4.0, 'Dec 4, 2023. Trailer 1.', '· Rockstar'),
         (8.0, '90.4M YouTube views in 24 h.', '· Guinness'),
         (12.0, 'Record: most-viewed game reveal on YouTube.', '· Guinness · first 24 h'),
         (16.0, 'May 6, 2025. Trailer 2.', '· Rockstar'),
         (20.0, 'On YouTube alone: short of Trailer 1.', '· Guinness: Trailer 1 still holds it'),
         (24.0, 'Then Rockstar counted every platform.', '· Rockstar, via THR'),
         (28.0, 'Movie-trailer record then: 365M.', '· Guinness, Feb 2024 · Deadpool & Wolverine')]
PAY = [('475M in 24 hours.', 'InterDisplay-SemiBold.otf', 72, -0.02, WHT),
       ('“The biggest video launch', 'InterDisplay-Medium.otf', 56, -0.02, DIM),
       ('of all time.”', 'InterDisplay-Medium.otf', 56, -0.02, DIM),
       ('— Rockstar, via THR, May 2025', 'Inter-Medium.otf', 30, 0, SEC)]
IDEA = [('Mar 2026: Spider-Man took it. 718.6M.', 'InterDisplay-SemiBold.otf', 72, -0.02, WHT),
        ('· per Sony · Brand New Day trailer', 'Inter-Medium.otf', 30, 0, SEC)]
END = [('YOUR TURN —', 'Inter-SemiBold.otf', 52, 0.04, DIM),
       ('Trailer 1 or 2: which hooked you?', 'InterDisplay-SemiBold.otf', 72, -0.02, WHT)]
TX = 88; BAND_Y = 1300

def reveal(items, t, t_in, stagger, t_out=None, out_dur=0.3, full_at_start=False):
    res = []
    for i, (a, x, y, col) in enumerate(items):
        p = 1.0 if full_at_start else E_IN(seg(t, t_in + i * stagger, t_in + i * stagger + 0.45))
        al = p; dy = 18 * (1 - p)
        if t_out is not None: q = E_OUT(seg(t, t_out, t_out + out_dur)); al *= 1 - q; dy -= 8 * q
        if al > 0.003: res.append((a, x, y + dy, col, al))
    return res

def chart_labels(t):
    """value on top, name + scope under the base, attached to the projected column (one text block: the chart)"""
    pos, tgt, mm, _, _ = camera(t); res = []
    for i, (name, scope, val, x0) in enumerate(COLS):
        h = height(i, t)
        if i == 0: on = smooth(seg(t, 4.6, 5.1)) * (1 - smooth(seg(t, 42.5, 42.9))); vtxt = f'{h * 100:.1f}M' if h > 0 else None
        elif i == 1:
            on = 1 - smooth(seg(t, 3.9, 4.2)) + smooth(seg(t, 16.6, 17.1)) * (t < 42.5) + smooth(seg(t, 42.9, 43.4)) * (t >= 42.5)
            if t < 4.0 or t >= 42.9: vtxt, scope = '475M', 'all platforms'
            elif 20.2 <= t < 24.3: vtxt, scope = '< 90.4M', 'YouTube'
            elif 32.6 <= t < 42.5: vtxt, scope = '475M', 'all platforms'
            elif t < 24.3: vtxt, scope = None, 'YouTube'
            else: vtxt, scope = None, 'all platforms'
        elif i == 2: on = smooth(seg(t, 28.0, 28.5)) * (1 - smooth(seg(t, 42.5, 42.9))); vtxt = f'{h * 100:.0f}M' if h > 0 else None
        else: on = smooth(seg(t, 36.1, 36.6)) * (1 - smooth(seg(t, 42.5, 42.9))); vtxt = f'{h * 100:.1f}M' if h > 0 else None
        on = clamp(on)
        if t >= 42.5 and i != 1: vtxt = f'{val:.1f}M' if val % 1 else f'{val:.0f}M'  # rewind: never show invented intermediate values
        if on <= 0.01: continue
        (tx,), (ty,), _ = project(np.array([[x0, h, CW / 2]]), pos, tgt, mm)
        items = []
        a, adv = word_img(name, 'Inter-SemiBold.otf', 28, 0.04)
        items.append((a, tx - adv / 2, ty - (104 if vtxt else 64), DIM))
        if vtxt and h > 0.01:
            a, adv = word_img(vtxt, 'InterDisplay-SemiBold.otf', 44, -0.01); items.append((a, tx - adv / 2, ty - 66, WHT))
        for a, x, y, c in items:
            x = min(max(x, 80), 928 - a.shape[1]); y = max(y, 250)
            res.append((a, x, y, c, on))
    return res

def scale_labels(t):
    pos, tgt, mm, _, _ = camera(t); res = []
    for k in range(1, 8):
        a = scale_on(k, t)
        if a <= 0.01: continue
        (x,), (y,), (z,) = project(np.array([[-2.7, k, -0.9]]), pos, tgt, mm)
        if z <= 0 or not (300 < y < 1080): continue
        img, adv = word_img(f'{k}00M', 'Inter-Medium.otf', 28, 0.02)
        res.append((img, max(x - adv - 16, 80), y - 20, SEC, 0.75 * a))
    return res

def text_items(t):
    out = [(word_img('ARCHIVE #010 · GTA VI · PART 4', 'Inter-Medium.otf', 30, 0.08)[0], TX, 200, SEC, 1.0)]
    hk, _ = block(HOOK, TX, BAND_Y)
    if t < 3.8: out += reveal(hk, t, 0, 0, 3.0, 0.8, full_at_start=True)
    if t >= 42.6: out += reveal(hk, t, 42.6, 0.08)
    for k, (tc, txt, src) in enumerate(CARDS):
        t_end = CARDS[k + 1][0] if k + 1 < len(CARDS) else 31.75
        if tc - 0.1 <= t < t_end:
            its, _ = block([(txt, 'Inter-Medium.otf', 52, -0.01, WHT), (src, 'Inter-Medium.otf', 30, 0, SEC)], TX, 1235)
            n = len(its); st = min(0.07, 0.20 / max(n - 1, 1))
            out += reveal(its, t, tc + 0.1, st, t_end - 0.3 if k + 1 < len(CARDS) else 31.5, 0.25)
    if 32.1 <= t < 36.2:
        its, _ = block(PAY, TX, BAND_Y - 40, gap=14); out += reveal(its, t, 32.1, 0.06, 35.8, 0.3)
    if 36.1 <= t < 39.45:
        its, _ = block(IDEA, TX, BAND_Y); out += reveal(its, t, 36.1, 0.08, 39.15, 0.3)
    if 39.45 <= t < 42.45:
        its, _ = block(END, TX, BAND_Y - 20, gap=12); out += reveal(its, t, 39.45, 0.10, 42.0, 0.4)
    return out + scale_labels(t) + chart_labels(t)

def compose_text(img, t):
    for a, x, y, col, al in text_items(t):
        x, y = int(round(x)), int(round(y)); h, w = a.shape
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0: continue
        aa = a[y0 - y:y1 - y, x0 - x:x1 - x] * al
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - aa) + col * aa
    return img

# ---------------------------------------------------------------- frame
DITHER = [np.random.default_rng(1000 + i).uniform(-0.5, 0.5, (H, W)).astype(np.float32) for i in range(4)]
def render(t):
    t = t % T; groups = []; sh = (0, 0)
    if 32.0 <= t < 32.5:
        k = t - 32.0; amp = 3 * np.exp(-k / 0.10); sh = (amp * np.sin(TAU * 31 * k), amp * np.cos(TAU * 23 * k + 0.7))
    if cam_speed(t) > 2 or 32.0 <= t < 32.6 or 42.5 <= t < 43.9 or 4.0 <= t < 4.8:
        for i in range(16): accumulate(t + (i / 15 - 0.5) / 120.0, groups, 1 / 16, sh)  # 16 samples, 180° shutter
    else:
        accumulate(t, groups, 1.0, sh)
    bn = big_number(t)
    if bn is not None: groups.append((bn[0][:, 0], bn[0][:, 1], 3.0 * bn[1], np.full(len(bn[1]), 3.2)))
    lv = new_bufs(); splat(lv, *[np.concatenate([g[i] for g in groups]) for i in range(4)])
    img = compose_text(post(lv, t), t)
    rgb = np.clip(img * 255 + DITHER[int(round(t * FPS)) % 4], 0, 255).astype(np.uint8)
    return np.repeat(rgb[:, :, None], 3, 2)

def encode_seg(i):
    a, b = i * 300, min((i + 1) * 300, NF); out = os.path.join(HERE, 'seg', f'seg{i:02d}.mkv')
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for f in range(a, b): p.stdin.write(render(f / FPS).tobytes())
    p.stdin.close(); p.wait()

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'still':
        for s in sys.argv[2:]:
            t = float(s); t0 = time.time(); cv2.imwrite(os.path.join(HERE, 'stills', f's_{t:05.2f}.png'), render(t)); print(s, f'{time.time() - t0:.2f}s')
    elif cmd == 'seg': encode_seg(int(sys.argv[2]))
    elif cmd == 'check':  # every frame: column tops/bases inside the frame, above the caption band
        worst = {}
        for f in range(0, NF, 6):
            t = f / FPS; pos, tgt, mm, _, _ = camera(t)
            for i, (_, _, _, x0) in enumerate(COLS):
                h = height(i, t)
                if h <= 0.01: continue
                x, y, _ = project(np.array([[x0 - CW / 2, 0, CW / 2], [x0 + CW / 2, 0, CW / 2], [x0, h, CW / 2]]), pos, tgt, mm)
                bad = []
                if y[2] < 300: bad.append(f'top y{y[2]:.0f}')
                if max(y[0], y[1]) > 1110: bad.append(f'base y{max(y[:2]):.0f}')
                if min(x) < 60 or max(x) > 1020: bad.append(f'x {min(x):.0f}..{max(x):.0f}')
                if bad: worst.setdefault(i, []).append((round(t, 1), bad))
        for i, v in worst.items(): print(COLS[i][0], len(v), v[:3], '...', v[-2:])
