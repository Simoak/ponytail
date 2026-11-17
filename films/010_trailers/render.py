"""ORRIKAN ARCHIVE #010 — "475M". Deterministic renderFrame(t) -> uint8 RGB 1080x1920.
Usage: render.py still t1 t2 ...  |  render.py seg i  (5 s lossless segment)  |  render.py bench"""
import numpy as np, cv2, sys, os, json, subprocess, time
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
    bx = 3*(1-s)**2*s*x1 + 3*(1-s)*s*s*x2 + s**3
    by = 3*(1-s)**2*s*y1 + 3*(1-s)*s*s*y2 + s**3
    return lambda x: float(np.interp(clamp(x), bx, by))
E_CAM = bez(.65, 0, .35, 1); E_IN = bez(.16, 1, .3, 1); E_OUT = bez(.7, 0, .84, 0)
def seg(t, a, b): return clamp((t - a) / (b - a))
def hsh(i):  # deterministic hash -> [0,1)
    i = (np.asarray(i, np.uint64) * np.uint64(2654435761) + np.uint64(97)) & np.uint64(0xFFFFFFFF)
    i ^= i >> np.uint64(15); i = (i * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF); i ^= i >> np.uint64(13)
    return (i & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)

# ---------------------------------------------------------------- world (built once, seeded)
rng = np.random.default_rng(10)
FAN = np.radians(56)
# hall: tiered seat rows (2 points per seat: back + cushion)
seats = []
for r in np.arange(1.25, 3.05, 0.065):
    n = int(2 * FAN * r / 0.030)
    a = np.linspace(-FAN, FAN, n)
    y = 0.36 * (r - 1.25)
    for dy in (0.0, 0.045):
        seats.append(np.stack([r*np.sin(a), np.full(n, y + dy), r*np.cos(a)], 1))
SEAT = np.concatenate(seats); NS = len(SEAT)
# city: building facades beyond the hall
city = []
for gx in np.arange(-9.5, 9.5, 0.42):
    for gz in np.arange(3.5, 11.5, 0.42):
        d = np.hypot(gx, gz); ang = np.arctan2(gx, gz)
        if d < 3.6 or d > 11.2 or abs(ang) > np.radians(60) or rng.random() < 0.22: continue
        h = 0.15 + rng.lognormal(-0.6, 0.55) * (0.6 + 0.12 * d)
        fw = 0.30; nx, ny = 3, max(2, int(h / 0.11))
        xs = gx + (np.arange(nx) / (nx - 1) - 0.5) * fw
        ys = 0.65 + np.arange(ny) * (h / ny)
        X, Y = np.meshgrid(xs, ys)
        keep = rng.random(X.size) < 0.8
        city.append(np.stack([X.ravel()[keep], Y.ravel()[keep], np.full(keep.sum(), gz + fw / 2)], 1))
for gx in np.arange(-9.5, 9.5, 0.84):   # street lamps along a grid
    zz = np.arange(3.6, 11.3, 0.07); city.append(np.stack([np.full(len(zz), gx - 0.21), np.full(len(zz), 0.66), zz], 1))
for gz in np.arange(3.5, 11.5, 0.84):
    xx = np.arange(-9.5, 9.5, 0.07); city.append(np.stack([xx, np.full(len(xx), 0.66), np.full(len(xx), gz - 0.21)], 1))
CITY = np.concatenate(city)
_d, _a = np.hypot(CITY[:, 0], CITY[:, 2]), np.arctan2(CITY[:, 0], CITY[:, 2])
CITY = CITY[(_d > 3.5) & (_d < 11.3) & (np.abs(_a) < np.radians(60))]
_j = np.random.default_rng(77); CITY = CITY + _j.normal(0, [0.045, 0.02, 0.045], CITY.shape)
_dn = 0.5 + 0.5 * np.sin(CITY[:, 0] * 0.9) * np.cos(CITY[:, 2] * 0.7)        # district density variance
CITY = CITY[_j.random(len(CITY)) < 0.35 + 0.65 * _dn]; NC = len(CITY)
CVAR = (0.45 + 1.1 * _j.random(NC) ** 2) * np.exp(-np.maximum(np.hypot(CITY[:, 0], CITY[:, 2]) - 6, 0) / 6)  # brightness variance + horizon falloff
# fill order: distance from screen base, hall = 0..120M, city = 120..475M
dS = np.hypot(SEAT[:, 0], SEAT[:, 2]); oS = np.argsort(dS + 0.002 * hsh(np.arange(NS)))
thrS = np.empty(NS); thrS[oS] = np.linspace(0, 120, NS)
dC = np.hypot(CITY[:, 0], CITY[:, 2]); oC = np.argsort(dC + 0.05 * hsh(np.arange(NC) + 7))
thrC = np.empty(NC); thrC[oC] = np.linspace(120, 475, NC)
def radius_at(v):
    return float(np.interp(v, np.sort(thrS), dS[oS])) if v <= 120 else float(np.interp(v, np.sort(thrC), dC[oC]))
R90, R365 = radius_at(90.4), radius_at(365)
def arc(r, y, n):
    a = np.linspace(-FAN * 1.06, FAN * 1.06, n)
    return np.stack([r*np.sin(a), np.full(n, y), r*np.cos(a)], 1), (a - a[0]) / (a[-1] - a[0])
RING90, RS90 = arc(R90, 0.36 * (R90 - 1.25) + 0.09, 900)
RING365, RS365 = arc(R365, 0.70, 1600)
# screen: vertical 9:16 panel
SX, SY = np.meshgrid(np.linspace(-1, 1, 72), np.linspace(0.9, 4.46, 128))
SCR = np.stack([SX.ravel(), SY.ravel(), np.full(SX.size, -0.05)], 1); NSC = len(SCR)
edge = np.minimum(np.minimum(SX.ravel() + 1, 1 - SX.ravel()), np.minimum(SY.ravel() - 0.9, 4.46 - SY.ravel()))
# hall walls: dotted horizontal rails on the outer arc + side walls
wall = []
for y in (0.2, 0.75, 1.3, 1.85):
    p, _ = arc(3.2, y, 500); wall.append(p)
for s in (-1, 1):
    rr = np.linspace(0.2, 3.2, 160)
    for y in (0.4, 1.0, 1.6, 2.2):
        wall.append(np.stack([s*rr*np.sin(FAN*1.06), np.full(160, y), rr*np.cos(FAN*1.06)], 1))
WALL = np.concatenate(wall)
# dust motes (dynamic, loop periodic)
ND = 1500
D0 = np.stack([rng.uniform(-3.5, 3.5, ND), rng.uniform(0.2, 5.0, ND), rng.uniform(-0.3, 7.5, ND)], 1)
DK = rng.integers(1, 4, (ND, 3)); DPH = rng.uniform(0, TAU, (ND, 3))
TW = rng.choice(NS, 320, replace=False); TWK = rng.integers(8, 15, 320) * 2; TWP = rng.uniform(0, TAU, 320)

# ---------------------------------------------------------------- story functions
def fill(t):
    if t < 9.4: return 0.0
    if t < 11.2: return 90.4 * smooth(seg(t, 9.4, 11.2))
    if t < 16.3: return 90.4
    if t < 17.3: return 90.4 * (1 - smooth(seg(t, 16.3, 17.3)))
    if t < 21.2: return 0.0
    if t < 22.8: return 83.6 * E_IN(seg(t, 21.2, 22.8))   # visual only, figure never displayed
    if t < 24.3: return 83.6
    if t < 28.0: return 83.6 + (330 - 83.6) * smooth(seg(t, 24.3, 28.0))
    if t < 31.75: return 330 + 22 * E_IN(seg(t, 28.0, 31.75))
    if t < 32.0: return 352.0
    if t < 32.8: return 352 + 123 * E_IN(seg(t, 32.0, 32.8))
    if t < 42.5: return 475.0
    return 475 * (1 - smooth(seg(t, 42.5, 43.8)))
def walls(t): return 1 - 0.85 * smooth(seg(t, 24.3, 26.0)) + 0.85 * smooth(seg(t, 42.5, 43.8)) * (t >= 42.5)
def dip(t, a, b, depth=0.85):  # projector restart: down then up
    k = seg(t, a, b); return depth * np.sin(np.pi * k) ** 2 if 0 < k < 1 else 0.0
def screen_on(t): return 1 - dip(t, 3.85, 4.45) - dip(t, 16.0, 17.2, 0.95)
def ring_draw(t, a, b): return smooth(seg(t, a, b)) * (1 - smooth(seg(t, 42.5, 43.3)))
def dim(t): return 1 - 0.5 * smooth(seg(t, 32.3, 33.0)) + 0.5 * smooth(seg(t, 42.4, 43.0)) * (t >= 42.4)

# camera poses: pos, target, mm, fstop, focus point
P = {
 'A':  ((0, 1.4, 13.0), (0, 2.4, 0), 50, 2.8, (0, 2.6, 0)),
 'A2': ((0, 1.3, 11.4), (0, 2.4, 0), 50, 2.8, (0, 2.6, 0)),
 'B':  ((0.3, 1.5, 12.9), (0, 2.4, 0), 60, 2.0, (0, 2.5, 0)),
 'C':  ((0, 10.5, 7.5), (0, 1.7, 1.4), 32, 2.8, (0, 0.5, 1.9)),
 'C2': ((2.4, 10.2, 7.2), (0, 1.95, 1.4), 32, 2.8, (0, 0.5, 1.9)),
 'B2': ((-0.3, 1.3, 10.7), (0, 2.4, 0), 50, 2.8, (0, 2.5, 0)),
 'D':  ((0, 17.0, 17.5), (0, 0.4, 4.2), 24, 5.6, (0, 0.8, 6.5)),
 'F':  ((0, 15.0, 16.5), (0, 0.4, 5.0), 24, 5.6, (0, 0.7, R365)),
 'E':  ((0, 19.0, 20.0), (0, 0.6, 4.2), 24, 5.6, (0, 0.8, 6.0)),
 'E2': ((1.6, 18.2, 19.4), (0, 0.6, 4.2), 24, 5.6, (0, 0.8, 6.0)),
}
MOVES = [(0.0, 3.6, 'A', 'A2'), (4.3, 7.6, 'A2', 'B'), (8.3, 10.6, 'B', 'C'), (12.3, 15.5, 'C', 'C2'),
         (16.3, 18.8, 'C2', 'B2'), (19.0, 21.2, 'B2', 'C'), (24.3, 28.6, 'C', 'D'), (29.0, 31.7, 'D', 'F'),
         (33.0, 37.0, 'F', 'E'), (38.8, 42.2, 'E', 'E2'), (42.5, 44.0, 'E2', 'A')]
def lerp(a, b, k): return tuple(np.asarray(a, float) * (1 - k) + np.asarray(b, float) * k)
def camera(t):
    cur = MOVES[-1][3]; pose = None
    for a, b, p0, p1 in MOVES:
        if t < a: pose = P[cur]; break
        if t <= b:
            k = E_CAM(seg(t, a, b)); A, B = P[p0], P[p1]
            pose = (lerp(A[0], B[0], k), lerp(A[1], B[1], k), A[2] + (B[2] - A[2]) * k,
                    A[3] + (B[3] - A[3]) * k, lerp(A[4], B[4], k)); break
        cur = p1
    if pose is None: pose = P[cur]
    pos = np.array(pose[0], float); tgt = np.array(pose[1], float)
    drift = 0.012 * np.array([np.sin(TAU * 2 * t / T), np.sin(TAU * 3 * t / T + 1), 0])  # micro-drift, periodic
    pos, tgt = pos + drift, tgt + drift * 0.5
    for _ in range(30):  # keep the screen top below the label (y >= 262) on every frame, moves included
        top = project(SCR_TOP, pos, tgt, pose[2])[1].min()
        if top >= 262: break
        tgt = tgt + np.array([0, 0.004 * (262 - top) + 0.002, 0])
    return pos, tgt, pose[2], pose[3], np.array(pose[4], float)
SCR_TOP = np.array([[-1, 4.46, -0.05], [1, 4.46, -0.05]])

CY = 0.40 * H  # lens shift: subject sits above the board band
EXPO = 3.5
def project(X, pos, tgt, mm):
    f = np.asarray(tgt) - pos; f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    d = X - pos; z = d @ f
    fp = H * mm / 36.0
    return W / 2 + fp * (d @ r) / np.maximum(z, 1e-3), CY - fp * (d @ u) / np.maximum(z, 1e-3), z

# ---------------------------------------------------------------- splatting / DOF
CTH = (0.5, 1.6, 3.0, 5.0); GAIN = (1.0, 1.1, 1.5, 2.6, 4.0)
def disc(r):
    k = int(np.ceil(r)) + 1; y, x = np.mgrid[-k:k+1, -k:k+1].astype(np.float32)
    d = np.sqrt((x / 0.78) ** 2 + y ** 2)
    m = np.clip(r + 0.5 - d, 0, 1) * (1 + 0.35 * np.clip(1 - abs(d - r + 0.6), 0, 1))
    return m / m.sum()
K3, K4 = disc(4.0), disc(6.5)
def splat(bufs, x, y, w, coc):
    lvl = np.searchsorted(CTH, coc)
    xs_all, ys_all, ws_all = [], [], []
    for L in range(5):
        m = lvl == L
        if not m.any(): continue
        s = 1 if L < 3 else 2
        b = bufs[L]; hh, ww = b.shape
        xs, ys, ws = x[m] / s, y[m] / s, w[m] * GAIN[L]
        x0 = np.floor(xs).astype(np.int64); y0 = np.floor(ys).astype(np.int64)
        fx, fy = xs - x0, ys - y0
        I, Wt = [], []
        for dx, dy, wt in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)), (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < ww) & (yi >= 0) & (yi < hh)
            I.append(yi[ok] * ww + xi[ok]); Wt.append(ws[ok] * wt[ok])
        b += np.bincount(np.concatenate(I), np.concatenate(Wt), hh * ww).astype(np.float32).reshape(hh, ww)
def new_bufs(): return [np.zeros((H, W), np.float32)] * 0 + [np.zeros((H, W), np.float32) for _ in range(3)] + [np.zeros((H // 2, W // 2), np.float32) for _ in range(2)]

def scene_points(t):
    v = fill(t); son = screen_on(t); wl = walls(t)
    # seats
    lit = np.clip((v - thrS) / 1.5 + 0.5, 0, 1) if v > 0 else np.zeros(NS)
    front = np.exp(-((v - thrS) / 4.0) ** 2) * (v > 0.5) * (v < 474)
    base = 0.05 + 0.10 * hsh(np.arange(NS) + 3) + 0.10 * np.exp(-dS / 1.6)  # screen spill
    tw = np.zeros(NS); tw[TW] = 0.9 * np.maximum(0, np.cos(TAU * TWK * t / T + TWP)) ** 30
    wave = np.exp(-((dS - 1.25 - 2.0 * ((t / 1.0) % 1)) / 0.12) ** 2) * (clamp(1 - seg(t, 3.0, 3.8)) + clamp(seg(t, 42.8, 43.6)))  # seat light wave each beat-pair in the hook
    bS = base * (0.6 + 0.4 * son) + 2.0 * lit + 1.2 * front + tw + 0.5 * wave
    # city
    litc = np.clip((v - thrC) / 6.0 + 0.5, 0, 1) if v > 120 else np.zeros(NC)
    frc = np.exp(-((v - thrC) / 10.0) ** 2) * (v > 120) * (v < 474)
    breath = 1 + 0.06 * np.cos(TAU * 22 * t / T + dC)  # cosine beat envelope (seam rule)
    sweep = 1 + 0.8 * np.exp(-((dC - 3.5 - 8.5 * ((t - 36.0) / 1.5 % 1)) / 0.8) ** 2) * clamp(seg(t, 36.0, 36.6) - seg(t, 41.8, 42.4))  # city light sweeps every 1.5 s
    bC = CVAR * ((0.02 + 0.04 * hsh(np.arange(NC) + 11)) + 2.8 * litc * breath * sweep + 1.4 * frc)
    # screen: abstract light, never footage
    yy = (SCR[:, 1] - 0.9) / 3.56
    pat = 0.55 + 0.25 * np.sin(TAU * (3 * yy - 8 * t / T)) + 0.2 * np.sin(TAU * (SCR[:, 0] * 0.7 + 5 * t / T))
    flick = 0.08 * hsh(np.floor(t * 24) * 7919 + np.arange(NSC) // 288) * son
    idle = 0.42 + 0.06 * np.cos(TAU * 22 * t / T)
    bSc = idle + son * (0.75 * pat * (0.6 + 0.4 * yy) + flick) + 0.35 * (edge < 0.03)
    # rings (stroke reveal with bright head)
    out = []
    for R, S, a, b, pulse in ((RING90, RS90, 10.4, 11.4, 12.3), (RING365, RS365, 28.2, 29.4, 99)):
        k = ring_draw(t, a, b)
        dash = (np.arange(len(R)) % 2 == 0)
        br = (S <= k) * (0.55 + 1.4 * np.exp(-((S - k) / 0.03) ** 2) * (k < 1)) * dash
        br *= 1 - 0.2 * np.sin(np.pi * clamp((t - pulse) / 0.4)) ** 2
        out.append((R, br * (0.6 if R is RING90 and t > 17 else 1.0)))
    # dust
    Dp = D0 + 0.12 * np.sin(TAU * DK * t / T + DPH)
    beam = np.exp(-(Dp[:, 0] / 1.3) ** 2) * (Dp[:, 2] > 0)
    bD = (0.05 + 0.35 * beam * (0.3 + son)) * 0.5
    return [(SEAT, bS, 1.0), (CITY, bC, 1.0), (WALL, np.full(len(WALL), 0.28 * wl), 1.0),
            (out[0][0], out[0][1], 1.2), (out[1][0], out[1][1], 3.0), (Dp, bD, 1.0)]

TEX = np.zeros((640, 360), np.float32)
yy_, xx_ = np.mgrid[0:640, 0:360].astype(np.float32)
TEX += 0.06 + 0.62 * np.clip(yy_ / 400, 0, 1) ** 2.2                           # dusk sky: dark top, bright horizon
TEX += 0.9 * np.exp(-(((xx_ - 180) ** 2 + (yy_ - 330) ** 2) / 2 / 46 ** 2)) * (yy_ < 400)  # sun
TEX[400:] = 0.10 + 0.35 * np.exp(-((xx_[400:] - 180) / 30) ** 2) * (0.5 + 0.5 * np.sin(yy_[400:] * 0.9))  # sea + sun streak
TEX[396:401] = 0.85                                                             # sea line
pr = np.random.default_rng(3)
for px, ph_ in ((60, 300), (300, 260), (330, 330)):                              # palm silhouettes
    cv2.line(TEX, (px, 640), (px + 14, 640 - ph_), 0.02, 7)
    for a in np.linspace(-2.6, -0.5, 6):
        cv2.ellipse(TEX, (px + 14, 640 - ph_), (55, 16), np.degrees(a), 0, 180, 0.02, -1)
TEX = cv2.GaussianBlur(TEX, (0, 0), 1.2)
SCRL = [None]
def scr_layer(t, pos, tgt, mm, A, zf, wscale, shake):
    son = screen_on(t); idle = 0.30 + 0.04 * np.cos(TAU * 22 * t / T)
    trailer2 = t >= 16.5
    tex = TEX[:, ::-1] if trailer2 else TEX
    shift = int(12 * np.sin(TAU * 4 * t / T))
    tex = np.roll(tex, shift, 1) * (0.92 + 0.08 * np.cos(TAU * 88 * t / T))
    lvl = son * (tex * 0.8)
    corners = np.array([[-1, 4.46, -0.05], [1, 4.46, -0.05], [1, 0.9, -0.05], [-1, 0.9, -0.05]])
    x, y, z = project(corners, pos, tgt, mm)
    if (z < 0.2).any(): return
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [360, 0], [360, 640], [0, 640]]), np.float32(np.stack([x + shake[0], y + shake[1]], 1)))
    lay = cv2.warpPerspective(lvl.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR)
    border = cv2.warpPerspective(np.pad(np.zeros((636, 356), np.float32), 2, constant_values=1.0), M, (W, H))
    lay += 0.6 * border
    coc = A * abs(z.mean() - zf) / z.mean()
    if coc > 0.6: lay = cv2.GaussianBlur(lay, (0, 0), coc * 0.5)
    SCRL[0] += lay * wscale * 1.3

def accumulate(t, bufs, wscale, shake=(0, 0)):
    pos, tgt, mm, fs, fpt = camera(t)
    zf = float(np.dot(fpt - pos, (tgt - pos) / np.linalg.norm(tgt - pos)))
    A = 3.2 * (mm / 50) ** 2 * (2.8 / fs)
    for X, b, g in scene_points(t):
        x, y, z = project(X, pos, tgt, mm)
        ok = (z > 0.15) & (b > 0.004) & (x > -20) & (x < W + 20) & (y > -20) & (y < H + 20)
        if not ok.any(): continue
        x, y, z, b = x[ok] + shake[0], y[ok] + shake[1], z[ok], b[ok]
        coc = A * np.abs(z - zf) / z
        w = EXPO * b * g * wscale * np.clip((6.0 / z) ** 1.0, 0.15, 2.2) * (mm / 50) ** 0.5
        bufs.append((x, y, w, coc))
    scr_layer(t, pos, tgt, mm, A, zf, wscale, shake)
    return pos, tgt, mm

def cam_speed(t):
    pr = np.concatenate([SEAT[::200], CITY[::400], SCR[::300]])
    a = project(pr, *camera(t)[:2], camera(t)[2]); b = project(pr, *camera(t + 1 / FPS)[:2], camera(t + 1 / FPS)[2])
    ok = (a[2] > 1.0) & (b[2] > 1.0) & (a[0] > 0) & (a[0] < W) & (a[1] > 0) & (a[1] < H)
    return float(np.percentile(np.hypot(a[0] - b[0], a[1] - b[1])[ok], 90)) if ok.sum() > 5 else 0
def burst_speed(t): return 32.0 <= t < 32.6 or 42.5 <= t < 43.9

# ---------------------------------------------------------------- post
dirt = np.zeros((H // 4, W // 4), np.float32); r2 = np.random.default_rng(46)
for _ in range(46):
    cv2.circle(dirt, (int(r2.uniform(0, W // 4)), int(r2.uniform(0, H // 4))), int(r2.uniform(6, 30)), float(r2.uniform(0.3, 1)), -1)
dirt = cv2.GaussianBlur(dirt, (0, 0), 8); dirt /= dirt.max()
def post(bufs, t):
    img = SCRL[0] + cv2.GaussianBlur(bufs[0], (0, 0), 0.6) + cv2.GaussianBlur(bufs[1], (0, 0), 1.1) + cv2.GaussianBlur(bufs[2], (0, 0), 2.2)
    half = cv2.filter2D(bufs[3], -1, K3) + cv2.filter2D(bufs[4], -1, K4)
    img += cv2.resize(half, (W, H), interpolation=cv2.INTER_LINEAR) * 0.25
    img *= dim(t)
    q = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bloom = 0.15 * cv2.GaussianBlur(q, (0, 0), 2) + 0.10 * cv2.GaussianBlur(q, (0, 0), 7) + 0.15 * cv2.GaussianBlur(q, (0, 0), 24) * (0.55 + 0.85 * dirt)
    hi = np.maximum(q - 0.52, 0)
    env = 0.5 + 0.3 * (0.5 + 0.5 * np.cos(TAU * t / 2.0)) ** 3
    flare = 1 + 1.3 * np.exp(-(t - 32.0) / 0.35) * (t >= 32.0)
    streak = (cv2.GaussianBlur(hi, (0, 0), sigmaX=14, sigmaY=1) + 0.5 * cv2.GaussianBlur(hi, (0, 0), sigmaX=55, sigmaY=1)) * 0.3 * env * flare
    img += cv2.resize(bloom + streak, (W, H), interpolation=cv2.INTER_LINEAR)
    img = 1 - np.exp(-1.6 * img)
    if 32.0 <= t < 33.4:  # shockwave refraction, payoff only
        k = t - 32.0; cx, cy = SHOCK_C
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dx, dy = xx - cx, yy - cy; rr = np.sqrt(dx * dx + dy * dy) + 1e-3
        front = 1.55 * W * k; wd = 0.075 * W; g = (rr - front) / wd
        disp = 16 * np.exp(-k / 0.55) * (-2 * g * np.exp(-g * g))
        img = cv2.remap(img, (xx + dx / rr * disp).astype(np.float32), (yy + dy / rr * disp).astype(np.float32), cv2.INTER_LINEAR)
        img += 0.0 * np.exp(-k / 0.55) * np.exp(-((rr - front) / (0.009 * W)) ** 2)
    return img
SHOCK_C = (W / 2, H * 0.55)

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
    for ch in text:
        dr.text((x, size * 0.1), ch, font=F, fill=255); x += F.getlength(ch) + track * size
    a = np.asarray(im, np.float32) / 255; _wc[key] = (a, x)
    return _wc[key]
def layout(text, name, size, track, x0, y0, maxw=822, lh=1.12):
    words, x, y, out = text.split(' '), x0, y0, []
    sp = font(name, size).getlength(' ')
    for w in words:
        a, adv = word_img(w, name, size, track)
        if x > x0 and x + adv - x0 > maxw: x, y = x0, y + size * lh
        out.append((a, x, y)); x += adv + sp
    return out, y + size * lh
DIM, SEC, WHT = 0.706, 0.541, 1.0
def block(lines, x0, y0, gap=10):
    """lines: (text, font, size, track, color). returns list of (img, x, y, color) and bottom y"""
    out, y = [], y0
    for text, name, size, track, col in lines:
        ws, y = layout(text, name, size, track, x0, y)
        out += [(a, x, yy, col) for a, x, yy in ws]; y += gap
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
       ('“The biggest video launch of all time.”', 'InterDisplay-Medium.otf', 56, -0.02, DIM),
       ('— Rockstar, via THR, May 2025', 'Inter-Medium.otf', 30, 0, SEC)]
IDEA = [('Mar 2026: Spider-Man took it. 718.6M.', 'InterDisplay-SemiBold.otf', 72, -0.02, WHT),
        ('· per Sony · Brand New Day trailer', 'Inter-Medium.otf', 30, 0, SEC)]
END = [('YOUR TURN —', 'Inter-SemiBold.otf', 52, 0.04, DIM),
       ('Trailer 1 or 2: which hooked you?', 'InterDisplay-SemiBold.otf', 72, -0.02, WHT)]
TX = 88; BAND_Y = 1300; PANEL = (56, 1170, 928, 1600)

def reveal(items, t, t_in, stagger, t_out=None, out_dur=0.3, full_at_start=False):
    res = []
    for i, (a, x, y, col) in enumerate(items):
        p = 1.0 if full_at_start else E_IN(seg(t, t_in + i * stagger, t_in + i * stagger + 0.45))
        al = p; dy = 18 * (1 - p)
        if t_out is not None:
            q = E_OUT(seg(t, t_out, t_out + out_dur)); al *= 1 - q; dy -= 8 * q
        if al > 0.003: res.append((a, x, y + dy, col, al))
    return res

def counter_items(t):
    cell = font('InterDisplay-SemiBold.otf', 96)
    def digits(s, x0, y0, col):
        cw = max(cell.getlength(c) for c in '0123456789'); out, x = [], x0
        for ch in s:
            a, adv = word_img(ch, 'InterDisplay-SemiBold.otf', 96, 0)
            w = cw if ch.isdigit() else adv
            out.append((a, x + (w - adv) / 2, y0, col)); x += w
        return out
    cap = word_img('YouTube views · first 24 h', 'Inter-Medium.otf', 30, 0)[0]
    res = []
    if 8.0 <= t < 16.6:
        v = 90.4 * smooth(seg(t, 9.4, 11.2)) if t < 16.3 else 90.4 * (1 - smooth(seg(t, 16.3, 16.6)))
        its = digits(f'{v:.1f}M', TX, 1420, WHT) + [(cap, TX, 1532, SEC)]
        wipe = E_IN(seg(t, 8.0, 8.5)); fade = 1 - E_OUT(seg(t, 16.3, 16.6))
        for a, x, y, c in its:
            al = clamp((wipe * 900 - (x - TX)) / 120) * fade  # soft left->right wipe
            pulse = 1 - 0.2 * np.sin(np.pi * clamp((t - 11.2) / 0.4)) ** 2
            if al > 0: res.append((a, x, y, c, al * pulse))
    if 22.8 <= t < 24.3:
        its = digits('< 90.4M', TX, 1420, DIM) + [(cap, TX, 1532, SEC)]
        res += reveal(its, t, 22.8, 0.04, 23.95, 0.3)
    return res

def text_items(t):
    out = [(word_img('ARCHIVE #010 · GTA VI · PART 4', 'Inter-Medium.otf', 30, 0.08)[0], TX, 200, SEC, 1.0)]
    hk, _ = block(HOOK, TX, BAND_Y)
    if t < 3.8: out += reveal(hk, t, 0, 0, 3.0, 0.8, full_at_start=True)
    if t >= 42.6: out += reveal(hk, t, 42.6, 0.08)
    for k, (tc, txt, src) in enumerate(CARDS):
        t_end = CARDS[k + 1][0] if k + 1 < len(CARDS) else 31.75
        if tc - 0.1 <= t < t_end + 0.3:
            its, _ = block([(txt, 'Inter-Medium.otf', 52, -0.01, WHT), (src, 'Inter-Medium.otf', 30, 0, SEC)], TX, 1205)
            n = len(its); st = min(0.07, 0.20 / max(n - 1, 1))
            out += reveal(its, t, tc + 0.1, st, t_end - 0.05 if k + 1 < len(CARDS) else 31.5, 0.3 if k + 1 < len(CARDS) else 0.25)
    out += counter_items(t)
    if 32.3 <= t < 36.2:
        its, _ = block(PAY, TX, BAND_Y - 40, gap=14); out += reveal(its, t, 32.1, 0.12, 35.8, 0.3)
    if 36.1 <= t < 38.5:
        its, _ = block(IDEA, TX, BAND_Y); out += reveal(its, t, 36.1, 0.16, 38.15, 0.3)
    if 38.5 <= t < 42.45:
        its, _ = block(END, TX, BAND_Y - 20, gap=12); out += reveal(its, t, 38.5, 0.16, 42.0, 0.4)
    return out

# glass panel (static geometry, precomputed)
px0, py0, px1, py1 = PANEL; pw, ph = px1 - px0, py1 - py0
pm = np.zeros((ph, pw), np.uint8); cv2.rectangle(pm, (24, 0), (pw - 25, ph - 1), 255, -1); cv2.rectangle(pm, (0, 24), (pw - 1, ph - 25), 255, -1)
for cx, cy in ((24, 24), (pw - 25, 24), (24, ph - 25), (pw - 25, ph - 25)): cv2.circle(pm, (cx, cy), 24, 255, -1)
PDT = cv2.distanceTransform(pm, cv2.DIST_L2, 5); PMASK = np.clip(PDT / 1.5, 0, 1).astype(np.float32)
edgew = np.clip(1 - PDT / 40, 0, 1) ** 5
gy, gx = np.mgrid[0:ph, 0:pw].astype(np.float32); ccx, ccy = pw / 2, ph / 2
PMX = (gx - (gx - ccx) * 0.20 * edgew).astype(np.float32); PMY = (gy - (gy - ccy) * 0.20 * edgew).astype(np.float32)
RIM = np.exp(-((PDT - 1.5) / 1.2) ** 2).astype(np.float32)
SHEEN = np.clip(1 - np.abs((gx / pw - gy / ph * 0.4) - 0.3) * 4, 0, 1).astype(np.float32) * 0.025
def panel_bottom(t):
    big = smooth(seg(t, 7.75, 8.3)) - smooth(seg(t, 16.3, 16.85)) + smooth(seg(t, 22.3, 22.8)) - smooth(seg(t, 24.0, 24.55))
    return int(round(1395 + (1600 - 1395) * clamp(big)))
_pm = {}
def pmask(ph):
    if ph in _pm: return _pm[ph]
    m = np.zeros((ph, pw), np.uint8); cv2.rectangle(m, (24, 0), (pw - 25, ph - 1), 255, -1); cv2.rectangle(m, (0, 24), (pw - 1, ph - 25), 255, -1)
    for cx, cy in ((24, 24), (pw - 25, 24), (24, ph - 25), (pw - 25, ph - 25)): cv2.circle(m, (cx, cy), 24, 255, -1)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5); e = np.clip(1 - dt / 40, 0, 1) ** 5
    gy, gx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    _pm[ph] = (np.clip(dt / 1.5, 0, 1).astype(np.float32), (gx - (gx - pw / 2) * 0.20 * e).astype(np.float32),
               (gy - (gy - ph / 2) * 0.20 * e).astype(np.float32), np.exp(-((dt - 1.5) / 1.2) ** 2).astype(np.float32),
               (np.clip(1 - np.abs((gx / pw - gy / ph * 0.4) - 0.3) * 4, 0, 1) * 0.025).astype(np.float32), gx, gy)
    return _pm[ph]
def panel_alpha(t): return E_IN(seg(t, 3.9, 4.4)) * (1 - E_OUT(seg(t, 31.45, 31.75)))
def glass(img, t):
    a = panel_alpha(t)
    if a <= 0: return img
    pb = panel_bottom(t); ph = pb - py0
    PMASK, PMX, PMY, RIM, SHEEN, gx, gy = pmask(ph)
    reg = img[py0:pb, px0:px1]
    g = cv2.GaussianBlur(cv2.remap(reg, PMX, PMY, cv2.INTER_LINEAR), (0, 0), 3.2) * 0.36 + 0.03
    sweep = 0
    for tc, _, _ in CARDS:  # one light sweep per header change
        k = seg(t, tc + 0.3, tc + 0.9)
        if 0 < k < 1: sweep = 0.06 * np.exp(-(((gx / pw - gy / ph * 0.5) - (1.6 * k - 0.3)) / 0.06) ** 2)
    g = g + 0.22 * RIM + SHEEN + sweep
    m = PMASK * a
    img[py0:pb, px0:px1] = reg * (1 - m) + g * m
    return img

def compose_text(img, t):
    for a, x, y, col, al in text_items(t):
        x, y = int(round(x)), int(round(y)); h, w = a.shape
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x1 <= x0 or y1 <= y0: continue
        aa = a[y0 - y:y1 - y, x0 - x:x1 - x] * al
        if 32.1 <= t < 36 and col == WHT and 33.3 < t < 34.1:  # one text shine across the payoff
            sx = (t - 33.3) / 0.8 * 1100 - 100
            xs = np.arange(x0, x1)[None, :]
            col = WHT
            img[y0:y1, x0:x1] += aa * 0.25 * np.exp(-((xs - sx) / 60) ** 2)
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - aa) + col * aa
    return img

# ---------------------------------------------------------------- frame
def render(t):
    t = t % T
    bufs = []; SCRL[0] = np.zeros((H, W), np.float32)
    sh = (0, 0)
    if 32.0 <= t < 32.5:
        k = t - 32.0; amp = 3 * np.exp(-k / 0.10)
        sh = (amp * np.sin(TAU * 31 * k), amp * np.cos(TAU * 23 * k + 0.7))
    if cam_speed(t) > 2 or burst_speed(t):
        for i in range(16):
            accumulate(t + (i / 15 - 0.5) / 120.0, bufs, 1 / 16, sh)
    else:
        accumulate(t, bufs, 1.0, sh)
    lv = new_bufs()
    splat(lv, *[np.concatenate([b[i] for b in bufs]) for i in range(4)])
    img = post(lv, t)
    img *= VIG
    img = glass(img, t)
    img = compose_text(img, t)
    fi = int(round(t * FPS))
    dith = (hsh(np.arange(H * W, dtype=np.uint64) + np.uint64(fi * 7)).reshape(H, W) - 0.5) / 255 if False else 0
    rgb = np.clip(img * 255 + DITHER[fi % 4], 0, 255).astype(np.uint8)
    return np.repeat(rgb[:, :, None], 3, 2)
VIG = (1 - 0.72 * np.clip(1 - np.arange(H) / 320, 0, 1) ** 1.5)[:, None].astype(np.float32)  # keeps the label legible
DITHER = [np.random.default_rng(1000 + i).uniform(-0.5, 0.5, (H, W)).astype(np.float32) for i in range(4)]

def encode_seg(i):
    a, b = i * 300, min((i + 1) * 300, NF)
    out = os.path.join(HERE, 'seg', f'seg{i:02d}.mkv')
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-c:v', 'libx264', '-qp', '0', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for f in range(a, b): p.stdin.write(render(f / FPS).tobytes())
    p.stdin.close(); p.wait()

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'still':
        for s in sys.argv[2:]:
            t = float(s); t0 = time.time(); im = render(t)
            cv2.imwrite(os.path.join(HERE, 'stills', f's_{t:05.2f}.png'), im); print(s, f'{time.time() - t0:.2f}s', 'speed', round(cam_speed(t), 2))
    elif cmd == 'seg': encode_seg(int(sys.argv[2]))
    elif cmd == 'info': print('seats', NS, 'city', NC, 'screen', NSC, 'R90', R90, 'R365', R365)
