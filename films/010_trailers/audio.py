"""Score + SFX for ARCHIVE #010 (B minor, 120 BPM, 22 bars = 44.0 s). Writes audio/full.wav and audio/sfx.wav (48 kHz, 24-bit)."""
import numpy as np, os, subprocess
from scipy import signal
from scipy.io import wavfile

SR, T, BPM = 48000, 44.0, 120.0
N = int(SR * T); BEAT = 60 / BPM; BAR = 4 * BEAT; S16 = BEAT / 4
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(2026)

def z(): return np.zeros((N, 2))
def place(buf, x, t0, gain=1.0, pan=0.0):
    """circular placement (loop-safe); x mono or stereo"""
    if x.ndim == 1:
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        x = np.stack([x * l, x * r], 1) * np.sqrt(2)
    i0 = int(round(t0 * SR)) % N; n = len(x)
    idx = (np.arange(n) + i0) % N
    np.add.at(buf, idx, x * gain)
def env(n, a, d, curve=4.0):
    t = np.arange(n) / SR; e = np.minimum(t / max(a, 1e-4), 1) * np.exp(-curve * np.maximum(t - a, 0) / max(d, 1e-4))
    return e
def bp(x, lo, hi, order=2): return signal.sosfilt(signal.butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)
def lp(x, f, order=2): return signal.sosfilt(signal.butter(order, f, 'low', fs=SR, output='sos'), x)
def hp(x, f, order=2): return signal.sosfilt(signal.butter(order, f, 'high', fs=SR, output='sos'), x)
def tiled(fn, x):  # IIR on 3x tiled signal, keep middle (loop seam)
    y = fn(np.concatenate([x, x, x], 0)); return y[len(x):2 * len(x)]
def nf(name):  # note name -> Hz
    names = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
    return 440 * 2 ** ((names[name[:-1]] + 12 * (int(name[-1]) - 4)) / 12)
def blep_saw(f, n, phase=0.0):
    ph = (phase + np.cumsum(np.full(n, f / SR) if np.isscalar(f) else f / SR)) % 1.0
    dt = (f if not np.isscalar(f) else np.full(n, f)) / SR
    y = 2 * ph - 1
    m1 = ph < dt; tt = ph[m1] / dt[m1]; y[m1] -= tt + tt - tt * tt - 1
    m2 = ph > 1 - dt; tt = (ph[m2] - 1) / dt[m2]; y[m2] -= tt * tt + tt + tt + 1
    return y
def square(f, n, duty=0.5, phase=0.0):
    a = blep_saw(f, n, phase); b = blep_saw(f, n, (phase + duty) % 1)
    return 0.5 * (a - b)
def tv_lp(x, cut, q=0.7, block=32):  # time-varying resonant lowpass, chunked biquads
    y = np.zeros_like(x); zi = np.zeros(2)
    for i in range(0, len(x), block):
        f = float(np.clip(cut[min(i, len(cut) - 1)], 30, SR * 0.45))
        w = 2 * np.pi * f / SR; al = np.sin(w) / (2 * q); c = np.cos(w)
        b = np.array([(1 - c) / 2, 1 - c, (1 - c) / 2]); a = np.array([1 + al, -2 * c, 1 - al])
        y[i:i + block], zi = signal.lfilter(b / a[0], a / a[0], x[i:i + block], zi=zi)
    return y
def sat(x, k=2.0): return np.tanh(k * x) / np.tanh(k)

# ------------------------------------------------------------ instruments
def kick(muffle=False):
    n = int(0.45 * SR); t = np.arange(n) / SR
    f = 61.7 + 110 * np.exp(-t / 0.03)  # tuned to B1
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    click = hp(rng.standard_normal(n), 2000) * np.exp(-t / 0.004) * 0.25
    x = body + click; x += 0.35 * bp(sat(x, 3), 150, 2500)  # phone layer
    return lp(x, 800) * 1.3 if muffle else x
def hat(open_=False):
    n = int((0.25 if open_ else 0.06) * SR); t = np.arange(n) / SR
    x = sum(square(f, n) for f in (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)) / 6
    return hp(x, 7000, 4) * np.exp(-t / (0.08 if open_ else 0.018))
def clap():
    n = int(0.5 * SR); t = np.arange(n) / SR; x = np.zeros(n)
    for d in (0, 0.011, 0.022):
        i = int(d * SR); m = n - i
        x[i:] += bp(rng.standard_normal(m), 900, 2600) * np.exp(-t[:m] / 0.007)
    x += bp(rng.standard_normal(n), 900, 2200) * np.exp(-t / 0.12) * 0.5
    return x
def rim():
    n = int(0.08 * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 1650 * t) + 0.5 * bp(rng.standard_normal(n), 2000, 6000)) * np.exp(-t / 0.012)
def snare():
    n = int(0.25 * SR); t = np.arange(n) / SR
    return (0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.04) + bp(rng.standard_normal(n), 1500, 8000) * np.exp(-t / 0.07))
def note(fn, f, dur, a=0.005, rel=0.05):
    n = int((dur + rel) * SR); x = fn(f, n); t = np.arange(n) / SR
    e = np.minimum(t / a, 1) * np.clip((dur + rel - t) / rel, 0, 1)
    return x * e
def supersaw(f, n):
    return sum(blep_saw(f * 2 ** (c / 1200), n, rng.random()) for c in np.linspace(-28, 28, 7)) / 7
def detuned_sq(f, n): return 0.5 * (square(f * 2 ** (12 / 1200), n, 0.5, 0.1) + square(f * 2 ** (-12 / 1200), n, 0.5, 0.6))
def pulse(f, n): return square(f, n, 0.3)

CH = [('B1', ['B3', 'D4', 'F#4']), ('G1', ['G3', 'B3', 'D4']), ('D2', ['D4', 'F#4', 'A4']), ('A1', ['A3', 'C#4', 'E4'])]
def chord(b): return CH[b % 4]

# ------------------------------------------------------------ arrangement
def kick_times():
    ks = []
    for b in range(22):
        t0 = b * BAR
        if b in (0, 1, 19, 20, 21): ks += [(t0 + i * BEAT, True) for i in range(4)]       # muffled intro/breakdown
        elif b in (8, 9): ks += [(t0, False), (t0 + 2 * BEAT, False)]                      # half-time
        elif b == 15: ks += [(t0 + i * BEAT, False) for i in range(0)]                     # kick out at 30.0
        elif b == 14: ks += [(t0 + i * BEAT, False) for i in range(4)]
        else: ks += [(t0 + i * BEAT, False) for i in range(4)]
    return [(t, m) for t, m in ks if not (6 * BAR <= t < 6 * BAR + BEAT)]  # bass/kick drop-out at 12.0
KICKS = kick_times()

def music():
    bus = {k: z() for k in ('kick', 'bass', 'hat', 'clap', 'rim', 'pad', 'arp', 'acid', 'lead', 'stab', 'roll')}
    K, KM = kick(), kick(True)
    for t, m in KICKS: place(bus['kick'], KM if m else K, t)
    for b in range(22):
        t0 = b * BAR; root, tones = chord(b)
        # pad (every bar, equal-power xfade by overlap)
        if b not in (16, 17, 18) or True:
            for k, nn in enumerate(tones):
                x = note(lambda f, n: blep_saw(f, n, rng.random()) + blep_saw(f * 1.004, n, rng.random()), nf(nn), BAR, 0.3, 0.3)
                place(bus['pad'], x, t0, 0.10, (-0.3, 0, 0.3)[k])
        # filtered arp: intro, wait, breakdown
        if b in (0, 1, 6, 7, 8, 9, 18, 19, 20, 21):
            seq = tones + [tones[1].replace('4', '5').replace('3', '4')]
            for i in range(16):
                place(bus['arp'], note(lambda f, n: square(f, n, 0.25), nf(seq[i % 4]) * 2, S16 * 0.7, 0.002, 0.03), t0 + i * S16, 0.11, 0.25 * np.sin(i))
        # rolling bass 16ths off the kick
        if 2 <= b <= 18 and b not in (8, 9, 15):
            for i in range(16):
                if i % 4 == 0: continue
                if b == 6 and i < 4: continue  # drop-out beat at 12.0
                place(bus['bass'], note(pulse, nf(root) * (2 if i % 4 == 2 else 1), S16 * 0.8, 0.003, 0.02), t0 + i * S16, 0.5)
        if b in (8, 9):  # half-time bass: long notes
            place(bus['bass'], note(pulse, nf(root), BAR * 0.9, 0.01, 0.1), t0, 0.45)
        # hats (swung 16ths)
        if 2 <= b <= 18 and b != 15:
            for i in range(16):
                sw = 0.08 * S16 if i % 2 else 0
                if b in (8, 9) and i % 2: continue
                place(bus['hat'], hat(i % 4 == 2), t0 + i * S16 + sw, 0.9 if i % 4 == 2 else 0.5, 0.2)
        if 4 <= b <= 18 and b not in (8, 9, 15):
            for i in (1, 3): place(bus['clap'], clap(), t0 + i * BEAT, 0.8)
        if 12 <= b <= 17 and b != 15:
            for i in (3, 7, 11, 14): place(bus['rim'], rim(), t0 + i * S16, 0.5, -0.2)
        # acid: race + drop (filtered by section)
        if 10 <= b <= 18 and b != 15:
            pat = [0, 0, 12, 0, 3, 0, 0, 7, 0, 12, 0, 3, 0, 10, 0, 7]
            f0 = nf(root) * 2
            for i, st in enumerate(pat):
                acc = i % 4 == 0
                x = note(blep_saw, f0 * 2 ** (st / 12), S16 * 0.9, 0.002, 0.02)
                place(bus['acid'], x, t0 + i * S16, 0.28 * (1.3 if acc else 1), 0.15)
    # lead riff (detuned square): race + drop
    riff = [('F#5', 0, 1), ('D5', 1, 1), ('B4', 2, 2), ('C#5', 4, 1), ('D5', 5, 1), ('E5', 6, 1), ('D5', 7, 1)]
    for b in list(range(12, 15)) + [16, 17, 18]:
        for nn, st, ln in riff:
            place(bus['lead'], note(detuned_sq, nf(nn), ln * BEAT * 0.9, 0.005, 0.08), b * BAR + st * BEAT / 2 * 1, 0.13, 0.1)
    # supersaw stabs on the drop (offbeat 8ths) + signature 3-note motif at 32.0
    for b in (16, 17, 18):
        root, tones = chord(b)
        for i in range(8):
            if i % 2 == 1 or i == 0:
                for nn in tones: place(bus['stab'], note(supersaw, nf(nn) * 2, BEAT * 0.35, 0.003, 0.12), b * BAR + i * BEAT / 2, 0.09, 0)
    for k, (nn, dt) in enumerate((('F#5', 0.0), ('D5', 0.5), ('B4', 1.0))):
        place(bus['stab'], note(supersaw, nf(nn), 0.45, 0.003, 0.3), 32.0 + dt, 0.22)
    # snare roll 30.0 -> 31.75 accelerating
    t, k = 30.0, 0
    while t < 31.75:
        place(bus['roll'], snare(), t, 0.25 + 0.5 * (t - 30) / 1.75); t += max(0.04, 0.25 * 0.85 ** k); k += 1
    # bus processing
    sec_filter = lambda t: 800 if t < 4.0 or t >= 38.0 else 20000
    cut = np.array([sec_filter(i / SR) for i in range(0, N, 32)]).repeat(32)[:N]
    for name in ('arp', 'pad'):
        bus[name] = np.stack([tiled(lambda x: lp(x, 3000), bus[name][:, c]) for c in range(2)], 1)
    bus['pad'] = np.stack([tiled(lambda x: hp(x, 150), bus['pad'][:, c]) for c in range(2)], 1)
    acut = np.array([(400 + 2600 * min(1, max(0, (i / SR - 20) / 12))) if i / SR < 32 else 3500 for i in range(0, N, 32)]).repeat(32)[:N]
    bus['acid'] = np.stack([tv_lp(sat(bus['acid'][:, c], 1.5), acut, 3.5) for c in range(2)], 1)
    bus['bass'] = np.stack([lp(bus['bass'][:, c], 900) for c in range(2)], 1)
    bus['bass'] += 0.3 * np.stack([bp(sat(bus['bass'][:, c], 3), 150, 2500) for c in range(2)], 1)
    # sidechain from the kick list
    sc = np.ones(N)
    for t, m in KICKS:
        i = int(t * SR); n = int(0.3 * SR); idx = (np.arange(n) + i) % N
        sc[idx] = np.minimum(sc[idx], 1 - 0.55 * np.exp(-np.arange(n) / SR / 0.09))
    for name in ('bass', 'pad', 'acid', 'stab', 'arp'): bus[name] *= sc[:, None]
    lv = dict(kick=0.9, bass=0.55, hat=0.35, clap=0.45, rim=0.35, pad=0.6, arp=0.5, acid=0.45, lead=0.6, stab=0.55, roll=0.5)
    mix = sum(bus[k] * lv[k] for k in bus)
    # intro/breakdown low-pass (muffled), circular
    mix = np.stack([tv_lp(mix[:, c], cut, 0.7) for c in range(2)], 1)
    # tape-stop 31.75 -> 31.875, silence to 32.0
    a, b = int(31.75 * SR), int(31.875 * SR)
    src = mix[a - int(0.2 * SR):b].copy(); n = b - a
    speed = np.linspace(1, 0.05, n); pos = np.cumsum(speed); pos = pos - pos[0] + int(0.2 * SR)
    for c in range(2): mix[a:b, c] = np.interp(pos, np.arange(len(src)), src[:, c]) * np.linspace(1, 0, n) ** 0.5
    mix[b:int(32.0 * SR)] = 0
    return mix

# ------------------------------------------------------------ SFX (transient + body + tail)
def noise(n): return rng.standard_normal(n)
def whoosh(dur, f0=400, f1=4000, rev=False):
    n = int(dur * SR); x = noise(n); t = np.arange(n) / SR
    fc = f0 * (f1 / f0) ** (t / dur); y = tv_lp(x, fc * 1.5, 1.2) - tv_lp(x, fc * 0.6, 0.7)
    e = np.sin(np.pi * t / dur) ** 2; y = y * e
    return y[::-1] if rev else y
def ding(f=1975.5, dec=0.9):
    n = int(1.4 * SR); t = np.arange(n) / SR
    body = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (dec / r ** 0.5)) for r, a in ((1, 1), (2.76, 0.4), (5.4, 0.2)))
    return (body + hp(noise(n), 3000) * np.exp(-t / 0.003) * 0.5) * 0.4
def tick(f=2600):
    n = int(0.03 * SR); t = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.004) + 0.4 * bp(noise(n), 2000, 5000) * np.exp(-t / 0.002))
def clatter(dur=0.45):  # projector: 24 Hz shutter clicks + motor hum
    n = int(dur * SR); x = 0.15 * np.sin(2 * np.pi * 96 * np.arange(n) / SR) * np.sin(np.pi * np.arange(n) / n)
    for k in range(int(dur * 24)):
        c = tick(1800 + 300 * (k % 2)) * 0.6; i = int(k / 24 * SR); x[i:i + len(c)] += c[:n - i]
    return x
def reel_start(dur=0.6):  # motor spin-up + clatter speeding up
    n = int(dur * SR); t = np.arange(n) / SR; f = 40 + 80 * (t / dur) ** 0.5
    x = 0.25 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(t / 0.1, 1); ph = 0.0
    while ph < dur:
        c = tick(1700) * 0.5; i = int(ph * SR); x[i:i + len(c)] += c[:n - i]; ph += 1 / (8 + 16 * ph / dur)
    return x
def grains(dur, rising=True, rate=60):
    n = int(dur * SR); x = np.zeros(n)
    for k in range(int(dur * rate)):
        p = k / (dur * rate); f = 900 * 2 ** (2 * p if rising else -2 * p)
        g = np.sin(2 * np.pi * f * np.arange(int(0.02 * SR)) / SR) * np.hanning(int(0.02 * SR))
        i = int(rng.uniform(0, n - len(g))); x[i:i + len(g)] += g * rng.uniform(0.3, 1)
    return x
def stroke(dur):  # pencil-like filtered noise
    n = int(dur * SR); t = np.arange(n) / SR
    return bp(noise(n), 2500, 7000) * (0.6 + 0.4 * np.sin(2 * np.pi * 9 * t)) * np.sin(np.pi * t / dur) ** 0.5 * 0.5
def denied():
    n = int(0.7 * SR); t = np.arange(n) / SR; f = np.where(t < 0.18, 233, 185)
    return (square(f, n, 0.5) * 0.3 + np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.exp(-t / 0.25) * np.minimum(t / 0.005, 1) * 0.5
def riser(dur):
    n = int(dur * SR); t = np.arange(n) / SR; fc = 300 * (6000 / 300) ** (t / dur)
    x = tv_lp(noise(n), fc, 1.5) * 0.6 + 0.2 * tv_lp(blep_saw(110 * 2 ** (2 * t / dur), n), fc, 2)
    return x * (t / dur) ** 1.5
def impact():
    n = int(2.5 * SR); t = np.arange(n) / SR
    shutter = hp(noise(n), 1500) * np.exp(-t / 0.006) * 0.9 + 0.5 * np.sin(2 * np.pi * 3100 * t) * np.exp(-t / 0.01)
    sub = np.sin(2 * np.pi * np.cumsum(60 * (35 / 60) ** np.minimum(t / 0.4, 1)) / SR) * np.exp(-t / 0.5) * 1.1
    burst = lp(noise(n), 3000) * np.exp(-t / 0.08) * 0.5
    mid = bp(np.tanh(4 * sub), 110, 220) * 0.5                       # phone-audible sub layer
    metal = sum(np.sin(2 * np.pi * f * t) for f in (523, 1307, 2213)) / 3 * np.exp(-t / 0.3) * 0.25
    return shutter + sub + mid + burst + metal
def tape_rewind(dur):
    n = int(dur * SR); t = np.arange(n) / SR; f = 300 + 2400 * (t / dur) ** 2
    return (bp(noise(n), 1500, 6000) * 0.4 + 0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)) * np.sin(np.pi * t / dur)

def sfx():
    d, dry = z(), z()  # 'd' gets reverb tail
    place(d, clatter(), 43.95 - T, 0.5)                          # projector clatter across the loop seam
    place(d, reel_start(), 3.85, 0.6); place(d, whoosh(0.5, 4000, 600, True), 3.55, 0.25, -0.2)
    for tc in (8.0, 12.0, 20.0, 24.0, 28.0):                     # card whooshes peak ~0.1 s before the card
        place(d, whoosh(0.45, 800, 5000), tc - 0.33, 0.18, -0.25)
    for t in np.arange(9.375, 11.2 - 2 / 60, S16): place(d, tick(2200 + 900 * (t - 9.4) / 1.8), t, 0.28)   # 1/16 grid, muted near the ding
    place(d, stroke(1.0), 10.4, 0.35, 0.2); place(d, ding(), 11.2, 0.55)
    place(d, ding(1480, 0.6), 12.3, 0.35)
    place(d, grains(0.7, False), 16.3, 0.25); place(d, reel_start(0.5), 16.95, 0.6); place(d, whoosh(0.6, 5000, 400, True), 16.4, 0.15, 0.2)
    for t in np.arange(21.25, 22.8 - 2 / 60, S16): place(d, tick(2200 + 700 * (t - 21.2) / 1.6), t, 0.25)
    place(d, denied(), 22.8, 0.45)
    place(d, whoosh(4.3, 400, 4000), 24.3, 0.35, 0.0); place(d, grains(3.6, True, 90), 24.4, 0.3)
    place(d, stroke(1.2), 28.2, 0.35, -0.2); place(d, ding(2349, 0.7), 29.4, 0.3)   # 365M ring locks
    place(d, ding(4186, 0.25), 33.3, 0.12, -0.3)                                          # payoff text shine glint
    place(d, riser(3.75), 28.0, 0.45)
    place(dry, bp(noise(int(0.125 * SR)), 2000, 9000) * 0.03, 31.875)  # tape-hiss breath in the gap
    place(d, impact(), 32.0, 1.0)
    for tc in (36.1, 38.5): place(d, whoosh(0.4, 900, 4500), tc - 0.3, 0.12, -0.25)
    place(d, tape_rewind(1.4), 42.5, 0.35)
    # circular FFT reverb, energy-normalised IR
    n = int(1.4 * SR); t = np.arange(n) / SR
    ir = noise(n) * np.exp(-t / 0.35); ir = lp(ir, 6000); ir /= np.sqrt(np.sum(ir ** 2))
    IR = np.fft.rfft(np.pad(ir, (0, N - n)))
    wet = np.stack([np.fft.irfft(np.fft.rfft(d[:, c]) * IR, N) for c in range(2)], 1)
    global S_hiss; S_hiss = dry.copy()
    return d + 0.25 * wet + dry

# ------------------------------------------------------------ mastering
def kweight(x):
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=0), axis=0)  # coefficients for 48 kHz
def lufs(x):
    if np.max(np.abs(x)) < 1e-9: return -99
    k = kweight(x); return -0.691 + 10 * np.log10(np.mean(np.sum(k ** 2, 1)) + 1e-12)
def short_term(x, t0, t1): return lufs(x[int(t0 * SR):int(t1 * SR)])
def mono_lows(x):
    lo = np.stack([tiled(lambda s: lp(lp(s, 150), 150), x[:, c]) for c in range(2)], 1).mean(1, keepdims=True)
    hi = np.stack([tiled(lambda s: hp(hp(s, 150), 150), x[:, c]) for c in range(2)], 1)
    return hi + lo
def eq_phone(x):  # -4 dB below 110 Hz, +3 dB around 2.2 kHz
    lo = np.stack([tiled(lambda s: lp(s, 110), x[:, c]) for c in range(2)], 1)
    pres = np.stack([tiled(lambda s: bp(s, 1600, 3000, 1), x[:, c]) for c in range(2)], 1)
    return x - 0.37 * lo + 0.41 * pres
def limit(x, ceil_db):
    c = 10 ** (ceil_db / 20)
    up = signal.resample_poly(x, 4, 1, axis=0); pk = np.abs(up).max(1).reshape(-1, 4).max(1)[:len(x)]
    look = int(0.0015 * SR); g = np.minimum(1, c / np.maximum(pk, 1e-9))
    g = -signal.convolve(np.pad(-g, (look, look), mode='wrap'), np.ones(2 * look + 1) / (2 * look + 1), 'valid') if False else g
    gm = np.array([g[max(0, i - look):i + look + 1].min() for i in range(0, len(g), 1)]) if False else None
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    gm = minimum_filter1d(g, 2 * look + 1, mode='wrap'); gm = uniform_filter1d(gm, look, mode='wrap')
    gm = np.minimum(gm, minimum_filter1d(g, 2 * look + 1, mode='wrap') * 1.0 + 0)
    y = x * gm[:, None]
    for _ in range(3):  # catch residual inter-sample overs
        up = signal.resample_poly(y, 4, 1, axis=0); p = np.abs(up).max()
        if p <= c: break
        y *= c / p
    return y
def tp_db(x): return 20 * np.log10(np.abs(signal.resample_poly(x, 4, 1, axis=0)).max() + 1e-12)

SECTIONS = [(0, 4, -18), (4, 16, -16), (16, 20, -18), (20, 24, -16), (24, 28, -14.5), (28, 32, -16.5), (32, 38, -11), (38, 42, -20), (42, 44, -18)]
def section_gain(m):
    g = np.ones(N); ref = None
    for a, b, tgt in SECTIONS:
        l = short_term(m, a, b); off = tgt - l
        g[int(a * SR):int(b * SR)] = 10 ** (off / 20)
    # 50 ms equal-power ramps at section borders, circular
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(g, int(0.05 * SR), mode='wrap')

def write(path, x):
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 2147483000).astype(np.int32))

if __name__ == '__main__':
    os.makedirs(os.path.join(HERE, 'audio'), exist_ok=True)
    M = music(); S = sfx()
    M = M * section_gain(M)[:, None]
    # SFX bus 6-10 dB under music peaks: measure and set
    S *= 10 ** ((lufs(M) - 9 - lufs(S)) / 20)
    full = mono_lows(eq_phone(M + S))
    g0, g1 = int(31.875 * SR), int(32.0 * SR); full[g0:g1] *= 0.0; full[g0:g1] += S_hiss[g0:g1]   # true silence before the impact
    for _ in range(4):
        full *= 10 ** ((-14 - lufs(limit(full, -3.6))) / 20)
    full = limit(full, -3.6)
    sfx_only = mono_lows(eq_phone(S)); sfx_only *= 10 ** ((-18 - lufs(sfx_only)) / 20); sfx_only = limit(sfx_only, -3.6)
    write(os.path.join(HERE, 'audio', 'full.wav'), full); write(os.path.join(HERE, 'audio', 'sfx.wav'), sfx_only)
    ph = np.stack([bp(full[:, c], 280, 9000, 4) for c in range(2)], 1)
    print('FULL lufs %.2f tp %.2f | SFX lufs %.2f tp %.2f | phone delta %.1f dB' % (lufs(full), tp_db(full), lufs(sfx_only), tp_db(sfx_only), lufs(full) - lufs(ph)))
    for a, b, tgt in SECTIONS: print(f'  {a:>2}-{b:<2} target {tgt} got {short_term(full, a, b):.1f}')
    lo = np.stack([lp(full[:, c], 150, 4) for c in range(2)], 1); print('low-end L/R corr %.3f' % np.corrcoef(lo[:, 0], lo[:, 1])[0, 1])
