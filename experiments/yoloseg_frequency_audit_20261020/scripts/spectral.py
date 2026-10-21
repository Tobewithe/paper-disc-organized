"""Physical-frequency, orthonormal FFT summaries; CPU only."""
from functools import lru_cache
import numpy as np
from scipy.fft import rfft2, irfft2

BANDS = ("low", "mid", "high")


@lru_cache(maxsize=48)
def geometry(h, w, dy, dx):
    q = 640 * np.hypot(np.fft.fftfreq(h, d=dy)[:, None],
                       np.fft.rfftfreq(w, d=dx)[None, :])
    band = np.where(q < 8, 0, np.where(q < 32, 1, 2))
    multiplicity = np.ones((h, w // 2 + 1))
    multiplicity[:, 1: (w // 2 if w % 2 == 0 else w // 2 + 1)] = 2
    active = q > 0
    bins = np.minimum((q * 48 / 160).astype(int), 48)
    hann = np.outer(np.hanning(h), np.hanning(w))
    counts = np.bincount(bins[active], multiplicity[active], minlength=49)
    return band, multiplicity, active, bins, hann, counts


def spectrum(x, spacing, window=False):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[None]
    n, h, w = x.shape
    band, mult, active, bins, hann, counts = geometry(h, w, *map(float, spacing))
    if window:
        mean = (x * hann).sum((-2, -1), keepdims=True) / hann.sum()
        centered = (x - mean) * hann
        centered -= centered.mean((-2, -1), keepdims=True)
    else:
        mean = x.mean((-2, -1), keepdims=True)
        centered = x - mean
    f = rfft2(centered, norm="ortho", workers=1)
    power = (f.real ** 2 + f.imag ** 2) * mult
    power[:, 0, 0] = 0
    e = np.stack([power[:, (band == i) & active].sum(1) for i in range(3)], 1)
    total = e.sum(1)
    spatial = (centered ** 2).sum((-2, -1))
    err = np.max(np.abs(total - spatial) / np.maximum(1, spatial))
    assert err < 1e-10, ("Parseval", err)
    fractions = np.divide(e, total[:, None], out=np.full_like(e, np.nan), where=total[:, None] > 1e-20)
    radial = np.stack([np.bincount(bins[active], p[active], minlength=49) for p in power])
    radial = np.divide(radial, total[:, None], out=np.full_like(radial, np.nan), where=total[:, None] > 1e-20)
    return dict(energy=e, fraction=fractions, total=total,
                dc=(mean.reshape(n) ** 2 * h * w) if not window else np.zeros(n),
                radial=radial, counts=counts, parseval=float(err))


def split_bands(x, spacing):
    h, w = x.shape
    band, _, active, _, _, _ = geometry(h, w, *map(float, spacing))
    f = rfft2(np.asarray(x, dtype=np.float64), norm="ortho", workers=1)
    pieces = [irfft2(f * ((band == i) & active), s=(h, w), norm="ortho", workers=1) for i in range(3)]
    dc = float(x.mean())
    assert np.max(np.abs(sum(pieces) + dc - x)) < 1e-10
    return pieces, dc


def tests():
    h = w = 128
    yy, xx = np.mgrid[:h, :w]
    for k, expected in [(1, 0), (4, 1), (12, 2)]:
        a = np.cos(2 * np.pi * k * xx / w)
        s = spectrum(a, (1, 1))
        assert s["fraction"][0, expected] > 1 - 1e-12
        parts, dc = split_bands(a, (1, 1))
        assert abs(dc) < 1e-12
    a = np.cos(2 * np.pi * 4 * xx / w)
    assert spectrum(a, (1, 4))["fraction"][0, 0] > 1 - 1e-12
    const = spectrum(np.ones((h, w)) * 2, (1, 1))
    assert const["total"][0] == 0 and const["dc"][0] == h * w * 4
    assert np.isnan(spectrum(np.zeros((h, w)), (1, 1))["fraction"]).all()
    rng = np.random.default_rng(1729)
    noise = rng.normal(size=(3, 61, 77))
    for window in [False, True]:
        s = spectrum(noise, (1.25, .8), window)
        assert np.isfinite(s["fraction"]).all()
        assert np.allclose(s["fraction"].sum(1), 1, atol=1e-12)
    split_bands(noise[0], (1.25, .8))
    return {"status": "passed", "checks": ["three_sinusoids", "anisotropic_spacing",
            "DC", "zero_signal", "Parseval", "odd_dimensions", "three_band_reconstruction", "Hann"]}
