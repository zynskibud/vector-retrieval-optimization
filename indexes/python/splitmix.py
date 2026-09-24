"""SplitMix64 PRNG, identical in all four languages (CONTRACT 5)."""

MASK = (1 << 64) - 1


def new(seed: int) -> dict:
    """Return a generator state seeded with `seed`."""
    return {"s": seed & MASK}


def next_u64(rng: dict) -> int:
    rng["s"] = (rng["s"] + 0x9E3779B97F4A7C15) & MASK
    z = rng["s"]
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def next_f64(rng: dict) -> float:
    """Float in [0, 1)."""
    return (next_u64(rng) >> 11) * 2.0**-53


def next_below(rng: dict, n: int) -> int:
    return next_u64(rng) % n
