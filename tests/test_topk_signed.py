"""Properties of the topk_signed write operator.

topk_signed exists to isolate *placement*: top-k by |W_fast|, signs kept, rescaled so the
write carries exactly the same total energy as `direct` (gamma*||W_fast||). If either of
those two properties breaks, the arm silently stops being an energy-matched placement test
and any conclusion drawn from it is void -- so they are asserted here rather than trusted.

The stub model/state below exists only to call the real `variant_sleep` from audit_b3; it is
the function under test, not a reimplementation of it.

Run: ~/.venv/bin/python tests/test_topk_signed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis" / "wfast_geom"))

import audit_b3 as A  # noqa: E402

GAMMA = 0.15  # the config default for consolidate_fast_direct


class _Tempos:
    def values(self):
        return {"consolidate_fast_direct": torch.tensor(GAMMA),
                "consolidate_beta": torch.tensor(0.3),
                "consolidate_fast_decay": torch.tensor(0.5),
                "consolidate_q_decay": torch.tensor(0.7)}


class _Mod:
    def __init__(self, w):
        self.weight = w


class _Model:
    """Only what variant_sleep touches: tempos and key_modules[].weight."""

    def __init__(self, store):
        self.tempos = _Tempos()
        self.key_modules = {k: _Mod(v["weight"]) for k, v in store.items()}


class _State:
    def __init__(self, store):
        self.store = store

    def reset_moments(self):
        pass


def _fresh(shape=(6, 4), seed=0):
    g = torch.Generator().manual_seed(seed)
    wf = torch.randn(*shape, generator=g)
    store = {"m": {"w_fast": wf.clone(), "q": torch.zeros_like(wf),
                   "weight": torch.zeros_like(wf)}}
    return wf, store


def _write(frac, shape=(6, 4), seed=0):
    """The increment variant_sleep actually adds, for a given frac."""
    wf, store = _fresh(shape, seed)
    model, state = _Model(store), _State(store)
    A.variant_sleep(model, state, "topk_signed", topk_frac=frac)
    return wf, model.key_modules["m"].weight


def test_energy_matches_direct_at_every_frac():
    """The write must carry exactly direct's energy: placement is the only variable."""
    for frac in (0.05, 0.2, 0.5, 0.95, 1.0):
        wf, add = _write(frac)
        target = GAMMA * float(wf.norm())
        got = float(add.norm())
        assert abs(got - target) <= 1e-12 * max(1.0, target), (
            f"frac={frac}: ||add||={got!r} != gamma*||wf||={target!r}")


def test_signs_are_preserved_where_kept():
    """topwrite's defect was dropping the sign; this is the property that fixes it."""
    for frac in (0.05, 0.2, 0.5):
        wf, add = _write(frac)
        kept = add != 0
        assert bool(kept.any()), f"frac={frac}: nothing was written"
        assert torch.equal(torch.sign(add[kept]), torch.sign(wf[kept])), (
            f"frac={frac}: sign pattern not preserved")


def test_frac_one_reduces_to_direct():
    """frac=1.0 keeps every coordinate, so it must equal `direct` exactly -- the self-check."""
    wf, add = _write(1.0)
    assert torch.allclose(add, GAMMA * wf, atol=1e-12), "frac=1.0 is not `direct`"
    # And `direct` itself must agree with the same statement of intent.
    _, store = _fresh()
    model, state = _Model(store), _State(store)
    A.variant_sleep(model, state, "direct")
    assert torch.allclose(model.key_modules["m"].weight, GAMMA * wf, atol=1e-12)


def test_concentration_is_actually_applied():
    """A frac-k write must zero at least (1-frac) of the coordinates, or the arm is a no-op."""
    for frac in (0.05, 0.2, 0.5):
        wf, add = _write(frac, shape=(20, 20))
        kept = int((add != 0).sum())
        n = wf.numel()
        assert kept == max(1, round(frac * n)), f"frac={frac}: kept {kept} of {n}"


def test_topwrite_still_drops_the_sign():
    """The defect topk_signed was written to replace: documented, not assumed away."""
    wf, store = _fresh()
    model, state = _Model(store), _State(store)
    A.variant_sleep(model, state, "topwrite")
    add = model.key_modules["m"].weight
    kept = add != 0
    assert bool(kept.any())
    assert bool((add[kept] > 0).all()), "topwrite no longer writes a constant magnitude"
    assert not torch.equal(torch.sign(add[kept]), torch.sign(wf[kept])), (
        "topwrite now preserves signs -- if this fails the old defect is gone and "
        "the topk_signed-vs-topwrite contrast in summary_topk.py no longer means what "
        "it says")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
