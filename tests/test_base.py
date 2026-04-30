"""Tests for mrphantom.base — MRPhantom ABC."""

from __future__ import annotations

import pytest

from mrphantom.base import MRPhantom


# ---------------------------------------------------------------------------
# Concrete stub for testing the abstract base
# ---------------------------------------------------------------------------


class _StubPhantom(MRPhantom):
    """Minimal concrete subclass for testing MRPhantom behaviour."""

    def load_case(self, config) -> "_StubPhantom":
        self._loaded = config
        return self

    def synthesize(self) -> "_StubPhantom":
        return self


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_abstract_cannot_be_instantiated():
    with pytest.raises(TypeError):
        MRPhantom()  # type: ignore[abstract]


def test_stub_instantiated_defaults():
    p = _StubPhantom()
    assert p.snr is None
    assert p.seed is None


def test_stub_instantiated_with_params():
    p = _StubPhantom(snr=30.0, seed=42)
    assert p.snr == 30.0
    assert p.seed == 42


# ---------------------------------------------------------------------------
# get_params
# ---------------------------------------------------------------------------


def test_get_params_default():
    p = _StubPhantom()
    assert p.get_params() == {"snr": None, "seed": None}


def test_get_params_with_values():
    p = _StubPhantom(snr=20.0, seed=7)
    assert p.get_params() == {"snr": 20.0, "seed": 7}


# ---------------------------------------------------------------------------
# set_params
# ---------------------------------------------------------------------------


def test_set_params_returns_self():
    p = _StubPhantom()
    result = p.set_params(snr=50.0)
    assert result is p


def test_set_params_updates_attribute():
    p = _StubPhantom()
    p.set_params(snr=50.0, seed=1)
    assert p.snr == 50.0
    assert p.seed == 1


def test_set_params_unknown_key_raises():
    p = _StubPhantom()
    with pytest.raises(ValueError, match="unknown parameter"):
        p.set_params(nonexistent=99)


def test_set_params_chaining():
    p = _StubPhantom()
    result = p.set_params(snr=10).set_params(seed=0)
    assert result.snr == 10
    assert result.seed == 0


# ---------------------------------------------------------------------------
# load_case / synthesize chaining
# ---------------------------------------------------------------------------


def test_load_case_returns_self():
    p = _StubPhantom()
    result = p.load_case("dummy_config")
    assert result is p


def test_synthesize_returns_self():
    p = _StubPhantom()
    result = p.synthesize()
    assert result is p


def test_method_chaining():
    p = _StubPhantom()
    result = p.load_case("cfg").synthesize()
    assert result is p
    assert p._loaded == "cfg"
