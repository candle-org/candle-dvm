"""Tests for the public Kernel graph-builder API (candle_dvm.api)."""

from candle_dvm import Kernel, float32
from candle_dvm.ops import DTYPE_BOOL


def test_public_kernel_builds_add_graph():
    """Build a simple add graph and verify output shape."""
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    out = k.store(k.add(a, b))
    assert out.shape_ref == (32, 32)


def test_kernel_codegen_produces_valid_code():
    """Codegen should produce a header with expected fields."""
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    k.store(k.add(a, b))
    k.codegen()
    h = k.debug_header()
    assert h["target"] == 0
    assert h["block_dim"] > 0
    assert h["data_size"] > 16


def test_kernel_relocs_count():
    """After codegen, relocs should contain one entry per IO operand."""
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    k.store(k.add(a, b))
    k.codegen()
    # 2 loads + 1 store = 3 relocs
    assert len(k.get_relocs()) == 3


def test_kernel_exposes_sqrt_and_log_methods():
    """Build a unary graph with sqrt and log and verify shapes propagate."""
    k = Kernel()
    a = k.load((16, 16), float32)
    s = k.sqrt(a)
    assert s.shape_ref == (16, 16)
    l = k.log(a)
    assert l.shape_ref == (16, 16)
    # Also verify the other unary methods exist and return correct shapes
    assert k.abs(a).shape_ref == (16, 16)
    assert k.exp(a).shape_ref == (16, 16)
    assert k.round(a).shape_ref == (16, 16)
    assert k.floor(a).shape_ref == (16, 16)
    assert k.ceil(a).shape_ref == (16, 16)
    assert k.trunc(a).shape_ref == (16, 16)


def test_kernel_exposes_isfinite_method():
    """isfinite should produce a node with DTYPE_BOOL after normalize."""
    k = Kernel()
    a = k.load((8, 8), float32)
    f = k.isfinite(a)
    assert f.shape_ref == (8, 8)
    # After codegen (which calls normalize), type_id should be DTYPE_BOOL
    k.store(f)
    k.codegen()
    assert f.type_id == DTYPE_BOOL


def test_reciprocal_is_not_exposed():
    """reciprocal should not be on the public Kernel API."""
    assert not hasattr(Kernel, "reciprocal")


def test_logical_not_is_not_exposed():
    """logical_not should not be on the public Kernel API."""
    assert not hasattr(Kernel, "logical_not")
