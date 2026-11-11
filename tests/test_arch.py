from transformers import AutoModelForCausalLM

from neshter.arch import (
    arch_info,
    decoder_layers,
    down_proj,
    embed,
    estimate_memory_bytes,
    fit_badge,
    o_proj,
)


def test_arch_info_tiny(tiny_model_dir):
    m = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    a = arch_info(m)
    assert (a.family, a.n_layers, a.n_heads, a.n_kv_heads, a.head_dim, a.d_model, a.d_mlp) == (
        "qwen2",
        2,
        4,
        2,
        16,
        64,
        128,
    )
    assert a.n_params == sum(p.numel() for p in m.parameters())


def test_module_accessors(tiny_model_dir):
    m = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    assert len(decoder_layers(m)) == 2
    assert o_proj(m, 1) is m.model.layers[1].self_attn.o_proj
    assert o_proj(m, 0).in_features == 4 * 16
    assert down_proj(m, 0) is m.model.layers[0].mlp.down_proj
    assert embed(m) is m.model.embed_tokens


def test_estimate_memory_bytes():
    assert estimate_memory_bytes(1000) == int(1000 * 2 * 1.15)
    assert estimate_memory_bytes(1000, dtype_bytes=4) == int(1000 * 4 * 1.15)


def test_fit_badge():
    G = 1024**3
    assert fit_badge(5 * G, 8 * G, 16 * G) == "ok"
    assert fit_badge(15 * G, 8 * G, 16 * G) == "slow"
    assert fit_badge(40 * G, 8 * G, 16 * G) == "no"
    # sınır değerleri
    assert fit_badge(int(0.9 * 10 * G), 10 * G, 0) == "ok"
    assert fit_badge(1, 0, 0) == "no"


def test_fit_badge_without_gpu_is_ok_when_it_fits_in_ram():
    # GPU yok (vram 0): "RAM desteği" bir yedek yol değil, tek yol; sığıyorsa rozet "ok" olmalı.
    G = 1024**3
    assert fit_badge(int(0.6 * G), 0, 31 * G) == "ok"
    assert fit_badge(20 * G, 0, 31 * G) == "ok"
    assert fit_badge(26 * G, 0, 31 * G) == "no"
    # GPU var ama model sığmıyor: yine "slow"
    assert fit_badge(15 * G, 8 * G, 16 * G) == "slow"
