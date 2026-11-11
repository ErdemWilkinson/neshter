import pytest
import torch
from tiny_models import VOCAB_SIZE, build_tiny_qwen2, build_tiny_tokenizer
from transformers import Gemma2Config, Gemma2ForCausalLM

torch.set_num_threads(2)


@pytest.fixture(scope="session", autouse=True)
def _isolated_neshter_home(tmp_path_factory):
    """Hiçbir test gerçek ~/Neshter/.state'e yazmasın (F2). Ayrı bir ev gereken test monkeypatch.setenv kullanır."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NESHTER_HOME", str(tmp_path_factory.mktemp("neshter_home")))
        yield


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    return build_tiny_qwen2(tmp_path_factory.mktemp("tiny_qwen2"))


@pytest.fixture(scope="session")
def lm(tiny_model_dir):
    from neshter.loader import load_model

    return load_model(tiny_model_dir, device="cpu", dtype=torch.float32)


def _build_gemma2(path, head_dim: int):
    """Minik rastgele Gemma2 (bağlı gömme). Post-norm ağırlıkları rastgele: init sıfır (γ = 1) testi
    önemsizleştirirdi; katman 0'da bir γ girdisi tam 0 (weight = -1) yapılır."""
    tokenizer = build_tiny_tokenizer()
    torch.manual_seed(0)
    config = Gemma2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        intermediate_size=128,
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = Gemma2ForCausalLM(config).to(torch.float32)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.post_attention_layernorm.weight.normal_(0, 0.5)
            layer.post_feedforward_layernorm.weight.normal_(0, 0.5)
        model.model.layers[0].post_attention_layernorm.weight[0] = -1.0
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_gemma2_dir(tmp_path_factory):
    """Minik rastgele Gemma2: head_dim=16, n_heads·head_dim = hidden (kare o_proj)."""
    return _build_gemma2(tmp_path_factory.mktemp("tiny_gemma2"), head_dim=16)


@pytest.fixture(scope="session")
def gemma_lm(tiny_gemma2_dir):
    from neshter.loader import load_model

    return load_model(tiny_gemma2_dir, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="session")
def gemma_wide_lm(tmp_path_factory):
    """Gerçek Gemma2 gibi n_heads·head_dim ≠ hidden (4·24 = 96 ≠ 64; o_proj [64, 96], kare değil)."""
    from neshter.loader import load_model

    path = _build_gemma2(tmp_path_factory.mktemp("tiny_gemma2_wide"), head_dim=24)
    return load_model(path, device="cpu", dtype=torch.float32)


def logits(lm, text):
    """Tek metin için ham logit'ler (müdahale testlerinde paylaşılan yardımcı)."""
    with torch.no_grad():
        return lm.model(**lm.tokenizer(text, return_tensors="pt")).logits


def recipe_for(lm, interventions, concept="code"):
    """Müdahale listesinden kesim tarifi (Task 10-12 testlerinde paylaşılan yardımcı)."""
    from neshter.recipe import from_interventions

    return from_interventions(interventions, lm, concept=concept, method="manual", seed=0)
