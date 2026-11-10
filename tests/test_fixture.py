from transformers import AutoModelForCausalLM, AutoTokenizer


def test_tiny_model_loads(tiny_model_dir):
    m = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    t = AutoTokenizer.from_pretrained(tiny_model_dir)
    assert m.config.num_attention_heads == 4
    assert t.pad_token == "<pad>"
    assert len(t("merhaba dünya")["input_ids"]) > 0
