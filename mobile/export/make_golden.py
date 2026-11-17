"""Telefondaki Kotlin tokenizer'ının birebir aynı token'ları vermesi için Hugging Face'ten altın referans üretir.

  python mobile/export/make_golden.py MODEL_KLASORU CIKIS.json
Her metin için: token id'leri, her token'ın ekranda görünen hali (Neşter'in gösterdiği gibi) ve geri çözülmüş metin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TEXTS = [
    "İstanbul Türkiye'nin en kalabalık şehri ve",
    "Hello, world! It's 2026 and I'LL say: we've done it.",
    "   leading spaces\n\n\ttabs and\r\nwindows lines  ",
    "emoji 🎉👨‍👩‍👧 test 🙂",
    "数字123456 and 中文 mixed",
    "ŞĞÜÇÖİı şğüçöiı",
    "a b c　d",  # bölünmez boşluk, em boşluğu, ideografik boşluk
    "é café",  # NFC: é
    "def f(x):\n    return x**2  # kare\n",
    "<|im_start|>user\nmerhaba<|im_end|>\n<|im_start|>assistant\n",
    "x  \n  y\n\n\n z",
    "Kapadokya'daki peri bacaları, volkanik tüf katmanlarının aşınmasıyla oluşmuştur.",
    "",
    "''''  ...!!!???",
    "Ж ж Ω ω א ب ह",
]


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(sys.argv[1])
    out = []
    for text in TEXTS:
        ids = tok(text, add_special_tokens=True)["input_ids"]
        pieces = [tok.convert_tokens_to_string([p]) for p in tok.convert_ids_to_tokens(ids)]
        out.append({"text": text, "ids": ids, "pieces": pieces, "decoded": tok.decode(ids, skip_special_tokens=True)})
    Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[2]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out)} metin -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
