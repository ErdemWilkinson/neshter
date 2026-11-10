[Türkçe README](README.tr.md)

# Neşter

Neşter loads language models that live on your own computer, shows which attention heads and
behavior directions do what, lets you measure and silence selected parts, and saves the result as a
new model file.

It is **100% local**: there is no cloud API in it, no account, and **no telemetry**.

## What it supports

- **Model families:** Llama, Qwen2 and Gemma2 (up to 8B parameters). Other architectures are rejected.
- **Cut types in v0.1:**
  - **Head ablation:** silence a query attention head (its `o_proj` columns are zeroed).
  - **Direction orthogonalization:** remove a behavior direction from the weights.
- On Gemma2, only head ablation can be saved; direction removal can be explored but not saved in v0.1.
- Both keep tensor shapes, so the result is GGUF compatible. Structural pruning is not in v0.1.
- The real effect of a cut is measured as KL divergence against a random-head control, with
  Wilson 95% intervals; an interval that includes zero is reported as "uncertain".

## Install and run

Neşter is not published on PyPI yet, so run it from a clone of this repository:

```
git clone <repo-url> neshter
cd neshter
uvx --from . neshter
```

or, without cloning, `uvx --from git+<repo-url> neshter`. Do not run a bare `uvx neshter`: until
the package is published, that name on PyPI is not ours. (Once it is published, `uvx neshter` will
work too.)

On Windows you can double-click or run `neshter.bat` in the clone; it installs
[uv](https://docs.astral.sh/uv/) if it is missing and then runs this checkout
(`uvx --from <clone folder> neshter`). Neşter starts a local server and opens your browser.
Options: `--no-browser`, `--port N`.

Apply a shared recipe (`.nester.json`) without the UI:

```
neshter apply recipe.nester.json path/to/model --out path/to/new_model --format safetensors
```

`--format` is `safetensors` (default), `gguf_q4_k_m` or `gguf_q8_0`. Exit codes: `0` ok, `1` error,
`2` the recipe is invalid or does not fit the model, `3` the recipe was made for a different model
(SHA-256 mismatch; add `--force` to apply anyway), `4` llama.cpp is required but not found.

## GGUF and Ollama

GGUF export needs [llama.cpp](https://github.com/ggml-org/llama.cpp) on your machine
(`convert_hf_to_gguf.py` and `llama-quantize`). Point the `NESHTER_LLAMA_CPP` environment variable at
its folder, or put `llama-quantize` on your `PATH`. Without it, save as safetensors; the GGUF option
reports "llama.cpp required".

Importing the saved GGUF into Ollama is optional and **off by default**. When enabled, Neşter runs
`ollama create` for you.

## Security model

- The server binds to `127.0.0.1` only. There is no option to listen on other addresses.
- A random session token is generated at start and required on every API request
  (`X-Neshter-Token`). It travels in the URL fragment, which browsers never send to a server.
- `Origin` must be `http://127.0.0.1:<port>` or `http://localhost:<port>` (or absent), and `Host` must
  be a local address, which blocks cross-site and DNS-rebinding requests.
- `.pt` files are loaded only with `torch.load(weights_only=True)`; otherwise they are rejected.
- `trust_remote_code` is never enabled.
- Your source model is never modified: it is opened read-only and its SHA-256 is checked before and
  after saving. Saves are atomic (`.tmp` folder, then `os.replace`) and never overwrite an existing
  folder.
- Recipes are untrusted input: strict schema, no code, no file paths.

## Responsible use

Neşter has a built-in `refusal` concept. Models saved after cutting it carry the metadata
`abliterated: refusal` and a `safety-ablated` tag in the generated model card, because their safety
behavior is reduced. Use such models responsibly, and respect the license of the base model
(for example, the Llama license has naming and use-policy requirements for derivatives).

## Status

v0.1 backend: model loading, scanning, head activity maps, concept hunting, comparison, recipes,
export, local API and CLI. The web UI is coming (Plan 2).

## License

License: TBD
