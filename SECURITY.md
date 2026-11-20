# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| older | no |

## Reporting a vulnerability

Please report vulnerabilities **privately** through GitHub's "Report a vulnerability" button
(private vulnerability reporting) on this repository's Security tab. Do not open a public issue for
a security problem.

Include the version, your operating system, what you did and what you expected. Neshter is a small
project run by one person, so replies may take a few days.

## Scope

In scope:

- The local server guard: the `127.0.0.1` binding, the session token, the `Origin` and `Host` checks
  and the security headers.
- Loading models and recipes: file parsing, `.pt` handling, recipe validation, anything that could
  read or write outside the folders you chose.
- Export: saving safetensors/GGUF, the source model staying untouched, atomic saves.

Out of scope:

- The llama.cpp folder you point `NESHTER_LLAMA_CPP` at (its converter runs with your rights; use one
  you trust).
- A model you chose to load. Neshter never enables `trust_remote_code`, but a model's own content and
  license are your responsibility.
- Reports that need an attacker who already controls your account or machine.
