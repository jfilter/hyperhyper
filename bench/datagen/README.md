# `bench/datagen` — offline analogy-dataset builder

Tooling that turns **verified base pairs** into the frozen four-column analogy
`.txt` files the package evaluates on. This implements **ADR 0001, phase P2**
(`docs/adr/0001-evaluation-data-strategy.md`).

It is deliberately **out of the runtime**: nothing here is imported by the
installed package, and the only third-party-ish import is
`hyperhyper.preprocessing.tokenize_string` (so "single token" means exactly what
it means at eval time). There is **no network call, no LLM, no
torch/transformers** in this directory. Fact verification is done out of band by
a human using web sources and recorded in the base-pairs file and the emitted
header. The frozen `.txt` is the artifact of record; re-running the builder is an
audit/regeneration step, **never** something the package does at install, train,
or eval time. A regenerated file is a *new version*, not a reproduction (ADR
decision 5).

## Layout

```
bench/datagen/
  README.md                 this file
  pairs/
    fr_capitals.tsv         canonical, verified base pairs: capital<TAB>country<TAB>source
  compile_analogies.py      base pairs  ->  frozen 4-column analogy .txt
```

The shipped proof produced by this pipeline:
`hyperhyper/evaluation_datasets/fr/analogy/capitals.txt`.

## The flow (what the ADR mandates, and how it maps to code)

1. **Schema — one-to-one, verifiable relations only.** For the French proof the
   relation is **capital→country** (`paris france`), the lowest-risk,
   web-checkable relation. Currency (eurozone → many-to-one) and morphological
   relations (surface-form, incompatible with the lemmatizing preprocessor) are
   avoided.
2. **Generate base *pairs*, not quadruples.** You verify N pairs, not N².
   Candidate pairs are proposed by an LLM and written to `pairs/<lang>_<rel>.tsv`.
3. **Verify every pair against a non-LLM authority** (Wikidata / an encyclopaedic
   source). The source is cited per file (header of the `.tsv`) and again in the
   emitted dataset header. Drop anything you cannot verify — never keep an
   unverified fact. Target 100 %.
4. **Filter for usability.** Every term must reduce to exactly **one token** under
   `hyperhyper.preprocessing.tokenize_string`. Multi-word / hyphenated forms split
   (`états-unis → ['états','unis']`, `pays-bas`, `royaume-uni`, `le caire`, …) and
   are therefore excluded. `compile_analogies.py` re-checks this and aborts loudly
   if any term slips through — it is a hard gate, not a warning.
5. **Compile quadruples deterministically, with a per-base-pair cap.** From N
   base pairs, each pair is the *source* of exactly `K = min(cap, N-1)` questions
   `a a_ b b_`, with partners drawn from a **fixed seed**. The file grows as
   `N*K`, not the `N*(N-1)` full Cartesian product the ADR forbids. For French:
   N = 63, K = 5, seed = 0 → **315 quadruples**.
6. **Freeze.** The result is written to
   `hyperhyper/evaluation_datasets/<lang>/analogy/<name>.txt` in the exact
   whitespace 4-column format, preceded by a `#`-comment provenance header.

### The header must not leak as a data row

`setup_test_tokens` has **no notion of comments** — it keeps every line that
splits into exactly the field count (4 for analogy). A header line like
`# capital country pairs` would split into 4 tokens and leak as a bogus data row.
`build_header()` asserts that **no** header line splits into exactly four
whitespace tokens, and the builder's self-check re-parses the frozen file through
`setup_test_tokens` and confirms it recovers exactly the intended quadruples with
zero leakage. The header records: generator model, build date, relation type,
verification sources, verified pair count, and that the data is LLM-proposed +
independently fact-verified.

## Reproduce the shipped French set

```bash
# from bench/datagen/ , with the package importable (e.g. an editable install)
python compile_analogies.py
```

That rebuilds `fr/analogy/capitals.txt` with the frozen defaults
(`--pairs pairs/fr_capitals.tsv --lang fr --relation capital-country --cap 5
--seed 0`) and runs the round-trip self-check. Because the seed and cap are
fixed, the output is byte-stable across runs.

## Acceptance gate

The dataset linter `tests/test_datasets.py` auto-discovers `fr` and enforces, on
`fr/analogy/capitals.txt`: 4 fields per row, single-token entries, four distinct
tokens, no answer-collapse (`b_ ∉ {a, a_, b}`), no duplicate rows, and no comment
leakage. Run it:

```bash
python -m pytest tests/test_datasets.py -k fr
```

## How to add a language

1. Create `pairs/<lang>_capitals.tsv` (same three columns; French-style header
   citing your verification source).
2. Propose base pairs with an LLM, then **verify each against a non-LLM
   authority** and drop unverifiable ones.
3. Keep only pairs whose capital *and* country are single tokens under
   `tokenize_string` (the builder enforces this; check first to avoid surprises).
4. Run:
   ```bash
   python compile_analogies.py \
     --pairs pairs/<lang>_capitals.tsv \
     --out ../../hyperhyper/evaluation_datasets/<lang>/analogy/capitals.txt \
     --lang <lang> --relation capital-country --cap 5 --seed 0
   ```
5. Add the language to the `_discover()` tuple in `tests/test_datasets.py` (an
   analogy-only pack is fine — the linter skips a missing `ws/` directory) and run
   the linter.

## How to add a relation

The builder is written for the `capital→country` shape (two columns → four-token
question). A new **one-to-one, non-morphological** relation with the same
"two single tokens per pair" shape (e.g. country→continent, verifiable against an
encyclopaedic source) reuses `compile_analogies.py` unchanged: point `--pairs` at
its `.tsv` and pass a descriptive `--relation`. Relations that are many-to-one
(currency in the eurozone) or surface-form/morphological (plurals, conjugations —
they collapse under the lemmatizing preprocessor) are **out of scope** for this
builder by ADR decision; a morphology set would need its own surface-form profile
and must be labelled as such.
