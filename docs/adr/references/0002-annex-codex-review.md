# ADR 0002 — Annex: GPT (codex) review, verbatim

> Independent review by GPT via the `codex` CLI (read-only), 2026-07-22, on the
> tokenization + on-disk-format brief. Preserved verbatim as source material for
> [0002-tokenization-and-storage.md](../0002-tokenization-and-storage.md). Not all
> recommendations were adopted — see the ADR's "Where the reviews diverged".

## Bottom line

Keep a small, deterministic, standard-library tokenizer as the default, but replace the current punctuation-destroying behavior. Make tokenization explicitly pluggable and versioned.

For evaluation data, move to strict UTF-8 TSV with headers and a metadata/comment preamble. Read both TSV and legacy `.txt`; write and bundle TSV going forward.

Keep `.npz` for matrices. Keep pickle temporarily for private caches, but treat it explicitly as trusted, disposable cache state—not an interchange format.

## Ranked plan

1. **Fix and version tokenization.** This recovers possessives and hyphenated words and removes surprising digit normalization.
2. **Introduce a strict TSV evaluation format.** This fixes comment leakage, malformed-row silence, and representation of multi-word fields.
3. **Version evaluation results and preprocessing in the bunch manifest/database.** Otherwise changed datasets or tokenization can collide with old recorded results.
4. **Convert bundled datasets and restore legitimately removed rows.**
5. **Leave cache pickle replacement for a later, holistic bunch-format revision.**
6. **Keep matrix `.npz` unchanged.**

---

# A — Tokenization

## Recommendation

A regex-based word tokenizer is still the right lightweight default in 2026. The problem is not regex; it is the current rule “all punctuation is a boundary” plus several unrelated normalization decisions hidden inside it.

The current implementation combines lowercase, per-digit replacement, tag removal, punctuation destruction, and tokenization in one pipeline ([preprocessing.py](/Users/user/code/jf/hyperhyper/hyperhyper/preprocessing.py:84)). That makes seemingly minor changes alter vocabulary identity.

Adopt a named, versioned default—conceptually `simple-v2`—with these semantics:

1. Normalize the input to NFC.
2. Lowercase with `str.lower()`.
3. Optionally remove simple tag-shaped markup, preserving today’s convenience but documenting that this is not an HTML parser.
4. Extract Unicode alphanumeric runs.
5. Preserve an apostrophe, hyphen, or underscore only when it occurs internally between alphanumeric characters.
6. Canonicalize common typographic variants:
   - `’` → `'`
   - non-breaking/Unicode hyphens such as `‑` and `‐` → `-`
7. Keep digits unchanged by default.

Expected behavior:

| Input | New default |
|---|---|
| `City's` | `["city's"]` |
| `city’s` | `["city's"]` |
| `ice-cream` | `["ice-cream"]` |
| `ice‑cream` | `["ice-cream"]` |
| `new_york` | `["new_york"]` |
| `2001` | `["2001"]` |
| `U.S.A.` | `["u", "s", "a"]` |

I would deliberately leave dotted abbreviations unresolved. Treating every internal dot as lexical would also join decimals, versions, hostnames, and sentence punctuation. A special `U.S.A. → usa` rule is possible, but it is the beginning of a language-specific exception list. Users who need that should use the hook.

NFC and Unicode-aware matching need only `unicodedata` and `re`; Python string regexes already use Unicode semantics by default. [Python documents both NFC normalization](https://docs.python.org/3/library/unicodedata.html) and [Unicode-aware regex word classes](https://docs.python.org/3/library/re.html).

## Configurable versus fixed

Keep the configuration surface small:

| Decision | Policy |
|---|---|
| NFC | Fixed in `simple-v2` |
| Unicode-aware matching | Fixed |
| Internal apostrophe/hyphen/underscore preservation | Fixed |
| Lowercasing | Configurable, default `True` |
| Digits | Configurable: `keep` default, `runs-to-0`, and explicit `legacy-each-digit-to-0` |
| Shallow tag stripping | Configurable, default `True` for compatibility |
| Stopwords | Not part of the simple tokenizer |
| Lemmatization | Not part of the simple tokenizer |
| Sentence segmentation | Not part of the tokenizer |
| More punctuation rules | Custom callable, not dozens of flags |

Do not use `casefold()` or NFKC by default. Both are useful for particular languages and search applications, but they perform more identity-changing transformations than this inspectable baseline should hide.

The present `2001 → 0000` behavior is a historical modeling choice, not neutral preprocessing. For small domain corpora, `IL-6`, years, doses, model numbers, statutory sections, and product identifiers can carry important distinctions. Make digit normalization explicit.

## Pluggability and API boundary

Yes, tokenization should be pluggable. The current package technically allows a `preproc_func`, but that hook is too broad: it accepts a batch of texts and may change cardinality. Evaluation then applies that same function to dataset columns and uses `strict=False` because spaCy sentence splitting may return a different number of outputs ([evaluation.py](/Users/user/code/jf/hyperhyper/hyperhyper/evaluation.py:212)). That is an unsafe abstraction.

Use two distinct concepts:

- `tokenizer: str -> sequence[str]`: scalar and cardinality-preserving at the input level.
- Optional document preprocessing/segmentation for users who want spaCy or another NLP pipeline.

Evaluation should apply the corpus’s scalar tokenizer to each complete TSV field. A field is answerable only when it resolves to exactly one vocabulary token.

Multi-word TSV entries should be parsed and retained, but the default evaluator should report them as uncovered/OOV. Do not silently average their component vectors. A user with phrase tokens can supply the same custom tokenizer or term resolver used to produce, for example, `new_york`.

## spaCy’s place

Make the lightweight tokenizer the default for **all** constructors. Currently `from_texts` and `from_text_files` default to spaCy ([corpus.py](/Users/user/code/jf/hyperhyper/hyperhyper/corpus.py:249)); that conflicts with the stated package identity.

The default document behavior should be explicit and simple:

- `from_sents`: each supplied string is one sequence.
- `from_texts`: each supplied document is one sequence, or split only on explicit newlines.
- `from_text_files`: preserve line boundaries already present in the files.

Keep `texts_to_sents` as an optional adapter. It should require an installed model and fail with installation instructions; it should not auto-download a model during preprocessing. Auto-downloads make offline and reproducible runs harder.

Removing `gensim.parsing` from the tokenizer is worthwhile for inspectability and semantic ownership, although it does not currently eliminate gensim as a package dependency because `Vocab`, `KeyedVectors`, and persistence still use it elsewhere.

## Tokenizer migration

Because the project is pre-1.0, change the default now rather than preserving a poor default indefinitely, but make the break controlled:

1. Name the new behavior `simple-v2`.
2. Retain an explicit `legacy-v1` tokenizer/preset.
3. Store `tokenizer_id`, tokenizer options, and package version in new corpora/bunches.
4. On loading an old corpus without metadata:
   - Recognize old built-in preprocessing callables and bind them to frozen v1 behavior.
   - Preserve unknown custom callables but label them `custom-unknown` and warn that their semantics cannot be reconstructed.
5. Do not let the existing function reference silently acquire v2 semantics. Old chunks and vocabularies were built with v1, while evaluation currently retrieves the stored function through `corpus.preproc_fun` ([bunch.py](/Users/user/code/jf/hyperhyper/hyperhyper/bunch.py:490)).
6. Tell users that retraining under v2 requires a new bunch. Do not mutate an existing bunch’s vocabulary in place.

---

# B — On-disk data

## 1. Evaluation datasets: move to TSV

TSV is the best fit—not comma CSV and not JSONL.

It remains glanceable and diffable like the traditional format, while a tab is unlikely to occur in a lexical item. Python’s standard `csv` module already supports alternate delimiters, quoting, and strict parsing, so this adds no dependency. [The standard-library CSV reader supports quoting and strict error handling](https://docs.python.org/3/library/csv.html).

### Exact format

UTF-8, LF line endings, `.tsv` extension.

Similarity:

```text
# hyperhyper-eval: 1
# language: en
# source: https://example.org/dataset
# citation: ...
# license: ...
word1	word2	score
city's	town	7.25
ice cream	frozen dessert	8.1
```

Analogy:

```text
# hyperhyper-eval: 1
# language: en
# source: ...
# license: ...
a	a_prime	b	b_prime
athens	greece	baghdad	iraq
```

Rules:

- Metadata/comments are allowed only in the preamble, before the header.
- Preamble syntax is `# key: value`; blank preamble lines are allowed.
- Required exact headers:
  - `word1`, `word2`, `score`
  - `a`, `a_prime`, `b`, `b_prime`
- Delimiter is one tab.
- Standard double-quote CSV escaping, `QUOTE_MINIMAL`.
- Spaces inside fields need no quoting.
- Empty required fields are invalid.
- Similarity scores must be finite numeric values.
- After the header, every nonblank record must have exactly the required columns.
- Malformed input raises an error containing filename and line number. Never silently discard it.
- No embedded line breaks in fields; they offer no useful evaluation semantics and complicate inspection.

### Why not the alternatives?

- **Comma CSV:** equally capable technically, but less pleasant for lexical material containing commas and less visually aligned.
- **JSONL:** explicit, but much noisier for four fixed scalar fields. A special metadata object also creates a second record schema. Use JSONL only if rows later acquire nested or heterogeneous structure.
- **Space-delimited text:** keep for import compatibility, not as the canonical format.

### Backward compatibility

- Read both `.tsv` and legacy `.txt`; select the parser by extension, never delimiter sniffing.
- Convert all bundled files to `.tsv` and stop bundling duplicate `.txt` copies.
- If a directory contains `foo.tsv` and `foo.txt`, raise a duplicate-dataset error rather than scoring both.
- For legacy `.txt`:
  - Skip blank lines and lines whose first non-whitespace character is `#`.
  - Preserve exact 3/4-field parsing.
  - Warn with filename and line number for malformed non-comment rows instead of silently losing them.
- Any dataset-writing or generation tooling writes TSV only.
- Restore removed possessive/hyphenated rows from authoritative upstream sources after v2 tokenization lands. Do not reconstruct them from comments or examples.
- Multi-word rows can now be represented faithfully even if a particular embedding cannot answer them.

Also add an `evaluation_id` to recorded experiment parameters: a digest over parser/schema version, canonical parsed rows, and tokenizer identity. Currently result deduplication keys contain model parameters but no dataset or evaluation version ([experiment.py](/Users/user/code/jf/hyperhyper/hyperhyper/experiment.py:85)); otherwise new scores may collide with legacy rows.

## 2. Training chunks: keep pickle for now

Do not replace `texts_N.pkl` with JSONL as an isolated change.

These files are internal chunks and normally contain ragged sequences of integer vocabulary IDs, not readable original tokens ([corpus.py](/Users/user/code/jf/hyperhyper/hyperhyper/corpus.py:154)). JSONL would be larger and slower while still being semantically opaque without the vocabulary mapping. More importantly, `corpus.pkl` would continue to make the bunch pickle-dependent.

The security caveat is real: unpickling malicious data can execute code, as the [Python documentation warns](https://docs.python.org/3.10/library/pickle.html). Therefore:

- Document that bunch directories are trusted local artifacts.
- Never load a downloaded or otherwise untrusted bunch.
- Treat chunks as disposable caches that can be rebuilt.
- Add atomic writes and a small `manifest.json` with cache/preprocessing version, corpus fingerprint, vocabulary size, chunk list, and package versions.
- Stop creating temporary `.pkl` files beside source corpus files; use a controlled temporary/bunch location.

If portable, shareable bunches become a goal, replace pickle holistically:

- `manifest.json` for corpus metadata.
- `vocab.tsv` for token ↔ ID mapping.
- Each text chunk as numeric `ids` plus `offsets` arrays in `.npz`/`.npy`.
- Dual-read old pickle bunches; write only the new format.

That is preferable to JSONL and should replace both `corpus.pkl` and chunk pickles together.

## 3. Matrices: retain `.npz`

The current choice is correct:

- SciPy compressed sparse `.npz` for co-occurrence and PMI matrices.
- NumPy compressed `.npz` for dense SVD arrays.

SciPy directly supports compressed sparse persistence with `save_npz`/`load_npz`, including the sparse format and shape. [That is the intended SciPy representation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html). CSV would destroy sparsity and create enormous, slow files.

Small improvements only:

- Record matrix dtype, shape, vocabulary fingerprint, parameters, and cache schema in the bunch manifest.
- Write atomically.
- Make `allow_pickle=False` explicit for NumPy loads.
- Keep the existing legacy sparse-layout reader until old bunch support is intentionally retired.

## What not to do

- Do not make spaCy, Hugging Face tokenizers, or model downloads the default.
- Do not add a large menu of punctuation flags; use the callable hook for specialized behavior.
- Do not preserve digit normalization merely because word2vec-era pipelines did it.
- Do not apply NFKC, casefolding, stemming, stopword removal, or lemmatization invisibly.
- Do not auto-compose multi-word evaluation entries from component vectors.
- Do not sniff evaluation delimiters or silently discard malformed rows.
- Do not adopt JSONL solely because it is more structured.
- Do not convert sparse matrices to CSV.
- Do not replace only `texts_N.pkl` while leaving the rest of the bunch pickle-based.