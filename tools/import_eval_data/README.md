# `tools/import_eval_data` — importing human-rated similarity sets

Maintainer tooling that converts **externally published, permissively licensed**
word-similarity datasets into the strict TSV the package evaluates on. This
implements **ADR 0001, phase P3** (`docs/adr/0001-evaluation-data-strategy.md`;
see the P3 addendum for the full reasoning and what was rejected).

It is **out of the runtime**: not in the wheel, not in the sdist, never invoked
at install, train or eval time. The frozen `.tsv` under
`hyperhyper/evaluation_datasets/` is the artifact of record; re-running the
importer is an audit/regeneration step, never a reproduction.

```
uv run python tools/import_eval_data/import_ws.py
```

## Why this is an import tool and not a downloader

Similarity gold must come from **human ratings** — ADR 0001 forbids generating
it, because an LLM assigning similarity scores turns "agreement with humans"
into "agreement with a language model". So coverage grows only by importing, and
importing is gated on licensing.

That gate turned out to be the binding constraint. Most published similarity
datasets state **no license at all**, which is "all rights reserved" as an
operational rule rather than "free"; the largest multilingual candidate's
license status could not be established at the artifact level. An earlier design
proposed a fetcher with a license-acknowledgement prompt. It was rejected: a
prompt cannot create permission the rightsholder never granted, and shipping one
would make the project appear to bless an ambiguous use.

**The rule this directory enforces:** a dataset is bundled only when its
permissive license is stated **on the artifact itself** — inside the distributed
archive, or in a `LICENSE` file shipped with the data. A paper footer, a
repository license that covers only code, and a third-party package's license
all fail this test. Each imported file records where its license was read, in the
`license-evidence` field of its own preamble.

Everything else is the user's call via `data_dir`, since only the user can
assess their own use of a restricted source.

## What the conversion is allowed to do

Per source: select the two word columns and the **published aggregate** score,
and write them as TSV. It never recomputes, rescales or re-averages a score.

Rows are dropped only where the unigram scorer forces it — an entry that is not
exactly one token under `tokenize_string_v2`, a self-pair, or a repeated
unordered pair — plus any row the upstream author flagged as doubtful. **Every
drop is counted in the emitted preamble**, so a bundled file never quietly
reports a smaller row count than its source's headline number.

Output is gated by `tests/test_datasets.py`, the same linter every bundled
dataset passes.

## Adding a source

1. Establish the license **at the artifact level** and record the exact location
   in `LICENSE_EVIDENCE`. If you cannot, stop — do not bundle it.
2. Pin the source by SHA-256. A hash mismatch aborts the run: an upstream file
   that changed is a new version of the dataset, and its conversion needs review
   before it can reach a bundled gold set.
3. Write a conversion function that touches columns only, and emit the standard
   provenance preamble via `provenance(...)`.
4. Create `hyperhyper/evaluation_datasets/<lang>/ws/` with `__init__.py` markers.
   The linter discovers languages from the filesystem, so the new files are
   linted automatically — no list to update.
5. Run the linter and note the row count in the CHANGELOG.
