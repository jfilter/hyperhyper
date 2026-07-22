"""
Import permissively-licensed human-rated word-similarity sets (ADR 0001, P3).

Run from the repository root::

    python tools/import_eval_data/import_ws.py

This is a **maintainer** tool, not a runtime feature. It is not shipped in the
wheel or the sdist, it is never invoked at install, train or eval time, and the
frozen `.tsv` under `hyperhyper/evaluation_datasets/` is the artifact of record
-- re-running this is an audit/regeneration step, never a reproduction.

Why an importer at all, and why only these sources
--------------------------------------------------

P3 asks for human-rated similarity coverage in more languages. Similarity gold
must come from human ratings; ADR 0001 forbids generating it (the validity
trap). So the only lever is import, and import is gated on **licensing**.

The survey found that gate to be the binding constraint. Multi-SimLex -- the
obvious multilingual candidate -- carries a CC BY-NC-ND notice on its
*Computational Linguistics article*; whether that notice reaches the separately
downloadable data files could not be established (the project site was
unreachable). NC also sits badly inside a BSD-2-Clause wheel. The RG-65/MC-30
translations, the Leviant translations and several SimLex-999 translations state
**no license at all** on their download pages, which is "all rights reserved" as
an operational rule, not "free". Four candidate data repositories were checked
against the GitHub API; every one reported no license.

Rather than build a fetch-and-acknowledge subsystem to route around missing
permission -- which cannot create permission the rightsholder never granted --
this imports only sources whose license is evidenced **on the artifact itself**,
and bundles them outright. Everything else stays a `data_dir` matter for the
user, who is the only party who can assess their own use.

`LICENSE_EVIDENCE` below records *where* each license statement was read. A paper
footer, a repository license covering only code, or a third-party package's
license does not count.

What the conversion does, and does not, do
------------------------------------------

Per source it selects the two word columns and the aggregate human score, and
writes them as the package's strict TSV. It never recomputes, rescales or
otherwise alters a score: the aggregate column is taken as upstream published
it. Rows are dropped only for reasons the scorer forces (see `curate`), and
every drop is counted in the emitted preamble, so the row count is never
silently smaller than the source's headline number.
"""

import csv
import hashlib
import io
import sys
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hyperhyper.preprocessing import tokenize_string_v2  # noqa: E402

DATASETS_ROOT = REPO_ROOT / "hyperhyper" / "evaluation_datasets"


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

SUPERSIM_URL = "https://spraakbanken.gu.se/resurser/data/supersim-superlim.zip"
SUPERSIM_SHA256 = "319e731c2d0801a7bc0abed1ef029f64b919b34614ec6e5f68a787888ea702fd"

WS353DA_URL = (
    "https://raw.githubusercontent.com/fnielsen/dasem/master/"
    "dasem/data/wordsim353-da/combined.csv"
)
# The Danish file is served from a branch tip rather than a tagged release, so it
# is pinned by content hash: a silent upstream edit must fail loudly here rather
# than quietly change a bundled gold set.
WS353DA_SHA256 = "b0c0b0ef0ea800472f78979e09fbbe877bf3cf1b78564c7dde98086c1228a822"

LICENSE_EVIDENCE = {
    "supersim": (
        "CC BY 4.0 -- stated in readme.txt *inside* supersim-superlim.zip: "
        "\"Data for 'SuperSim: a test set for word similarity and relatedness "
        "in Swedish', v1; Simon Hengchen, Nina Tahmasebi; University of "
        'Gothenburg; DOI 10.5281/zenodo.4660084; CC BY 4.0 to Hengchen/Tahmasebi"'
    ),
    "ws353da": (
        "CC-BY -- stated in the dataset's own LICENSE file at "
        "fnielsen/dasem/dasem/data/wordsim353-da/LICENSE, which also records "
        "that the rights to the original English WordSim-353 belong to Evgeniy "
        "Gabrilovich and that the Danish translation is by Finn Aarup Nielsen."
    ),
}


def fetch(url, expected_sha256):
    """
    Download `url` and verify it against `expected_sha256`.

    A hash mismatch aborts. An upstream file that changed under a pinned hash is
    a new version of the dataset and needs the conversion reviewed against it --
    it must never be folded silently into a bundled gold set.
    """
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is None:
        print(f"  (unpinned) sha256 {digest}  <- pin this in the source constants")
    elif digest != expected_sha256:
        raise SystemExit(
            f"checksum mismatch for {url}\n  expected {expected_sha256}\n"
            f"  actual   {digest}\nUpstream changed; review the conversion."
        )
    return payload, digest


# --------------------------------------------------------------------------
# curation -- exactly the invariants tests/test_datasets.py enforces
# --------------------------------------------------------------------------


def curate(rows):
    """
    Drop the rows the unigram scorer cannot represent, and report what went.

    Three reasons, all forced by the evaluator rather than by taste:

    * **multi-token** -- an entry that is not exactly one token under the
      package's current default preprocessing (`tokenize_string_v2`) can never be
      looked up; keeping it would only inflate the out-of-vocabulary rate.
    * **self-pair** -- `w w` has cosine 1 by construction and measures nothing.
    * **duplicate** -- the same unordered pair twice would weight one judgement
      double. Cosine is symmetric, so `a b` and `b a` are the same pair; the
      comparison is case-insensitive and on NFC-normalized text, matching the
      linter.

    Returns `(kept, counts)`; the first occurrence of a duplicated pair is kept.
    """
    counts = {"multitoken": 0, "self_pair": 0, "duplicate": 0}
    seen = set()
    kept = []
    for word1, word2, score in rows:
        if not all(len(tokenize_string_v2(w)) == 1 for w in (word1, word2)):
            counts["multitoken"] += 1
            continue
        norm = [unicodedata.normalize("NFC", w).casefold() for w in (word1, word2)]
        if norm[0] == norm[1]:
            counts["self_pair"] += 1
            continue
        key = frozenset(norm)
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        kept.append((word1, word2, score))
    return kept, counts


def write_tsv(path, preamble, rows):
    """Write the strict TSV the package reads: `# key: value` preamble, header, data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    for line in preamble:
        buffer.write(f"# {line}\n" if line else "#\n")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(["word1", "word2", "score"])
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({len(rows)} rows)")


def provenance(
    *,
    title,
    source_url,
    source_sha256,
    license_id,
    evidence,
    citation,
    transformation,
    counts,
    source_rows,
    kept_rows,
):
    """The `# key: value` preamble every bundled file carries (ADR 0001/0002)."""
    lines = [
        "hyperhyper-eval: 1",
        f"title: {title}",
        f"source-url: {source_url}",
        f"source-sha256: {source_sha256}",
        f"license: {license_id}",
        f"license-evidence: {evidence}",
        f"citation: {citation}",
        "",
        "Imported by tools/import_eval_data/import_ws.py (ADR 0001, phase P3).",
        f"Transformation: {transformation}",
        "No score was recomputed, rescaled or averaged; the aggregate column is",
        "taken exactly as upstream published it.",
        "",
        f"Source rows: {source_rows}; bundled rows: {kept_rows}.",
    ]
    for reason, n in counts.items():
        if n:
            lines.append(f"  dropped-{reason}: {n}")
    return lines


# --------------------------------------------------------------------------
# per-source conversion
# --------------------------------------------------------------------------


def import_supersim():
    """
    Swedish SuperSim -> `sv/ws/supersim_{similarity,relatedness}.tsv`.

    Upstream ships the two judgements as separate files: the same 1360 pairs
    were rated *independently* for similarity and for relatedness by five
    annotators. They stay separate here, because collapsing them would destroy
    the distinction the dataset exists to draw -- and because the aggregation in
    `evaluation.aggregate` de-duplicates shared items across datasets, so the
    overlapping pairs are counted once in the micro average regardless.

    Each row is `Word 1, Word 2, Anno 1..5, Average`; only the two words and
    `Average` are kept. The header cells carry stray spaces around the tabs
    upstream, so fields are stripped.
    """
    print("SuperSim (sv)")
    payload, digest = fetch(SUPERSIM_URL, SUPERSIM_SHA256)
    archive = zipfile.ZipFile(io.BytesIO(payload))
    for aspect in ("similarity", "relatedness"):
        member = f"supersim_superlim/data/gold_{aspect}.tsv"
        text = archive.read(member).decode("utf-8")
        reader = csv.reader(io.StringIO(text), delimiter="\t")
        next(reader)  # header
        rows = []
        for record in reader:
            fields = [cell.strip() for cell in record]
            if len(fields) < 3 or not fields[0]:
                continue
            rows.append((fields[0], fields[1], fields[-1]))
        kept, counts = curate(rows)
        write_tsv(
            DATASETS_ROOT / "sv" / "ws" / f"supersim_{aspect}.tsv",
            provenance(
                title=f"SuperSim ({aspect}), Swedish",
                source_url=SUPERSIM_URL,
                source_sha256=digest,
                license_id="CC-BY-4.0",
                evidence=LICENSE_EVIDENCE["supersim"],
                citation=(
                    "Hengchen, Simon and Nina Tahmasebi (2021). SuperSim: a test "
                    "set for word similarity and relatedness in Swedish. "
                    "Proceedings of NoDaLiDa 2021, Reykjavik."
                ),
                transformation=(
                    f"selected columns 'Word 1', 'Word 2' and 'Average' from "
                    f"{member}; the five per-annotator columns are not carried "
                    f"over (the package scores against the aggregate)"
                ),
                counts=counts,
                source_rows=len(rows),
                kept_rows=len(kept),
            ),
            kept,
        )


def import_ws353da():
    """
    Danish WordSim-353 -> `da/ws/ws353.tsv`.

    **The scores are the original English human ratings**, not Danish ones: this
    is a translation of the word pairs by Finn Aarup Nielsen, distributed with
    the English gold column unchanged. That is a real limitation and is stated in
    the emitted preamble rather than buried here -- a Danish speaker's judgement
    of the translated pair could differ from an English speaker's judgement of
    the original.

    The translator marked 34 rows as problematic in a `Problem` column (pairs
    whose translation is doubtful). Those are dropped: scoring a doubtful
    translation against a score elicited for the English original measures the
    translation, not the embedding.
    """
    print("WordSim-353 (da)")
    payload, digest = fetch(WS353DA_URL, WS353DA_SHA256)
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    rows, flagged = [], 0
    for record in reader:
        if (record.get("Problem") or "").strip():
            flagged += 1
            continue
        rows.append((record["da1"], record["da2"], record["Human (mean)"]))
    kept, counts = curate(rows)
    counts = {"flagged-by-translator": flagged, **counts}
    write_tsv(
        DATASETS_ROOT / "da" / "ws" / "ws353.tsv",
        provenance(
            title="WordSim-353, Danish translation",
            source_url=WS353DA_URL,
            source_sha256=digest,
            license_id="CC-BY",
            evidence=LICENSE_EVIDENCE["ws353da"],
            citation=(
                "Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, "
                "Z., Wolfman, G., Ruppin, E. (2002). Placing Search in Context: "
                "The Concept Revisited. ACM TOIS 20(1):116-131. Danish "
                "translation by Finn Aarup Nielsen."
            ),
            transformation=(
                "selected the Danish columns 'da1'/'da2' and the score column "
                "'Human (mean)'. NOTE: the scores are the ORIGINAL ENGLISH human "
                "ratings, carried over unchanged with the translated pairs; they "
                "were not re-elicited from Danish speakers"
            ),
            counts=counts,
            source_rows=len(rows) + flagged,
            kept_rows=len(kept),
        ),
        kept,
    )


def main():
    import_supersim()
    import_ws353da()


if __name__ == "__main__":
    main()
