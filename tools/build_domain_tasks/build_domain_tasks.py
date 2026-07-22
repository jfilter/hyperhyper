"""
Build domain proxy evaluation tasks from a glossary or taxonomy (ADR 0001, P4).

    uv run python tools/build_domain_tasks/build_domain_tasks.py synonym \
        --glossary my_glossary.tsv --out eval_data --lang en --name glossary

    uv run python tools/build_domain_tasks/build_domain_tasks.py category \
        --taxonomy my_taxonomy.tsv --out eval_data --lang en --name taxonomy

The output directory is laid out as ``<out>/<lang>/<kind>/<name>.tsv``, which is
exactly what `read_test_data(..., data_dir=...)` expects, so::

    bunch.eval_synonym(embedding, data_dir="eval_data")
    bunch.eval_category(embedding, data_dir="eval_data")

Why this exists
---------------

The bundled similarity and analogy sets measure *general language*. On the
small, domain-specific corpora this package is for, they are largely
out-of-vocabulary and therefore measure close to nothing -- ADR 0001 records
that tension as the central one. The honest fix is not to invent domain
similarity ratings (that needs human judgement, which ADR 0001 forbids
generating) but to build tasks whose gold answer is a **membership fact** the
domain already records:

* **synonym multiple choice** -- gold is a thesaurus/glossary entry;
* **category purity** -- gold is a taxonomy's class assignment.

Verifying either is a lookup, not an opinion. That is the whole point, and it is
why this builder reads *your* glossary rather than generating one.

Nothing here is shipped in the wheel, and no dataset of these kinds is bundled:
a general-language synonym set would recreate the very problem P4 exists to
solve.
"""

import argparse
import csv
import hashlib
import io
import random
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hyperhyper.preprocessing import tokenize_string_v2  # noqa: E402

DEFAULT_DISTRACTORS = 3


def read_pairs(path):
    """
    Read a two-column TSV, skipping `#` comments and an optional header.

    The glossary format is deliberately the plainest thing that can express the
    input: one ``a<TAB>b`` per line. For a glossary that is ``term<TAB>synonym``
    (repeat the term for several synonyms); for a taxonomy it is
    ``word<TAB>category``.
    """
    rows = []
    for lineno, line in enumerate(
        Path(path).read_text(encoding="utf-8").split("\n"), 1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = next(csv.reader([line], delimiter="\t"))
        if len(fields) != 2 or not all(f.strip() for f in fields):
            raise SystemExit(
                f"{path}:{lineno}: expected 2 non-empty tab-separated columns, "
                f"got {fields!r}"
            )
        rows.append((fields[0].strip(), fields[1].strip()))
    if not rows:
        raise SystemExit(f"{path}: no data rows")
    return rows


def single_token(word):
    """Whether `word` survives the package's default preprocessing as one token."""
    return len(tokenize_string_v2(word)) == 1


def normalized(word):
    """The comparison key used for identity throughout: NFC, case-folded."""
    return unicodedata.normalize("NFC", word).casefold()


def build_synonym(rows, n_distractors, seed):
    """
    Turn ``term<TAB>synonym`` pairs into multiple-choice questions.

    One question per usable pair: the term is the target, the listed synonym is
    the answer, and `n_distractors` distractors are drawn from the other terms in
    the glossary.

    The distractor draw is where a multiple-choice set is usually spoiled, so
    three rules are enforced rather than assumed:

    * **A distractor is never a known synonym of the target** -- not just not
      *this* answer, but none of the target's synonyms anywhere in the glossary,
      and not the target itself. Otherwise a "wrong" answer is silently right and
      the score is capped below what the embedding deserves.
    * **Distractors come from the glossary's own vocabulary**, so they are
      in-domain. Sampling general-language words would make the task easy for the
      wrong reason: any domain-trained embedding separates domain terms from
      unrelated ones without knowing a thing about synonymy.
    * **The draw is seeded and order-independent.** Candidates are sorted before
      sampling, so the output depends on the glossary content and the seed, not
      on dictionary iteration order. Re-running gives a bit-identical file.

    A term with fewer than `n_distractors` eligible distractors is skipped rather
    than padded with a weaker distractor.
    """
    rng = random.Random(seed)

    synonyms = defaultdict(set)
    for term, synonym in rows:
        synonyms[normalized(term)].add(normalized(synonym))
        # synonymy is symmetric: if the glossary lists a -> b, then b must not be
        # offered as a distractor for a question whose target is b's synonym a
        synonyms[normalized(synonym)].add(normalized(term))

    surface = {}
    for term, synonym in rows:
        surface.setdefault(normalized(term), term)
        surface.setdefault(normalized(synonym), synonym)

    pool = sorted(key for key in surface if single_token(surface[key]))

    questions = []
    skipped = {"multitoken": 0, "too_few_distractors": 0, "self_synonym": 0}
    for term, synonym in rows:
        target_key, answer_key = normalized(term), normalized(synonym)
        if target_key == answer_key:
            skipped["self_synonym"] += 1
            continue
        if not (single_token(term) and single_token(synonym)):
            skipped["multitoken"] += 1
            continue
        excluded = synonyms[target_key] | {target_key}
        eligible = [key for key in pool if key not in excluded]
        if len(eligible) < n_distractors:
            skipped["too_few_distractors"] += 1
            continue
        distractors = rng.sample(eligible, n_distractors)
        questions.append([term, synonym, *(surface[key] for key in distractors)])

    # a question is identified by its target and its candidate set; the same pair
    # listed twice in the glossary must not become two questions
    seen, unique = set(), []
    duplicates = 0
    for question in questions:
        key = (normalized(question[0]), frozenset(normalized(c) for c in question[1:]))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(question)
    skipped["duplicate"] = duplicates
    return unique, skipped


def build_category(rows):
    """
    Turn ``word<TAB>category`` pairs into a category-purity dataset.

    Almost a pass-through, but three things are checked because the evaluator
    cannot recover from them: multi-token words are dropped (unscoreable), a word
    assigned to two different categories is dropped (its gold is contradictory,
    and keeping the first assignment would silently pick a winner), and a
    repeated identical assignment is collapsed.
    """
    dropped = {"multitoken": 0, "conflicting_category": 0, "duplicate": 0}

    assignments = defaultdict(set)
    surface = {}
    for word, category in rows:
        assignments[normalized(word)].add(category)
        surface.setdefault(normalized(word), word)

    kept = []
    emitted = set()
    for word, category in rows:
        key = normalized(word)
        if not single_token(word):
            dropped["multitoken"] += 1
            continue
        if len(assignments[key]) > 1:
            dropped["conflicting_category"] += 1
            continue
        if key in emitted:
            dropped["duplicate"] += 1
            continue
        emitted.add(key)
        kept.append([word, category])
    return kept, dropped


def write_tsv(path, header, preamble, rows):
    """Write the strict TSV the package reads: `# key: value` preamble, header, data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    for line in preamble:
        buffer.write(f"# {line}\n" if line else "#\n")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"wrote {path}  ({len(rows)} rows)")


def preamble_lines(*, kind, source, source_sha256, extra, counts, source_rows, kept):
    lines = [
        "hyperhyper-eval: 1",
        f"kind: {kind}",
        f"source: {source}",
        f"source-sha256: {source_sha256}",
        "",
        "Built by tools/build_domain_tasks/build_domain_tasks.py (ADR 0001, P4).",
        "Gold is a membership fact taken from the source, never a generated or",
        "model-assigned judgement.",
        *extra,
        "",
        f"Source rows: {source_rows}; emitted rows: {kept}.",
    ]
    lines.extend(f"  dropped-{reason}: {n}" for reason, n in counts.items() if n)
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="task", required=True)

    syn = sub.add_parser("synonym", help="build synonym multiple choice")
    syn.add_argument("--glossary", required=True, help="TSV: term<TAB>synonym")
    syn.add_argument("--distractors", type=int, default=DEFAULT_DISTRACTORS)
    syn.add_argument("--seed", type=int, default=0)

    cat = sub.add_parser("category", help="build a category-purity set")
    cat.add_argument("--taxonomy", required=True, help="TSV: word<TAB>category")

    for p in (syn, cat):
        p.add_argument("--out", required=True, help="output data_dir root")
        p.add_argument("--lang", default="en")
        p.add_argument("--name", required=True, help="dataset name (no suffix)")

    args = parser.parse_args(argv)

    source = args.glossary if args.task == "synonym" else args.taxonomy
    payload = Path(source).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    rows = read_pairs(source)

    out = Path(args.out) / args.lang / args.task / f"{args.name}.tsv"

    if args.task == "synonym":
        if args.distractors < 2:
            raise SystemExit("--distractors must be at least 2")
        built, counts = build_synonym(rows, args.distractors, args.seed)
        header = [
            "target",
            "answer",
            *(f"distractor{i}" for i in range(1, args.distractors + 1)),
        ]
        extra = [
            f"Distractors per question: {args.distractors}; sampling seed: "
            f"{args.seed}. Distractors are drawn from the glossary's own terms,",
            "never from a term known to be synonymous with the target, and the",
            "draw is deterministic -- rebuilding gives a bit-identical file.",
            f"Chance accuracy for this file is 1/{args.distractors + 1} = "
            f"{1 / (args.distractors + 1):.3f}; compare against that, not 0.",
        ]
    else:
        built, counts = build_category(rows)
        header = ["word", "category"]
        categories = {row[1] for row in built}
        extra = [
            f"Categories: {len(categories)}. Scored by nearest-neighbour purity",
            "within this dataset; the evaluator reports the chance floor for",
            "these category sizes alongside the score.",
        ]
        if len(categories) < 2:
            print(
                "warning: fewer than 2 categories survive; the evaluator will "
                "skip this dataset because purity would be trivially 1"
            )

    write_tsv(
        out,
        header,
        preamble_lines(
            kind=args.task,
            source=Path(source).name,
            source_sha256=digest,
            extra=extra,
            counts=counts,
            source_rows=len(rows),
            kept=len(built),
        ),
        built,
    )


if __name__ == "__main__":
    main()
