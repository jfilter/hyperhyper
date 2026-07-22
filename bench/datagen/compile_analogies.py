#!/usr/bin/env python3
"""
Offline analogy-dataset builder (ADR 0001, P2).

Turns a file of verified one-to-one *base pairs* (`capital<TAB>country`) into a
frozen four-column analogy `.txt` in the exact format the runtime reads, and
writes it into the package's `evaluation_datasets/` tree.

This is a *build tool*, not a runtime dependency: it imports only the standard
library plus `hyperhyper.preprocessing.tokenize_string` (to apply the package's
own single-token rule). It performs no network calls and loads no
LLM/torch/transformers -- fact verification happens out of band by a human with
web sources, and is recorded in the base-pairs file (see `pairs/*.tsv`) and in
the emitted header. The frozen `.txt` is the artifact of record; re-running this
script is an audit/regeneration step, never something the package does at
install, train or eval time.

Design (per the ADR):

* Verify N base *pairs*, never N**2 quadruples. Each pair is one line in the TSV.
* Every capital and country must reduce to exactly one token under the package
  preprocessing; multi-token forms are rejected loudly rather than silently
  truncated.
* Quadruples are compiled *deterministically* with a fixed seed and a fixed cap
  K: each base pair is the source (leading) pair of exactly K questions, so the
  file grows as N*K, not the N*(N-1) full Cartesian product the ADR forbids.
* The provenance header is written as `#`-comment lines, and this script asserts
  that not one of them splits into exactly four whitespace tokens -- otherwise it
  would leak into the parser as a bogus data row (see `setup_test_tokens`).

Usage:
    python compile_analogies.py \
        --pairs pairs/fr_capitals.tsv \
        --out ../../hyperhyper/evaluation_datasets/fr/analogy/capitals.txt \
        --lang fr --relation capital-country --cap 5 --seed 0

Run with no arguments to rebuild the shipped French capitals set with the frozen
defaults, then self-check the round-trip through the package parser.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# The build tool is allowed exactly one dependency on the package: the very
# preprocessing the runtime uses, so "single token" here means the same thing it
# means at eval time. Make an editable/installed checkout importable when the
# script is run straight from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hyperhyper.preprocessing import tokenize_string

HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "pairs" / "fr_capitals.tsv"
DEFAULT_OUT = (
    HERE.parents[1]
    / "hyperhyper"
    / "evaluation_datasets"
    / "fr"
    / "analogy"
    / "capitals.txt"
)
GENERATOR = "Claude Opus 4.8 (LLM-proposed, human web-verified)"
BUILD_DATE = "2026-07-22"
ANALOGY_FIELDS = 4  # a a_ b b_


def load_pairs(path: Path) -> list[tuple[str, str]]:
    """Read `capital<TAB>country[...]` rows, skipping `#`-comment/blank lines."""
    pairs: list[tuple[str, str]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) < 2:
            raise ValueError(
                f"{path}:{lineno}: expected TAB-separated columns: {raw!r}"
            )
        pairs.append((fields[0].strip(), fields[1].strip()))
    return pairs


def check_single_token(pairs: list[tuple[str, str]]) -> None:
    """Reject any term the package preprocessing would not reduce to one token.

    A multi-token entry is unscoreable (`to_item` drops it), so it must never
    reach the frozen file. Fail loudly with the offending term.
    """
    offenders = []
    for a, b in pairs:
        for term in (a, b):
            toks = tokenize_string(term)
            if len(toks) != 1:
                offenders.append((term, toks))
    if offenders:
        raise SystemExit(f"multi-token terms (drop or fix these): {offenders}")


def check_distinct(pairs: list[tuple[str, str]]) -> None:
    """No repeated capital or country, and no `capital == country` self-pair."""
    caps = [a for a, _ in pairs]
    countries = [b for _, b in pairs]
    if len(set(caps)) != len(caps):
        dups = sorted({c for c in caps if caps.count(c) > 1})
        raise SystemExit(f"duplicate capitals: {dups}")
    if len(set(countries)) != len(countries):
        dups = sorted({c for c in countries if countries.count(c) > 1})
        raise SystemExit(f"duplicate countries: {dups}")
    self_pairs = [(a, b) for a, b in pairs if a == b]
    if self_pairs:
        raise SystemExit(f"capital equals country: {self_pairs}")


def compile_quadruples(
    pairs: list[tuple[str, str]], cap: int, seed: int
) -> list[tuple[str, str, str, str]]:
    """Deterministically form `a a_ b b_` questions, capping per base pair.

    Each base pair `i` is the *source* pair of exactly `min(cap, N-1)` questions:
    partners `j != i` are drawn with a seeded RNG, and the question is
    `(cap_i, country_i, cap_j, country_j)`. This caps questions per base pair (no
    Cartesian blow-up: N*cap rows, not N*(N-1)), keeps the build reproducible,
    and -- because all capitals and all countries are distinct and no capital
    equals a country -- guarantees four distinct tokens with `b_` never colliding
    with `a`, `a_` or `b`.
    """
    rng = random.Random(seed)
    n = len(pairs)
    k = min(cap, n - 1)
    quads: list[tuple[str, str, str, str]] = []
    for i, (cap_i, country_i) in enumerate(pairs):
        others = [j for j in range(n) if j != i]
        partners = rng.sample(others, k)
        for j in partners:
            cap_j, country_j = pairs[j]
            quads.append((cap_i, country_i, cap_j, country_j))
    return quads


def build_header(
    lang: str, relation: str, n_pairs: int, n_quads: int, cap: int, seed: int
) -> list[str]:
    """Provenance `#`-comment lines. Asserts none splits into 4 tokens."""
    lines = [
        f"# {lang}/analogy/capitals.txt -- generated offline by bench/datagen (ADR 0001, phase P2).",
        f"# Generator model was {GENERATOR}; build date {BUILD_DATE}.",
        "# Status of this file: LLM-proposed base pairs, then independently fact-verified against non-LLM sources; NOT a raw LLM dump.",
        f"# Relation type is {relation}; it is one-to-one and fact-verifiable (a capital has exactly one country).",
        "# Verification sources (2026-07-22): Wikipedia 'List of national capitals' for the capital fact,",
        "#   https://en.wikipedia.org/wiki/List_of_national_capitals ;",
        "#   French Wikipedia 'Liste des capitales du monde' for the French spelling,",
        "#   https://fr.wikipedia.org/wiki/Liste_des_capitales_du_monde .",
        f"# Verified base pairs used: {n_pairs}, at a 100 percent verification rate; canonical pairs live in bench/datagen/pairs/fr_capitals.tsv.",
        f"# Compiled deterministically with seed {seed} and a per-base-pair source cap of K equal to {cap} questions; {n_quads} quadruples total.",
        "# Every term is a single token under hyperhyper.preprocessing.tokenize_string; multi-token country names were excluded upstream.",
        "# These comment lines are ignored by the parser: setup_test_tokens keeps only 4-field lines, and none of these lines has exactly four whitespace-separated fields.",
    ]
    leaks = [ln for ln in lines if len(ln.split()) == ANALOGY_FIELDS]
    if leaks:
        raise SystemExit(
            "header line(s) would split into exactly 4 tokens and leak as data: "
            + repr(leaks)
        )
    return lines


def write_dataset(out: Path, header: list[str], quads) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{a}\t{a_}\t{b}\t{b_}" for a, a_, b, b_ in quads)
    out.write_text("\n".join(header) + "\n" + body + "\n", encoding="utf-8")


def selfcheck(out: Path, quads) -> None:
    """Round-trip the frozen file through the package parser and confirm exactly
    the intended quadruples come back, with zero comment leakage."""
    from hyperhyper.evaluation import read_test_data, setup_test_tokens

    # locate the file the way the runtime does, via read_test_data(lang, kind)
    lang = out.parent.parent.name
    found = [p for p in read_test_data(lang, "analogy") if p.name == out.name]
    if not found:
        raise SystemExit(f"read_test_data({lang!r}, 'analogy') did not find {out.name}")
    columns = list(setup_test_tokens(found[0], ANALOGY_FIELDS))
    parsed = list(zip(*columns, strict=True)) if columns else []
    if parsed != quads:
        raise SystemExit(
            f"round-trip mismatch: parser returned {len(parsed)} rows, expected {len(quads)}"
        )
    print(f"self-check OK: {len(parsed)} quadruples round-trip with no comment leakage")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lang", default="fr")
    ap.add_argument("--relation", default="capital-country")
    ap.add_argument(
        "--cap", type=int, default=5, help="questions sourced per base pair"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    pairs = load_pairs(args.pairs)
    check_single_token(pairs)
    check_distinct(pairs)
    quads = compile_quadruples(pairs, cap=args.cap, seed=args.seed)
    header = build_header(
        args.lang, args.relation, len(pairs), len(quads), args.cap, args.seed
    )
    write_dataset(args.out, header, quads)
    print(
        f"wrote {args.out} : {len(pairs)} base pairs -> {len(quads)} quadruples "
        f"(cap K={args.cap}, seed={args.seed})"
    )
    selfcheck(args.out, quads)


if __name__ == "__main__":
    main()
