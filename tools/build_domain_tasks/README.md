# `tools/build_domain_tasks` — domain proxy evaluation tasks

Builds the two **domain proxy tasks** of ADR 0001, phase P4, from data a domain
already has. Not shipped in the wheel; nothing here runs at install, train or
eval time.

```
uv run python tools/build_domain_tasks/build_domain_tasks.py synonym \
    --glossary my_glossary.tsv --out eval_data --lang en --name glossary

uv run python tools/build_domain_tasks/build_domain_tasks.py category \
    --taxonomy my_taxonomy.tsv --out eval_data --lang en --name taxonomy
```

Then:

```python
bunch.dataset_coverage(kind="synonym", data_dir="eval_data")   # run this first
bunch.eval_synonym(embedding, data_dir="eval_data")
bunch.eval_category(embedding, data_dir="eval_data")
```

## The problem these solve

The bundled similarity and analogy sets measure **general language**. This
package is for **small, domain-specific** corpora — and on those, a general set
is largely out-of-vocabulary, so it measures close to nothing. ADR 0001 records
that as the central tension, and P4 as the long-term answer.

The tempting fix — have a model rate domain word pairs for similarity — is the
one thing ADR 0001 forbids: it turns "agreement with humans" into "agreement
with a language model", a circular benchmark. So these tasks are built the other
way round, on gold that is a **membership fact** the domain already records:

| Task | Gold comes from | Verifying it is |
|---|---|---|
| synonym multiple choice | a glossary/thesaurus entry | a lookup |
| category purity | a taxonomy's class assignment | a lookup |

No judgement is elicited, so none can be fabricated.

## Input formats

Both are plain two-column TSV; `#` comments and a header row are skipped.

```
# glossary: term <TAB> synonym   (repeat the term for several synonyms)
hypertension	high-blood-pressure
tachycardia	palpitations

# taxonomy: word <TAB> category
aspirin	drug
liver	organ
```

## What the builder guarantees

**Distractors are never accidentally right.** A distractor is excluded if it is
a known synonym of the target *anywhere* in the glossary — not merely if it is
this row's answer — and synonymy is treated as symmetric. Without this a "wrong"
option is silently right, and the score is capped below what the embedding
deserves, invisibly.

**Distractors are in-domain.** They are drawn from the glossary's own terms.
Sampling general-language words would make the task easy for the wrong reason:
any domain-trained embedding separates domain terms from unrelated ones without
knowing anything about synonymy.

**The build is deterministic.** Candidates are sorted before sampling, so output
depends on the glossary and the seed — not on dictionary iteration order.
Rebuilding gives a bit-identical file.

**Nothing is padded to hit a row count.** A term with too few eligible
distractors is skipped, a multi-token entry is dropped, and a word with
conflicting category labels is dropped rather than having a winner picked for
it. Every drop is counted in the emitted preamble.

## Reading the scores

**Neither score has a floor of 0**, which is the easiest way to misread them:

- Synonym accuracy has a **chance floor of `1/(K+1)`** for `K` distractors —
  0.25 with the default 3. The emitted preamble states it per file.
- Category purity's chance floor depends on the category sizes, so the evaluator
  reports a **`baseline`** next to each `score`. One dominant category pushes
  that floor high; a score below it is worse than random.

Run `dataset_coverage` before training. A glossary-built set whose coverage is
near zero cannot be scored at all, and finding that out early is the point.

## Why nothing is bundled

A general-language synonym set would recreate exactly the problem P4 exists to
solve. The useful version of each task is built from *your* glossary, so these
live behind `data_dir`.
