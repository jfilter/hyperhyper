"""
store and retrieve experiment results in a database
"""

import functools
import inspect
import logging
import re
import time
from collections.abc import Mapping

import sqlalchemy

from .pair_counts import default_pair_args

logger = logging.getLogger(__name__)

MAX_DB_ATTEMPTS = 5

# Flags that shape the *return value* rather than the embedding. Two calls that
# differ only in these produce the same vectors and the same scores, so they
# must not become separate rows -- nor separate filterable columns.
NON_PARAMETERS = frozenset({"keyed_vectors", "evaluate"})

_COLUMN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ORDER_TERM = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+(asc|desc))?$", re.IGNORECASE)


def flatten_dict(prefix, mapping):
    """
    flatten a dict Django-style
    """
    for k, v in mapping.items():
        yield {f"{prefix}__{k}": v}


def effective_arguments(func, args, kwargs):
    """
    Resolve a call into the arguments `func` really ran with.

    Inspecting only `kwargs` loses every positional argument (`b.svd(5)` and
    `b.svd(10)` become indistinguishable rows) and every default that was not
    spelled out (a run at the default `cds=0.75` is invisible to a
    `query={"cds": 0.75}`). Returns `(named, leftover)`, where `leftover` is
    whatever landed in the function's `**kwargs`.
    """
    signature = inspect.signature(func)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()

    named = dict(bound.arguments)
    named.pop("self", None)

    leftover = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            leftover = named.pop(name, None) or {}

    return named, leftover


def record(func):
    """
    record the evaluation of an embedding in a database
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)

        params, leftover = effective_arguments(func, args, kwargs)

        if not params.get("evaluate", True):
            return results

        # Loose `**kwargs` are forwarded all the way down to `count_pairs`, so
        # they belong *in* `pair_args`. Emitting them as sibling columns made a
        # row claim `pair_args__window=2` and `window=10` for the same run.
        params["pair_args"] = {
            **default_pair_args,
            **(params.get("pair_args") or {}),
            **leftover,
        }

        if len(results) > 1:
            db_dic = {}
            # params to dict
            db_dic.update({"method": func.__name__})
            for k, v in params.items():
                if k in NON_PARAMETERS:
                    continue
                # `Mapping`, not `dict`: `default_pair_args` is a MappingProxyType
                if isinstance(v, Mapping):
                    for x in flatten_dict(k, v):
                        db_dic.update(x)
                else:
                    db_dic.update({k: v})

            # Everything added from here on is a *score*, not a parameter, so
            # the dedupe key is fixed now: keying on the whole row would let a
            # float score that differs in its last digit create a duplicate.
            parameter_columns = list(db_dic.keys())

            # results to dicts
            db_dic.update({"micro_results": results[1]["micro"]})
            db_dic.update({"macro_results": results[1]["macro"]})
            for r in results[1]["results"]:
                db_dic.update({f"{r['name']}_score": r["score"]})
                db_dic.update({f"{r['name']}_oov": r["oov"]})
                db_dic.update({f"{r['name']}_fullscore": r["fullscore"]})

            # The database may be locked by a concurrent writer, so retry a
            # couple of times with an exponential backoff before giving up.
            for attempt in range(MAX_DB_ATTEMPTS):
                try:
                    # the first positional argument is the bunch instance
                    db = args[0].get_db()
                    table = db["experiments"]
                    # specify type because dataset guesses them sometimes wrongly
                    # ensure that rows are not duplicated. This may happen, if the same function is called multiple times.
                    table.insert_ignore(
                        db_dic,
                        parameter_columns,
                        types={
                            k: sqlalchemy.types.String
                            if isinstance(v, str)
                            else sqlalchemy.types.Float
                            for k, v in db_dic.items()
                        },
                    )
                    break
                except sqlalchemy.exc.SQLAlchemyError as e:
                    logger.warning(
                        "db write failed (attempt %d/%d): %s",
                        attempt + 1,
                        MAX_DB_ATTEMPTS,
                        e,
                    )
                    # no point backing off after the final attempt
                    if attempt < MAX_DB_ATTEMPTS - 1:
                        time.sleep(2**attempt)
            else:
                logger.error("giving up on db write after %d attempts", MAX_DB_ATTEMPTS)
        return results

    return wrapper


def _order_clause(order):
    """
    Validate an `order by` expression against a strict `column [asc|desc]` form.

    It cannot be a bound parameter, so it is whitelisted instead of interpolated.
    """
    if not order:
        return ""
    terms = []
    for term in str(order).split(","):
        match = _ORDER_TERM.match(term.strip())
        if match is None:
            raise ValueError(
                f"invalid `order` expression: {term.strip()!r}; "
                "expected `column` or `column asc|desc`"
            )
        column, direction = match.group(1), match.group(2)
        terms.append(f"{column} {direction.lower()}" if direction else column)
    return "order by " + ", ".join(terms)


def results_from_db(db, query=None, order="micro_results desc", limit=100):
    """
    retrieve (the best) results from a database
    """
    query = {} if query is None else query
    where = []
    bound = {}

    def add_condition(column, value):
        # Values were interpolated raw, so any string filter became a bare
        # identifier: `{"impl": "scipy"}` asked SQLite for a column `scipy`.
        if not _COLUMN.match(str(column)):
            raise ValueError(f"invalid column name in `query`: {column!r}")
        name = f"p{len(bound)}"
        where.append(f"{column} = :{name}")
        bound[name] = value

    for k, v in query.items():
        if isinstance(v, Mapping):
            for fkfv in flatten_dict(k, v):
                # ugly
                for fk, fv in fkfv.items():
                    add_condition(fk, fv)
        else:
            add_condition(k, v)

    where = "where " + " and ".join(where) if len(where) > 0 else ""
    order = _order_clause(order)
    # coerced, never interpolated verbatim
    limit = "" if limit is None else f"limit {int(limit)}"

    query_string = f"select distinct * from experiments {where} {order} {limit}"
    return list(db.query(query_string, **bound))
