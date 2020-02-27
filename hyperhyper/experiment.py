"""
store and retrieve experiment results in a database
"""

import time

import sqlalchemy

from .pair_counts import default_pair_args


def flatten_dict(prefix, dict):
    """
    flatten a dict Django-style
    """
    for k, v in dict.items():
        yield {f"{prefix}__{k}": v}


def record(func):
    """
    record the evaluation of an embedding in a database
    """

    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)

        if not "pair_args" in kwargs:
            kwargs["pair_args"] = default_pair_args

        if "evaluate" in kwargs and not kwargs["evaluate"]:
            return results

        if len(results) > 1:
            db_dic = {}
            # params to dict
            print(func.__name__)
            db_dic.update({"method": func.__name__})
            for k, v in kwargs.items():
                if type(v) is dict:
                    print("k", k)
                    if k == "pair_args":
                        # merge with default arguments of pair counts
                        v = {**default_pair_args, **v}
                    for x in flatten_dict(k, v):
                        db_dic.update(x)
                else:
                    db_dic.update({k: v})
            # results to dicts
            db_dic.update({"micro_results": results[1]["micro"]})
            db_dic.update({"macro_results": results[1]["macro"]})
            for r in results[1]["results"]:
                db_dic.update({f"{r['name']}_score": r["score"]})
                db_dic.update({f"{r['name']}_oov": r["oov"]})
                db_dic.update({f"{r['name']}_fullscore": r["fullscore"]})

            # couldn't figure out the timeout param for datasets
            while True:
                try:
                    # args[0] is self
                    db = args[0].get_db()
                    table = db["experiments"]
                    # specify type because dataset guesses them sometimes wrongly
                    # ensure that rows are not duplicated. This may happen, if the same function is called multiple times.
                    table.insert_ignore(
                        db_dic,
                        db_dic.keys(),
                        types={
                            k: sqlalchemy.types.String
                            if type(v) is str
                            else sqlalchemy.types.Float
                            for k, v in db_dic.items()
                        },
                    )
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)
        return results

    return wrapper


def results_from_db(db, query={}, order="micro_results desc", limit=100):
    """
    retrieve (the best) results from a database
    """
    where = []
    for k, v in query.items():
        if type(v) is dict:
            for fkfv in flatten_dict(k, v):
                # ugly
                for fk, fv in fkfv.items():
                    where.append(f"{fk}={fv}")
        else:
            where.append(f"{k}={v}")
    if len(where) > 0:
        where = "where " + " and ".join(where)
    else:
        where = ""

    if order is None:
        order = ""
    if len(order) > 0:
        order = f"order by {order}"

    if limit is None:
        limit = ""
    else:
        limit = f"limit {limit}"

    query_string = f"select distinct * from experiments {where} {order} {limit}"
    return list(db.query(query_string))
