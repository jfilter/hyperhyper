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
            db_dic.update({"method": func.__name__})
            for k, v in kwargs.items():
                if type(v) is dict:
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


# TODO
# def get_embedding_from_params(row):
#     pair_args = {}
#     args = {}
#     for k, v in row.items():
#         k_parts = k.split("__")
#         if len(k_parts) > 1:
#             pair_args[k_parts[1]] = v
#         else:
#             arg[k] = v

#     for best in list(db.query(statement)):
#         oov = True if best["pair_args__delete_oov"] == 1 else False
#         window = int(best["pair_args__window"])
#         if not isinstance(window, int):
#             window = int.from_bytes(window, "little")
#         neg = float(best["neg"])
#         if neg.is_integer():
#             neg = int(neg)
#         dim = int(best["dim"])

#         print(oov, best)
#         try:
#             print(best["neg"])
#             kv, res = b.svd(
#                 impl="scipy",
#                 evaluate=True,
#                 pair_args={
#                     "subsample": "deter",
#                     "subsample_factor": best["pair_args__subsample_factor"],
#                     "delete_oov": True,
#                     "decay_rate": best["pair_args__decay_rate"],
#                     "window": window,
#                     "dynamic_window": "decay",
#                 },
#                 neg=neg,
#                 eig=best["eig"],
#                 dim=dim,
#                 keyed_vector=True,
#             )
#             print(res)
#             print(best)
#         except Exception as e:
#             print(e)
#     return kv


# def get_best(db, query):
