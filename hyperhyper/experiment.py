import time


def flatten_dict(prefix, dict):
    for k, v in dict.items():
        yield {f"{prefix}__{k}": v}


# TODO: make sure to store byes to boolean?


def record(func):
    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)

        if "evaluate" in kwargs and not kwargs["evaluate"]:
            return results

        if len(results) > 1:
            db_dic = {}
            # params to dict
            print(func.__name__)
            db_dic.update({"method": func.__name__})
            for k, v in kwargs.items():
                if type(v) is dict:
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

            # couldn't figure out the timeout param for datasets
            while True:
                try:
                    # args[0] is self
                    db = args[0].get_db()
                    table = db["experiments"]
                    table.insert(db_dic)
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)
        return results

    return wrapper
