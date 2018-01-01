__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
import json
import hashlib
import sqlite3
import numpy as np
from dg.persistence import metrics_sql as sql


SINGLE_TYPES = {int, str, float, bool, type(None)} | set(np.typeDict.values())


def convert(x):
    t = type(x)
    if t in SINGLE_TYPES:
        return x
    elif t in (list, tuple, set):
        return t(convert(i) for i in x)
    elif t == type:
        return x.__name__
    else:
        return x.__class__.__name__


def encode_dict(d):
    params = {
        str(name): convert(value)
        for name, value in d.items()
    }
    return json.dumps(params, sort_keys=True)


def hash_dict(d):
    return hashlib.sha256(encode_dict(d).encode('utf-8')).hexdigest()


class Database(object):
    def __init__(self):
        self.config = dg.Config()
        self._create()

    def _create(self):
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            cur.execute(sql.CREATE_TABLE)
            cur.execute(sql.CREATE_INDEX)
            c.commit()

    def get(self, model):
        params_hash = hash_dict(model.get_params())
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            cur.execute(sql.SELECT_SQL, (model.id, params_hash))
            all = cur.fetchall()
            if len(all) == 0:
                return {'train': None, 'eval': None, 'test': None}
            else:
                return json.loads(all[0][0])

    def add(self, model, metrics):
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            params_json = encode_dict(model.get_params())
            params_hash = hash_dict(model.get_params())
            metrics_json = json.dumps(metrics, sort_keys=True)
            cur.execute(sql.INSERT_SQL,
                        (model.id, params_hash, params_json, metrics_json))
            c.commit()

    def update(self, model, metrics):
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            params_hash = hash_dict(model.get_params())
            metrics_json = json.dumps(metrics, sort_keys=True)
            cur.execute(sql.UPDATE_SQL, (metrics_json, model.id, params_hash))

    def metrics(self, models):
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            in_clause = ', '.join(['?'] * len(models))
            cur.execute(sql.METRICS_SELECT.format(in_clause), models)
            return [
                (model, json.loads(params), json.loads(metrics))
                for model, params, metrics in cur.fetchall()
            ]
