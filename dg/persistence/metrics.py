__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
import json
import hashlib
import sqlite3
import functools
import numpy as np
from datetime import datetime
from contextlib import contextmanager
from dg.persistence import metrics_sql as sql


SINGLE_TYPES = {int, str, float, bool, type(None)}
NUMPY_TYPES = set(np.typeDict.values())


def convert(x):
    t = type(x)
    if t in SINGLE_TYPES:
        return x
    elif t in NUMPY_TYPES:
        return x.item()
    elif t in (list, tuple, set, np.ndarray):
        return [convert(i) for i in x]
    elif t == dict:
        return {
            str(key): convert(value)
            for key, value in x.items()
        }
    elif t == type:
        return x.__name__
    else:
        return x.__class__.__name__


def encode_dict(d):
    return json.dumps({
        str(name): convert(value)
        for name, value in d.items()
    }, sort_keys=True)


def hash_dict(d):
    return hashlib.sha256(encode_dict(d).encode('utf-8')).hexdigest()


def secure(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            return None
    return wrapper


class Database(object):
    def __init__(self):
        self.config = dg.Config()
        self._create()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.config.metrics_db,
                               detect_types=sqlite3.PARSE_DECLTYPES)
        yield conn
        conn.close()

    def _create(self):
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            cur.execute(sql.CREATE_TABLE)
            for create_index in sql.CREATE_INDEX:
                cur.execute(create_index)
            c.commit()

    def get(self, model):
        try:
            params_hash = hash_dict(model.get_params())
            with sqlite3.connect(self.config.metrics_db) as c:
                cur = c.cursor()
                cur.execute(sql.SELECT_SQL, (model.id, params_hash))
                all = cur.fetchall()
                if len(all) == 0:
                    return {'train': None, 'eval': None, 'test': None}
                else:
                    return json.loads(all[0][0])
        except:
            return {'train': None, 'eval': None, 'test': None}

    @secure
    def add(self, model, metrics):
        with self._connect() as c:
            cur = c.cursor()
            params_json = encode_dict(model.get_params())
            params_hash = hash_dict(model.get_params())
            metrics_json = encode_dict(metrics)
            cur.execute(sql.INSERT_SQL,
                        (model.id, params_hash, datetime.now(),
                         params_json, metrics_json))
            c.commit()

    @secure
    def update(self, model, metrics):
        with self._connect() as c:
            cur = c.cursor()
            params_hash = hash_dict(model.get_params())
            metrics_json = encode_dict(metrics)
            cur.execute(sql.UPDATE_SQL, (metrics_json, model.id, params_hash))
            c.commit()

    def metrics(self, models):
        with self._connect() as c:
            try:
                cur = c.cursor()
                in_clause = ', '.join(['?'] * len(models))
                cur.execute(sql.METRICS_SELECT.format(in_clause), list(models))
                return [
                    (model, timestamp, json.loads(params), json.loads(metrics))
                    for model, timestamp, params, metrics in cur.fetchall()
                ]
            except:
                return []
