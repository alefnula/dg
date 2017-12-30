__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 30 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'

import dg
import io
import os
import json
import hashlib
import sqlite3
import numpy as np
from dg.utils import ensure_dir
from dg.model import Model, conform_sklearn_to_model, strip_model_function
from sklearn.externals import joblib


def save(model, model_dir):
    """Save the model

    Args:
        model (dg.models.Model): Model that should be saved
        model_dir (str): Path to the directory where the model should be
            saved.
    """
    model_dir = os.path.join(model_dir, model.id)
    ensure_dir(model_dir, directory=True)
    if hasattr(model, 'save'):
        model.save(model_dir)
    else:
        model_file = os.path.join(model_dir, f'{model.id}.pickle')
        with io.open(model_file, 'wb') as f:
            # strip object
            if not isinstance(model, Model):
                model = strip_model_function(model)
                joblib.dump(model, f)
                # Return the functionality
                model = conform_sklearn_to_model(model.id, model)
            else:
                joblib.dump(model, f)


def load(model, model_dir=None):
    """Load the model

    Args:
        model (dg.models.Model): Model class or instance of the model
        model_dir (str): If `model_dir` is provided, loads the model from
            the model dir, else loads the production model.
    Returns:
        Estimator: Returns the estimator loaded from the save point
    """
    model_dir = model_dir or dg.Config().get_model_dir(production=True)
    model_dir = os.path.join(model_dir, model.id)
    if hasattr(model, 'load'):
        model = model.load(model_dir)
    else:
        model_file = os.path.join(model_dir, f'{model.id}.pickle')
        with io.open(model_file, 'rb') as f:
            model = joblib.load(f)
    if not isinstance(model, Model):
        model = conform_sklearn_to_model(model.id, model)
    return model


class Database(object):
    CREATE_TABLE = '''
    CREATE TABLE IF NOT EXISTS results (
      id INTEGER PRIMARY KEY,
      model VARCHAR(256),
      hash VARCHAR(64) UNIQUE,
      parameters VARCHAR,
      metrics VARCHAR
    );
    '''

    CREATE_INDEX = '''
    CREATE UNIQUE INDEX IF NOT EXISTS hash_index ON results(hash);
    '''

    INSERT_SQL = '''
    INSERT INTO results (model, hash, parameters, metrics)
    VALUES (?, ?, ?, ?);
    '''

    SELECT_SQL = '''
    SELECT metrics FROM results
    WHERE model = ? AND hash = ?;
    '''

    UPDATE_SQL = '''
    UPDATE results SET metrics = ?
    WHERE model = ? AND hash = ?;
    '''

    SINGLE_TYPES = {int, str, float, bool, type(None)} | set(
        np.typeDict.values())

    def convert(self, x):
        t = type(x)
        if t in self.SINGLE_TYPES:
            return x
        elif t in (list, tuple, set):
            return t(self.convert(i) for i in x)
        elif t == type:
            return x.__name__
        else:
            return x.__class__.__name__

    def encode_dict(self, d):
        params = {
            str(name): self.convert(value)
            for name, value in d.items()
        }
        return json.dumps(params, sort_keys=True)

    def hash_dict(self, d):
        return hashlib.sha256(self.encode_dict(d).encode('utf-8')).hexdigest()

    def __init__(self):
        self.config = dg.Config()
        self._create()

    def _create(self):
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            cur.execute(self.CREATE_TABLE)
            cur.execute(self.CREATE_INDEX)
            c.commit()

    def get(self, model):
        params_hash = self.hash_dict(model.get_params())
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            cur.execute(self.SELECT_SQL, (model.id, params_hash))
            all = cur.fetchall()
            if len(all) == 0:
                return {'train': None, 'eval': None, 'test': None}
            else:
                return json.loads(all[0][0])

    def add(self, model, metrics):
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            params_json = self.encode_dict(model.get_params())
            params_hash = self.hash_dict(model.get_params())
            metrics_json = json.dumps(metrics, sort_keys=True)
            cur.execute(self.INSERT_SQL,
                        (model.id, params_hash, params_json, metrics_json))
            c.commit()

    def update(self, model, metrics):
        with sqlite3.connect(self.config.metrics_db) as c:
            cur = c.cursor()
            params_hash = self.hash_dict(model.get_params())
            metrics_json = json.dumps(metrics, sort_keys=True)
            cur.execute(self.UPDATE_SQL, (metrics_json, model.id, params_hash))
