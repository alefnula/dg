__author__ = 'Viktor Kerkez <alefnula@gmail.com>'
__date__ = ' 31 December 2017'
__copyright__ = 'Copyright (c)  2017 Viktor Kerkez'


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

METRICS_SELECT = '''
SELECT model, parameters, metrics
FROM results
WHERE model IN ({});
'''
