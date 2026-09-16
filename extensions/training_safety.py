"""Cross-host exclusion for writes to a trained model.

Use a dedicated PostgreSQL session so ORM connection cleanup cannot release the
lock. Workers should also cancel late-acknowledged tasks on broker loss.
"""
from contextvars import ContextVar
from functools import wraps
from inspect import signature
import os

_lock_connection = ContextVar('training_lock_connection', default=None)
_LOCK_NAMESPACE = 1163084626

def assert_training_lock():
    connection = _lock_connection.get()
    if connection is not None:
        # Never reconnect this session: a lost connection means a lost lease.
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
            assert cursor.fetchone()[0] == 1

def exclusive_training(function):
    params = signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        if os.getenv('DISTRIBUTED_GPU_ENABLED') != '1':
            return function(*args, **kwargs)
        model_pk = params.bind_partial(*args, **kwargs).arguments.get('model_pk')
        if model_pk is None:
            raise ValueError('Distributed training requires an explicit model_pk')
        from django.db import connections
        # Use a separate connection for the lifetime of this task.
        backend = connections['default']
        connection_params = backend.get_connection_params()
        connection_params.update(connect_timeout=10, application_name='escriptorium-training-lease',
                                 keepalives=1, keepalives_idle=10, keepalives_interval=5, keepalives_count=2)
        connection = backend.Database.connect(**connection_params)
        token = None
        try:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute('SELECT pg_try_advisory_lock(%s, %s)', [_LOCK_NAMESPACE, int(model_pk)])
                if not cursor.fetchone()[0]:
                    raise RuntimeError(f'Training is already active for model {model_pk}')
            token = _lock_connection.set(connection)
            result = function(*args, **kwargs)
            assert_training_lock()
            return result
        finally:
            if token is not None:
                _lock_connection.reset(token)
            # Closing the session releases the lock, including on task failure.
            connection.close()
    return wrapped
