import os,time,threading,concurrent.futures
os.environ.setdefault('DJANGO_SETTINGS_MODULE','escriptorium.settings')
os.environ['DISTRIBUTED_GPU_ENABLED']='1'
import django;django.setup()
from training_safety import exclusive_training,assert_training_lock,_lock_connection
from django.db import connections
entered=threading.Event();release=threading.Event()
@exclusive_training
def hold(model_pk):
 entered.set();assert release.wait(15);return 'first'
@exclusive_training
def quick(model_pk):return 'next'
with concurrent.futures.ThreadPoolExecutor(2) as pool:
 first=pool.submit(hold,2000000100)
 if not entered.wait(10):first.result(timeout=1);raise AssertionError('Lock entry timed out')
 try:quick(2000000100)
 except RuntimeError as e:assert 'already active' in str(e)
 else:raise AssertionError('Duplicate training was allowed')
 assert quick(2000000101)=='next'
 release.set();assert first.result()=='first'
assert quick(2000000100)=='next'
@exclusive_training
def lose_connection(model_pk):
 with _lock_connection.get().cursor() as c:c.execute('SELECT pg_backend_pid()');pid=c.fetchone()[0]
 with connections['default'].cursor() as c:c.execute('SELECT pg_terminate_backend(%s)',[pid])
 assert_training_lock()
try:lose_connection(2000000100)
except Exception as e:
 assert type(e).__name__ in ('OperationalError','InterfaceError'),type(e).__name__
else:raise AssertionError('Lost lock silently reconnected')
assert quick(2000000100)=='next'
print('PASS: duplicate exclusion, independent models, release, and lost-connection fence')

