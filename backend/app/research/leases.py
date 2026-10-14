"""Session-held PostgreSQL leases shared by CLI stages and queued workers."""
from contextlib import contextmanager
from sqlalchemy import text
from ..db import engine


def stage_active(db,name):
    # Observe the actual granted lease without acquiring a competing lock.
    return bool(db.scalar(text('''SELECT EXISTS (
        SELECT 1 FROM pg_locks WHERE locktype='advisory' AND granted AND objsubid=1
        AND database=(SELECT oid FROM pg_database WHERE datname=current_database())
        AND classid=((hashtextextended(:key,0) >> 32) & 4294967295)::oid
        AND objid=(hashtextextended(:key,0) & 4294967295)::oid)'''),{'key':'kneews-stage-'+name}))


@contextmanager
def stage_lease(name):
    with engine.connect() as connection:
        key='kneews-stage-'+name
        acquired=connection.scalar(text('SELECT pg_try_advisory_lock(hashtextextended(:key,0))'),{'key':key})
        connection.commit()
        if not acquired:raise RuntimeError(f'Another {name} stage is already running')
        try:yield
        finally:
            connection.execute(text('SELECT pg_advisory_unlock(hashtextextended(:key,0))'),{'key':key})
            connection.commit()
