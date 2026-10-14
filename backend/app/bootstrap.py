"""Create only the dedicated local KneeWS database, then its initial schema."""
from sqlalchemy.engine import make_url
import psycopg
from psycopg import sql
from .config import settings
from .db import Base,engine
from . import models
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.autogenerate import compare_metadata
from sqlalchemy import inspect
from .config import ROOT


def main():
    url=make_url(settings.database_url)
    if url.host not in ('localhost','127.0.0.1',None) or url.database!='kneews':
        raise ValueError('Automatic bootstrap is restricted to the local kneews database')
    args=dict(host=url.host or 'localhost',port=url.port or 5432,dbname='postgres')
    if url.username: args['user']=url.username
    if url.password: args['password']=url.password
    with psycopg.connect(**args,autocommit=True) as conn:
        if not conn.execute('SELECT 1 FROM pg_database WHERE datname=%s',('kneews',)).fetchone():
            conn.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier('kneews')))
    config=Config(str(ROOT/'alembic.ini'))
    with engine.connect() as connection:
        tables=set(inspect(connection).get_table_names())
        if 'alembic_version' not in tables and tables:
            differences=compare_metadata(MigrationContext.configure(connection),Base.metadata)
            if differences:raise RuntimeError('Unversioned database differs from the expected initial schema; migration review is required')
            # Adopt only the verified initial schema without recreating populated tables.
            command.stamp(config,'0001_initial')
    command.upgrade(config,'head')
    print('KneeWS PostgreSQL schema ready',flush=True)

if __name__=='__main__': main()
