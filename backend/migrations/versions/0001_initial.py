"""Initial PostgreSQL research schema, frozen from the first implementation."""
from alembic import op
revision='0001_initial'
down_revision=None
branch_labels=None
depends_on=None

def upgrade():
    op.execute('\nCREATE TABLE experiments (\n\tid VARCHAR(36) NOT NULL, \n\tname VARCHAR(160) NOT NULL, \n\tstatus VARCHAR(32) NOT NULL, \n\tconfig JSONB NOT NULL, \n\tmanifest_hash VARCHAR(64) NOT NULL, \n\tresults JSONB NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)\n\n')
    op.execute('\nCREATE TABLE jobs (\n\tid VARCHAR(36) NOT NULL, \n\tkind VARCHAR(64) NOT NULL, \n\tstatus VARCHAR(16) NOT NULL, \n\tconfig JSONB NOT NULL, \n\tprogress FLOAT NOT NULL, \n\tmessage TEXT NOT NULL, \n\tresult JSONB NOT NULL, \n\terror TEXT, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tstarted_at TIMESTAMP WITH TIME ZONE, \n\tfinished_at TIMESTAMP WITH TIME ZONE, \n\theartbeat_at TIMESTAMP WITH TIME ZONE, \n\tPRIMARY KEY (id)\n)\n\n')
    op.execute('CREATE INDEX ix_jobs_status ON jobs (status)')
    op.execute('\nCREATE TABLE studies (\n\tid VARCHAR(128) NOT NULL, \n\tsource VARCHAR(32) NOT NULL, \n\tsplit VARCHAR(16) NOT NULL, \n\treport TEXT NOT NULL, \n\treport_hash VARCHAR(64), \n\tlanguage VARCHAR(32) NOT NULL, \n\tsite_group VARCHAR(128) NOT NULL, \n\tgroup_basis VARCHAR(32) NOT NULL, \n\tis_gold BOOLEAN NOT NULL, \n\tgold JSONB NOT NULL, \n\tmetadata_json JSONB NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id)\n)\n\n')
    op.execute('CREATE INDEX ix_studies_language ON studies (language)')
    op.execute('CREATE INDEX ix_studies_site_group ON studies (site_group)')
    op.execute('CREATE INDEX ix_studies_is_gold ON studies (is_gold)')
    op.execute('CREATE INDEX ix_studies_report_hash ON studies (report_hash)')
    op.execute('\nCREATE TABLE label_sets (\n\tid VARCHAR(36) NOT NULL, \n\tstudy_id VARCHAR(128) NOT NULL, \n\tmethod VARCHAR(64) NOT NULL, \n\tversion VARCHAR(128) NOT NULL, \n\tfindings JSONB NOT NULL, \n\tprovenance JSONB NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tUNIQUE (study_id, method, version), \n\tFOREIGN KEY(study_id) REFERENCES studies (id)\n)\n\n')
    op.execute('CREATE INDEX ix_label_sets_study_id ON label_sets (study_id)')
    op.execute('\nCREATE TABLE series (\n\tid VARCHAR(128) NOT NULL, \n\tstudy_id VARCHAR(128) NOT NULL, \n\tplane VARCHAR(32) NOT NULL, \n\tformat VARCHAR(16) NOT NULL, \n\trelative_path TEXT NOT NULL, \n\tmetadata_json JSONB NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(study_id) REFERENCES studies (id)\n)\n\n')
    op.execute('CREATE INDEX ix_series_study_id ON series (study_id)')
    op.execute('\nCREATE TABLE reviews (\n\tid VARCHAR(36) NOT NULL, \n\tstudy_id VARCHAR(128) NOT NULL, \n\tlabel_set_id VARCHAR(36), \n\tfinding VARCHAR(32) NOT NULL, \n\tstate VARCHAR(16) NOT NULL, \n\tevidence TEXT NOT NULL, \n\treviewer VARCHAR(128) NOT NULL, \n\tnote TEXT NOT NULL, \n\tcreated_at TIMESTAMP WITH TIME ZONE NOT NULL, \n\tPRIMARY KEY (id), \n\tFOREIGN KEY(study_id) REFERENCES studies (id), \n\tFOREIGN KEY(label_set_id) REFERENCES label_sets (id)\n)\n\n')
    op.execute('CREATE INDEX ix_reviews_study_id ON reviews (study_id)')

def downgrade():
    op.drop_table('reviews')
    op.drop_table('series')
    op.drop_table('label_sets')
    op.drop_table('studies')
    op.drop_table('jobs')
    op.drop_table('experiments')
