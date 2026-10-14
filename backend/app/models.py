import uuid
from datetime import datetime,timezone
from sqlalchemy import String,Text,DateTime,ForeignKey,UniqueConstraint,Integer,Float,Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped,mapped_column,relationship
from .db import Base

def now(): return datetime.now(timezone.utc)
def uid(): return str(uuid.uuid4())

class Study(Base):
    __tablename__='studies'
    id:Mapped[str]=mapped_column(String(128),primary_key=True)
    source:Mapped[str]=mapped_column(String(32),default='rsna')
    split:Mapped[str]=mapped_column(String(16),default='train')
    report:Mapped[str]=mapped_column(Text,default='')
    report_hash:Mapped[str|None]=mapped_column(String(64),index=True)
    language:Mapped[str]=mapped_column(String(32),default='unknown',index=True)
    site_group:Mapped[str]=mapped_column(String(128),default='unknown',index=True)
    group_basis:Mapped[str]=mapped_column(String(32),default='unknown')
    is_gold:Mapped[bool]=mapped_column(Boolean,default=False,index=True)
    gold:Mapped[dict]=mapped_column(JSONB,default=dict)
    metadata_json:Mapped[dict]=mapped_column(JSONB,default=dict)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)
    series:Mapped[list['Series']]=relationship(back_populates='study',cascade='all, delete-orphan')

class Series(Base):
    __tablename__='series'
    id:Mapped[str]=mapped_column(String(128),primary_key=True)
    study_id:Mapped[str]=mapped_column(ForeignKey('studies.id'),index=True)
    plane:Mapped[str]=mapped_column(String(32))
    format:Mapped[str]=mapped_column(String(16),default='dicom')
    relative_path:Mapped[str]=mapped_column(Text)
    metadata_json:Mapped[dict]=mapped_column(JSONB,default=dict)
    study:Mapped[Study]=relationship(back_populates='series')

class LabelSet(Base):
    __tablename__='label_sets'
    id:Mapped[str]=mapped_column(String(36),primary_key=True,default=uid)
    study_id:Mapped[str]=mapped_column(ForeignKey('studies.id'),index=True)
    method:Mapped[str]=mapped_column(String(64))
    version:Mapped[str]=mapped_column(String(128))
    findings:Mapped[dict]=mapped_column(JSONB)
    provenance:Mapped[dict]=mapped_column(JSONB,default=dict)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)
    __table_args__=(UniqueConstraint('study_id','method','version'),)

class Review(Base):
    __tablename__='reviews'
    id:Mapped[str]=mapped_column(String(36),primary_key=True,default=uid)
    study_id:Mapped[str]=mapped_column(ForeignKey('studies.id'),index=True)
    label_set_id:Mapped[str|None]=mapped_column(ForeignKey('label_sets.id'))
    finding:Mapped[str]=mapped_column(String(32))
    state:Mapped[str]=mapped_column(String(16))
    evidence:Mapped[str]=mapped_column(Text,default='')
    reviewer:Mapped[str]=mapped_column(String(128))
    note:Mapped[str]=mapped_column(Text,default='')
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)

class Job(Base):
    __tablename__='jobs'
    id:Mapped[str]=mapped_column(String(36),primary_key=True,default=uid)
    kind:Mapped[str]=mapped_column(String(64))
    status:Mapped[str]=mapped_column(String(16),default='queued',index=True)
    config:Mapped[dict]=mapped_column(JSONB,default=dict)
    progress:Mapped[float]=mapped_column(Float,default=0)
    message:Mapped[str]=mapped_column(Text,default='Queued')
    result:Mapped[dict]=mapped_column(JSONB,default=dict)
    error:Mapped[str|None]=mapped_column(Text)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)
    started_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True))
    finished_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True))
    heartbeat_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True))

class Experiment(Base):
    __tablename__='experiments'
    id:Mapped[str]=mapped_column(String(36),primary_key=True,default=uid)
    name:Mapped[str]=mapped_column(String(160))
    status:Mapped[str]=mapped_column(String(32),default='registered')
    config:Mapped[dict]=mapped_column(JSONB)
    manifest_hash:Mapped[str]=mapped_column(String(64))
    results:Mapped[dict]=mapped_column(JSONB,default=dict)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),default=now)
