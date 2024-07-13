from sqlalchemy import Column
from sqlalchemy import String, Float
from sqlalchemy.orm import registry, Session
from sqlalchemy import create_engine
from hashlib import sha1

mapper_registry = registry()
Base = mapper_registry.generate_base()


class JobSave(Base):
    __tablename__ = "job_save"
    pkey = Column(String(41), primary_key=True)
    chip = Column(String, nullable=False)
    job = Column(String, nullable=False)

    def __repr__(self):
        return f"JobSave(pkey={self.pkey!r}, chip={self.chip!r}, job={self.job!r})"


class JobCommit(Base):
    __tablename__ = "job_commit"
    pkey = Column(String(41), primary_key=True)
    status = Column(String, nullable=False)
    time = Column(Float, nullable=False)

    def __repr__(self):
        return f"JobCommit(pkey={self.pkey!r}, status={self.status!r}, time={self.time!r})"


def create_db_engine():
    engine = create_engine("postgresql://username:password@ipaddr:port/dbname")
    mapper_registry.metadata.create_all(engine)
    return engine


def save_a_job(engine, chip: str, job: str):
    with Session(engine) as session:
        pkey = sha1((chip + job).encode(encoding="utf-8")).hexdigest()
        if session.query(JobSave).where(JobSave.pkey == pkey).first() is None:
            job_row = JobSave(pkey=pkey, chip=chip, job=job)
            session.add(job_row)
            session.commit()
            return True
        else:
            return False


def commit_a_job(engine, chip: str, job: str, status: str, time: str):
    with Session(engine) as session:
        pkey = sha1((chip + job).encode(encoding="utf-8")).hexdigest()
        if session.query(JobCommit).where(JobCommit.pkey == pkey).first() is None:
            job_row = JobCommit(pkey=pkey, status=status, time=float(time))
            session.add(job_row)
            session.commit()
            return True
        else:
            return False


def find_undo_job(engine, chip: str):
    with Session(engine) as session:
        q = session.query(JobCommit.pkey)
        job = session.query(JobSave.job).where(JobSave.chip == chip).where(~JobSave.pkey.in_(q)).first()
        return job

