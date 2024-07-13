from sqlalchemy.orm import registry, Session
from sqlalchemy import Column, String, Float, Integer
from sqlalchemy import create_engine

mapper_registry = registry()
Base = mapper_registry.generate_base()


class Main(Base):
    __tablename__ = "main"
    id = Column(Integer, primary_key=True)
    gene = Column(String, nullable=False)

    def __repr__(self):
        return f"Main(id={self.id!r}, gene={self.gene!r})"


class Score(Base):
    __tablename__ = "score"
    id = Column(Integer, primary_key=True)
    score = Column(Float, nullable=False)

    def __repr__(self):
        return f"Score(id={self.id!r}, score={self.score!r})"


class Runtime(Base):
    __tablename__ = "runtime"
    id = Column(Integer, primary_key=True)
    runtime = Column(Float, nullable=False)

    def __repr__(self):
        return f"Runtime(id={self.id!r}, runtime={self.runtime!r})"


class SR(Base):
    __tablename__ = "sr"
    id = Column(Integer, primary_key=True)
    psnr = Column(Float, nullable=False)
    ssim = Column(Float, nullable=False)

    def __repr__(self):
        return f"SR(id={self.id!r}, psnr={self.psnr!r}, ssim={self.ssim!r})"


class Generation(Base):
    __tablename__ = "generation"
    id = Column(Integer, primary_key=True)
    father = Column(Integer, nullable=False)
    mother = Column(Integer, nullable=False)
    iteration = Column(Integer, nullable=False)

    def __repr__(self):
        return f"Generation(id={self.id!r}, father={self.father!r}, mother={self.mother!r}, iteration={self.iteration!r})"


class Length(Base):
    __tablename__ = "length"
    id = Column(Integer, primary_key=True)
    length = Column(Integer, nullable=False)
    gene_code_length = Column(Integer, nullable=False)

    def __repr__(self):
        return f"Length(id={self.id!r}, length={self.length!r}, gene_code_length={self.gene_code_length!r})"


class Number(Base):
    __tablename__ = "number"
    id = Column(Integer, primary_key=True)
    number = Column(Integer, nullable=False)
    avg_length = Column(Integer, nullable=False)

    def __repr__(self):
        return f"Number(id={self.id!r}, number={self.number!r}, avg_length={self.avg_length!r})"


class Code(Base):
    __tablename__ = "code"
    id = Column(Integer, primary_key=True)
    code = Column(String, nullable=False)

    def __repr__(self):
        return f"Code(id={self.id!r}, code={self.code!r})"


class Status(Base):
    __tablename__ = "status"
    id = Column(Integer, primary_key=True)
    status = Column(Integer, nullable=False)
    iteration = Column(Integer, nullable=False)

    def __repr__(self):
        return f"Status(id={self.id!r}, status={self.status!r}, iteration={self.iteration!r})"


class CodeFile(Base):
    __tablename__ = "code_file"
    id = Column(Integer, primary_key=True)
    file = Column(String, nullable=False)
    ckpt_dir = Column(String, nullable=False)
    save_dir = Column(String, nullable=False)
    log_dir = Column(String, nullable=False)

    def __repr__(self):
        return f"CodeFile(id={self.id!r}, file={self.file!r}, ckpt_dir={self.ckpt_dir!r}, " \
               f"save_dir={self.save_dir!r}, log_dir={self.log_dir!r})"


class ResFile(Base):
    __tablename__ = "res_file"
    id = Column(Integer, primary_key=True)
    save_name = Column(String, nullable=False)
    save_dir = Column(String, nullable=False)

    def __repr__(self):
        return f"ResFile(id={self.id!r}, save_name={self.save_name!r}, save_dir={self.save_dir!r})"


class BadCode(Base):
    __tablename__ = "bad_code"
    id = Column(Integer, primary_key=True)
    train_log = Column(String, nullable=False)

    def __repr__(self):
        return f"BadCode(id={self.id!r}, train_log={self.train_log!r})"


class GeneExpressed(Base):
    __tablename__ = "gene_expressed"
    id = Column(Integer, primary_key=True)
    gene_expressed = Column(String, nullable=False)

    def __repr__(self):
        return f"GeneExpressed(id={self.id!r}, gene_expressed={self.gene_expressed!r})"


class Hash(Base):
    __tablename__ = "hash"
    id = Column(Integer, primary_key=True)
    hash = Column(String, nullable=False)
    seq_info = Column(String, nullable=False)

    def __repr__(self):
        return f"Hash(id={self.id!r}, hash={self.hash!r})"


class Task(Base):
    __tablename__ = "task"
    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return f"Task(id={self.id!r})"


class TaskGet(Base):
    __tablename__ = "task_get"
    id = Column(Integer, primary_key=True)
    worker = Column(Integer, nullable=False)

    def __repr__(self):
        return f"TaskGet(id={self.id!r}, worker={self.worker!r})"


class WorkerRegister(Base):
    __tablename__ = "worker_register"
    worker_id = Column(Integer, primary_key=True)
    lucky_string = Column(String, nullable=False)

    def __repr__(self):
        return f"WorkerRegister(worker_id={self.worker_id!r}, lucky_string={self.lucky_string!r})"


class TaskEnd(Base):
    __tablename__ = "task_end"
    id = Column(Integer, primary_key=True)

    def __repr__(self):
        return f"TaskEnd(id={self.id!r})"


# Differential search related tables
class DSRecord(Base):
    __tablename__ = "ds_record"
    ds_describe = Column(String, primary_key=True)
    iteration = Column(Integer, nullable=False)
    init_id = Column(Integer, nullable=False)
    init_gene = Column(String, nullable=False)
    init_score = Column(Float, nullable=False)
    cover_ids = Column(String, nullable=False)

    def __repr__(self):
        return f"DSRecord(ds_describe={self.ds_describe!r}, iteration={self.iteration!r}, " \
               f"init_id={self.init_id!r}, init_gene={self.init_gene!r}, init_score={self.init_score!r}," \
               f" cover_ids={self.cover_ids!r})"


class DSTaskGet(Base):
    __tablename__ = "ds_task_get"
    ds_describe = Column(String, primary_key=True)
    worker = Column(Integer, nullable=False)

    def __repr__(self):
        return f"DSTaskGet(ds_describe={self.ds_describe!r}, worker={self.worker!r})"


class DSTaskEnd(Base):
    __tablename__ = "ds_task_end"
    ds_describe = Column(String, primary_key=True)

    def __repr__(self):
        return f"DSTaskEnd(ds_describe={self.ds_describe!r})"


class DSBadCode(Base):
    __tablename__ = "ds_bad_code"
    ds_describe = Column(String, primary_key=True)
    train_log = Column(String, nullable=False)

    def __repr__(self):
        return f"DSBadCode(ds_describe={self.ds_describe!r}, train_log={self.train_log!r})"


class DSResult(Base):
    __tablename__ = "ds_result"
    ds_describe = Column(String, primary_key=True)
    new_id = Column(Integer, nullable=False)
    new_score = Column(Float, nullable=False)

    def __repr__(self):
        return f"DSResult(ds_describe={self.ds_describe!r}, new_id={self.new_id!r}, " \
               f"new_score={self.new_score!r})"


def create_db_engine():
    engine = create_engine("postgresql://username:password@ipaddr:port/dbname",
                           pool_recycle=60,
                           pool_pre_ping=True,
                           pool_use_lifo=True,
                           echo_pool=True,
                           pool_size=2)
    mapper_registry.metadata.create_all(engine)
    return engine


def read_status(engine):
    with Session(engine) as session:
        row = session.query(Status).order_by(Status.id.desc()).first()
        if row is None:
            genes = [
                '0-1,1,1,17,5,13,0-1,1,1,4,3,1,1-1,1,1,28,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-2,1,4,3-1,3,1,28,5,1,1-2,4,7,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,3,1,1-1,2,1,28,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,26,5,13,2-1,2,1,16,1,4,0-1,1,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,3,1,1-1,2,1,15,3,1,1-1,1,1,4,3,1,1-1,4,1,28,3,1,0-1,3,1,16,3,1,1-1,1,1,28,3,1,2-1,1,1,4,3,1,1-1,3,1,28,5,1,1-2,1,5,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,3,1,0-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,16,3,1,1-1,2,1,22,1,1,1-1,2,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,3,1,1-1,2,1,15,3,1,1-1,1,1,4,3,1,1-1,4,1,28,3,1,0-1,5,1,16,3,1,1-1,1,1,28,3,1,2-1,1,1,4,3,1,1-1,3,1,28,5,1,1-2,1,5,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,3,1,1-1,2,1,9,5,15,0-1,2,1,28,3,1,1-1,3,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-2,6,1,2-1,1,1,26,5,13,2-1,2,1,16,1,4,0-1,1,1,12,3,1,1-2,2,9,2-255',
                '0-1,1,1,22,3,1,1-2,1,2,2-1,1,1,28,3,1,1-1,1,1,26,5,13,2-2,1,2,3-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,3,19,1-1,2,1,15,3,1,1-1,1,1,4,3,1,1-1,4,1,28,3,1,0-1,3,1,16,3,1,1-1,1,1,28,3,1,2-1,1,1,4,3,1,1-1,3,1,28,5,1,1-2,1,5,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,3,1,1-1,2,1,28,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,5,1,31,3,12,1-1,1,1,28,3,1,1-1,1,1,26,5,13,2-2,1,2,3-1,1,1,12,3,1,1-2,2,9,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,16,3,1,1-1,2,1,28,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-2,1,4,3-1,2,1,16,1,4,0-1,1,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,1,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-2,1,4,2-1,3,1,22,1,1,1-1,1,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,3,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,16,3,1,1-1,2,1,22,1,1,1-1,1,1,12,3,1,1-2,2,8,2-1,1,1,48,3,1,0-255',
                '0-1,1,1,22,1,1,1-1,2,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-1,1,1,28,3,1,1-2,1,4,2-1,3,1,22,1,1,1-1,1,1,12,3,1,1-2,2,8,2-255',
                '0-1,1,1,22,3,1,1-1,2,1,28,3,1,1-2,1,2,2-1,1,1,28,3,1,1-1,3,1,28,3,1,1-1,1,1,26,5,13,2-2,1,2,3-2,1,5,2-1,1,1,48,3,1,0-255']
            i = 1
            for gene in genes:
                r = Main(id=i, gene=gene)
                g = Generation(id=i, father=0, mother=0, iteration=0)
                session.add_all([r, g])
                i += 1
            s = Status(id=0, status=0, iteration=0)
            session.add(s)
            session.commit()
            status_dict = {
                'status': 0,
                'iteration': 0
            }
            return status_dict
        else:
            status_dict = {
                'status': row.status,
                'iteration': row.iteration,
            }
            session.commit()
            return status_dict


if __name__ == '__main__':
    engine = create_db_engine()
    print(read_status(engine))
