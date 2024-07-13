import hashlib
import argparse
import os
import subprocess
import init_ea
import shutil
from sqlalchemy.orm import registry, Session
from sqlalchemy import Column, String, Float, Integer

parser = argparse.ArgumentParser(description="Second train for ea.")
parser.add_argument('-n', '--name', type=str, help="Unique name of train.")
parser.add_argument('-g', '--gpu', type=int, help="Index of GPU.")
parser.add_argument('-i', '--id', type=int, help="ID of model.")
parser.add_argument('-l', '--lr', type=float, help="Learning rate.")
parser.add_argument('-b', '--bs', type=int, help="Batch size.")
parser.add_argument('-e', '--epochs', type=int, help="Epochs.")
parser.add_argument('-p', '--prefix', type=str, required=False, help="manual path prefix")
parser.add_argument('-a', '--hash', type=str, required=False, help="manul path hash")
parser.add_argument('-m', '--manual', type=int, required=False, default=0, help="is manual path")
parser.add_argument('--ds', type=int, default=0, help="is ds path")
args = parser.parse_args()

mapper_registry = registry()
Base = mapper_registry.generate_base()


class SecondMain(Base):
    __tablename__ = "second_main"
    hash = Column(String, primary_key=True)
    id = Column(Integer, nullable=False)
    lr = Column(Float, nullable=False)
    bs = Column(Integer, nullable=False)
    epochs = Column(Integer, nullable=False)
    ckpt_dir = Column(String, primary_key=True)
    save_dir = Column(String, primary_key=True)
    log_dir = Column(String, primary_key=True)

    def __repr__(self):
        return f"SecondMain(hash={self.hash!r}, id={self.id!r}, lr={self.lr!r}, bs={self.bs!r}" \
               f"epochs={self.epochs!r}, ckpt_dir={self.ckpt_dir!r}, save_dir={self.save_dir!r}, " \
               f"log_dir={self.log_dir!r})"


class SecondScore(Base):
    __tablename__ = "second_score"
    hash = Column(String, primary_key=True)
    psnr = Column(Float, nullable=False)
    ssim = Column(Float, nullable=False)
    runtime = Column(Float, nullable=False)
    score = Column(Float, nullable=False)

    def __repr__(self):
        return f"SecondScore(hash={self.hash!r}, psnr={self.psnr!r}, ssim={self.ssim!r}, runtime={self.runtime!r}" \
               f"score={self.score!r})"


def connect_db():
    engine = init_ea.create_db_engine()
    mapper_registry.metadata.create_all(engine)
    return engine


def get_real_id_and_dir(engine, id: int):
    with Session(engine) as session:
        row = session.execute("select * from code_file "
                              "where id=(select id from code where code like "
                              "(select code from code where id={}) limit 1)".format(id)).first()
    return row[0], row[2], row[3], row[4]


def get_train_task_info(engine, hash: str):
    with Session(engine) as session:
        row = session.execute("select * from second_main where hash='{}'".format(hash)).first()
    ckpt_dir = row[5]
    save_dir = row[6]
    log_dir = row[7]
    return ckpt_dir, save_dir, log_dir


def try_score_info(engine, hash: str):
    with Session(engine) as session:
        row = session.execute("select * from second_score where hash='{}'".format(hash)).first()
    if row is None:
        return 404
    else:
        return 200


def do_task(engine, hash: str, id: int, lr: float, bs: int, epochs: int, gpu: int, lucky_string: str, model_dir: str):
    ckpt_dir, save_dir, log_dir = get_train_task_info(engine, hash)
    if try_score_info(engine, hash) == 200:
        print("Second train ", hash, " is done.")
        return
    train_script = '''import train_tf
from init_ea import create_db_engine
from sqlalchemy.orm import Session
import score
import {}.model_{} as model_gen
import tensorflow as tf

tf.random.set_seed(666666)


def insert_db(sql: str):
    engine = create_db_engine()
    with Session(engine) as session:
        session.execute(sql)
        session.commit()


hash = "{}"
ckpt_dir = "{}"
save_dir = "{}"
log_dir = "{}"
final_model_dirname, best_psnr, the_ssim = train_tf.train_second(gpu_idx={}, model_class=model_gen.GEN_SR_MODEL,
                                                                 model_ckpt_dir=ckpt_dir, model_save_dir=save_dir,
                                                                 log_dir=log_dir, epochs={}, batch_size={}, lr={})
runtime_time = 123
model_score = score.score_sr(best_psnr, the_ssim, runtime_time)
insert_db("insert into second_score values ('"+hash+"', "+best_psnr+", "+the_ssim+", "+runtime_time+", "+model_score+")")
    '''.format(model_dir, id, hash, ckpt_dir, save_dir, log_dir, gpu, epochs, bs, lr)
    script_path = './second_train_script_tf_gen_' + lucky_string + '.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(train_script)
    if not os.path.exists("./second_train_outputs/"):
        os.mkdir("./second_train_outputs/")
    with open('./second_train_outputs/{}.out'.format(hash), 'a', encoding='utf-8') as f:
        return_code = subprocess.call(['python', '-u', script_path], stdout=f, stderr=f)
    if return_code != 0:
        print(
            "ERROR: Train process of id={} model is failed, return code={}, please check output log at ./second_train_outputs/{}.out".format(
                id, return_code, hash))
    else:
        print("Second train ", hash, " is done.")


def main():
    name = args.name
    gpu = args.gpu
    lr = args.lr
    bs = args.bs
    epochs = args.epochs
    engine = connect_db()
    model_dir = "models"
    if args.manual <= 0:
        if args.ds <= 0:
            id, o_ckpt_dir, o_save_dir, o_log_dir = get_real_id_and_dir(engine, args.id)
            hash = hashlib.sha1(
                (str(id) + '|' + str(lr) + '|' + str(bs) + '|' + str(epochs)).encode(encoding='utf-8')).hexdigest()
            print(id, lr, bs, epochs, hash)
        else:
            model_dir = "ds_model"
            o_ckpt_dir = "./ds_ckpt/{}/re/".format(args.id)
            o_save_dir = "./ds_save/{}/re/".format(args.id)
            o_log_dir = "./ds_logs/{}/re/".format(args.id)
            id = args.id
            hash = hashlib.sha1(
                (str(id) + '|ds|' + str(lr) + '|' + str(bs) + '|' + str(epochs)).encode(encoding='utf-8')).hexdigest()
    else:
        o_ckpt_dir = "./{}_ckpt/{}/".format(args.prefix, args.hash)
        o_save_dir = "./{}_save/{}/".format(args.prefix, args.hash)
        o_log_dir = "./{}_logs/{}/".format(args.prefix, args.hash)
        hash = hashlib.sha1(
            (str(args.hash) + '|' + str(lr) + '|' + str(bs) + '|' + str(epochs)).encode(encoding='utf-8')).hexdigest()
        print(args.hash, lr, bs, epochs, hash)
        id = args.id
    with Session(engine) as session:
        row = session.execute("select * from second_main where hash='{}'".format(hash)).first()
        session.commit()
    if row is None:
        os.makedirs("./second_ckpt/", exist_ok=True)
        os.makedirs("./second_save/", exist_ok=True)
        os.makedirs("./second_logs/", exist_ok=True)
        ckpt_dir = "./second_ckpt/{}/".format(hash)
        save_dir = "./second_save/{}/".format(hash)
        log_dir = "./second_logs/{}/".format(hash)
        shutil.copytree(o_ckpt_dir, ckpt_dir, dirs_exist_ok=True)
        shutil.copytree(o_save_dir, save_dir, dirs_exist_ok=True)
        shutil.copytree(o_log_dir, log_dir, dirs_exist_ok=True)
        with Session(engine) as session:
            session.execute("insert into second_main values ('{}', {}, {}, {}, {}, '{}', '{}', '{}')"
                            .format(hash, id, lr, bs, epochs, ckpt_dir, save_dir, log_dir))
            session.commit()
    do_task(engine, hash, id, lr, bs, epochs, gpu, name, model_dir)


if __name__ == '__main__':
    main()
