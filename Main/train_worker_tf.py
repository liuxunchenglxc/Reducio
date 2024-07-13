import time
import argparse
import os
import subprocess

import init_ea
from sqlalchemy.orm import Session

parser = argparse.ArgumentParser(description="A worker of training for ea.")
parser.add_argument('-n', '--name', type=str, help="Unique name of worker.")
parser.add_argument('-g', '--gpu', type=int, help="Index of GPU.")
args = parser.parse_args()


def get_train_task_info(engine, id: int):
    with Session(engine) as session:
        row = session.execute("select * from code_file where id={}".format(id)).first()
        session.commit()
    ckpt_dir = row[2]
    save_dir = row[3]
    log_dir = row[4]
    return ckpt_dir, save_dir, log_dir


def try_score_info(engine, id: int):
    with Session(engine) as session:
        row = session.execute("select * from score where id={}".format(id)).first()
        session.commit()
    if row is None:
        return 404
    else:
        return 200


def worker_register(engine, lucky_string: str):
    # check lucky string
    if lucky_string.count('%') > 0 or lucky_string.count('_') > 0:
        return -1
    # give or get an id
    with Session(engine) as session:
        row = session.execute("select worker_id from worker_register "
                              "where lucky_string like '{}'".format(lucky_string)).first()
        if row is None:
            row = session.execute("select count(*) from worker_register").first()
            count = row[0]
            print("New worker id: ", count, " Name: ", lucky_string)
            session.execute("insert into worker_register values ({}, '{}')".format(count, lucky_string))
            session.commit()
            return count
        else:
            session.commit()
            return row[0]


def find_task(engine, worker_id: int):
    with Session(engine) as session:
        while True:
            try:
                # find available task
                row = session.execute("select id from task where id not in (select id from task_get)").first()
                if row is None:
                    time.sleep(60)
                    continue
                # try to get task
                id = row[0]
                session.execute("insert into task_get values ({}, {})".format(id, worker_id))
                session.commit()
                time.sleep(10)
                row = session.execute("select id from task_get where id={} and worker={}".format(id, worker_id)).first()
                if row is None:
                    continue
                else:
                    return id
            except:
                time.sleep(10)
                continue
        session.commit()


def check_task_continue(engine, worker_id: int):
    # find all task of worker and check it finish or not
    with Session(engine) as session:
        row = session.execute("select id from task_get where worker={} and "
                              "id not in (select id from task_end)".format(worker_id)).first()
        session.commit()
    if row is None:
        return -1
    else:
        return row[0]


def finish_task(engine, id: int):
    with Session(engine) as session:
        row = session.execute("select id from task_end where id={}".format(id)).first()
        if row is None:
            session.execute("insert into task_end values ({})".format(id))
        session.commit()


def log_bad_code(engine, id: int):
    with Session(engine) as session:
        session.execute("insert into bad_code values ({}, '{}')".format(id, './train_outputs/{}.out'.format(id)))
        session.commit()


def do_task(engine, id: int, gpu: int, lucky_string: str):
    ckpt_dir, save_dir, log_dir = get_train_task_info(engine, id)
    epochs = 200
    if try_score_info(engine, id) == 200:
        print("task ", id, " is done, and submitting.")
        finish_task(engine, id)
        print("task ", id, " is done, and submitted.")
        return
    train_script = '''import train_tf
from init_ea import create_db_engine
from sqlalchemy.orm import Session
import score
import models.model_{} as model_gen
import tensorflow as tf

tf.random.set_seed(666666)

id = {}
ckpt_dir = "{}"
save_dir = "{}"
log_dir = "{}"
final_model_dirname, best_psnr, the_ssim = train_tf.train(gpu_idx={}, model_class=model_gen.GEN_SR_MODEL,
                                                                 model_ckpt_dir=ckpt_dir, model_save_dir=save_dir,
                                                                 log_dir=log_dir, epochs={}, batch_size=4)
engine = create_db_engine()
with Session(engine) as session:
    session.execute("insert into res_file values ("+str(id)+", '"+str(final_model_dirname)+"', '"+str(save_dir)+"')")
    session.execute("insert into sr values ("+str(id)+", "+str(best_psnr)+", "+str(the_ssim)+")")
    session.commit()
engine.dispose()
'''.format(id, id, ckpt_dir, save_dir, log_dir, gpu, epochs)
    script_path = './train_script_tf_gen_' + lucky_string + '.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(train_script)
    if not os.path.exists("./train_outputs/"):
        os.mkdir("./train_outputs/")
    with open('./train_outputs/{}.out'.format(id), 'a', encoding='utf-8') as f:
        return_code = subprocess.call(['python', '-u', script_path], stdout=f, stderr=f)
    if return_code != 0:
        print("ERROR: Train process of id={} model is failed, return code={}, please check output log at ./train_outputs/{}.out".format(
            id, return_code, id))
        log_bad_code(engine, id)
    print("task ", id, " is done, and submitting.")
    finish_task(engine, id)
    print("task ", id, " is done, and submitted.")


def workflow(lucky_string: str, gpu: int):
    engine = init_ea.create_db_engine()
    # register worker
    worker_id = worker_register(engine, lucky_string)
    while True:
        # check non-finish task
        id = check_task_continue(engine, worker_id)
        while id != -1:
            print("find undone task ", id)
            do_task(engine, id, gpu, lucky_string)
            id = check_task_continue(engine, worker_id)
        # find task and do it
        id = find_task(engine, worker_id)
        do_task(engine, id, gpu, lucky_string)


def main():
    workflow(args.name, args.gpu)


if __name__ == '__main__':
    main()
