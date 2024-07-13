import argparse
import os
import subprocess
import time

from sqlalchemy.orm import Session

import init_ea

parser = argparse.ArgumentParser(description="A worker of differentiable searching for ea. "
                                             "This phrase will process after all training finished. "
                                             "The name and gpu can be the same as train wokers.")
parser.add_argument('-n', '--name', type=str, help="Unique name of worker.")
parser.add_argument('-g', '--gpu', type=int, help="Index of GPU.")
args = parser.parse_args()


def get_ds_task_info(engine, init_id: int):
    init_id = str(init_id)
    ckpt_dir = './ds_ckpt/' + init_id
    save_dir = './ds_save/' + init_id
    log_dir = './ds_logs/' + init_id
    return ckpt_dir, save_dir, log_dir


def try_score_info(engine, ds_describe: str):
    with Session(engine) as session:
        row = session.execute("select * from ds_result where ds_describe like '{}'".format(ds_describe)).first()
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
        session.commit()
        if row is None:
            row = session.execute("select count(*) from worker_register").first()
            session.commit()
            count = row[0]
            print("New worker id: ", count, " Name: ", lucky_string)
            session.execute("insert into worker_register values ({}, '{}')".format(count, lucky_string))
            session.commit()
            return count
        else:
            session.commit()
            return row[0]


def find_task(engine, worker_id: int):
        while True:
            with Session(engine) as session:
                # find available task
                row = session.execute("select ds_describe from ds_record where ds_describe not in (select ds_describe from ds_task_get)").first()
                session.commit()
            if row is None:
                time.sleep(60)
                continue
            # try to get task
            ds_describe = row[0]
            try:
                with Session(engine) as session:
                    session.execute("insert into ds_task_get values ('{}', {})".format(ds_describe, worker_id))
                    session.commit()
                    time.sleep(10)
                    row = session.execute("select ds_describe from ds_task_get "
                                          "where ds_describe like '{}' and worker={}".format(ds_describe, worker_id)).first()
                    session.commit()
                if row is None:
                    continue
                else:
                    return ds_describe
            except:
                time.sleep(10)
                continue


def check_task_continue(engine, worker_id: int):
    # find all task of worker and check it finish or not
    with Session(engine) as session:
        row = session.execute("select ds_describe from ds_task_get where worker={} and "
                              "ds_describe not in (select ds_describe from ds_task_end)".format(worker_id)).first()
        session.commit()
    if row is None:
        return ""
    else:
        return row[0]


def finish_task(engine, ds_describe: str):
    with Session(engine) as session:
        row = session.execute("select ds_describe from ds_task_end where ds_describe like '{}'".format(ds_describe)).first()
        session.commit()
        if row is None:
            session.execute("insert into ds_task_end values ('{}')".format(ds_describe))
        session.commit()


def log_bad_code(engine, ds_describe: str):
    with Session(engine) as session:
        session.execute("insert into ds_bad_code values ('{}', '{}')"
                        .format(ds_describe, './ds_train_outputs/{}.out'.format(ds_describe)))
        session.commit()


def do_task(engine, ds_describe: str, gpu: int, lucky_string: str):
    with Session(engine) as session:
        row = session.execute("select * from ds_record where ds_describe like '{}'".format(ds_describe)).first()
        session.commit()
        iteration = row[1]
        init_id = row[2]
        init_gene = row[3]
        init_score = row[4]
        session.commit()
    ckpt_dir, save_dir, log_dir = get_ds_task_info(engine, init_id)
    epochs = 200
    if try_score_info(engine, ds_describe) == 200:
        print("task ", ds_describe, " is done, and submitting.")
        finish_task(engine, ds_describe)
        print("task ", ds_describe, " is done, and submitted.")
        return
    train_script = '''import ds_tf
from init_ea import create_db_engine
from sqlalchemy.orm import Session
import tensorflow as tf

tf.random.set_seed(666666)

iteration = {}
ds_describe = "{}"
init_id = {}
init_gene = "{}"
init_score = {}
ckpt_dir = "{}"
save_dir = "{}"
log_dir = "{}"
new_id, new_score = ds_tf.search({}, iteration, ds_describe, init_id, init_gene, 
                                 init_score, ckpt_dir=ckpt_dir, save_dir=save_dir,
                                 log_dir=log_dir, epochs_ds={}, epochs_re=200, batch_size=16)
engine = create_db_engine()
with Session(engine) as session:
    session.execute("insert into ds_result values ('"+str(ds_describe)+"', "+str(new_id)+", "+str(new_score)+")")
    session.commit()
engine.dispose()
'''.format(iteration, ds_describe, init_id, init_gene, init_score, ckpt_dir, save_dir, log_dir, gpu, epochs)
    script_path = './ds_script_tf_gen_' + lucky_string + '.py'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(train_script)
    if not os.path.exists("./ds_train_outputs/"):
        os.mkdir("./ds_train_outputs/")
    with open('./ds_train_outputs/{}.out'.format(ds_describe), 'a', encoding='utf-8') as f:
        return_code = subprocess.call(['python', '-u', script_path], stdout=f, stderr=f)
    if return_code != 0:
        print("ERROR: DS process of ds_describe={} is failed, return code={}, please check output log at ./ds_train_outputs/{}.out".format(
            ds_describe, return_code, ds_describe))
        log_bad_code(engine, ds_describe)
    print("task ", ds_describe, " is done, and submitting.")
    finish_task(engine, ds_describe)
    print("task ", ds_describe, " is done, and submitted.")


def workflow(lucky_string: str, gpu: int):
    engine = init_ea.create_db_engine()
    # register worker
    worker_id = worker_register(engine, lucky_string)
    while True:
        # check non-finish task
        ds_describe = check_task_continue(engine, worker_id)
        while ds_describe != "":
            print("find undone task ", ds_describe)
            do_task(engine, ds_describe, gpu, lucky_string)
            ds_describe = check_task_continue(engine, worker_id)
        # find task and do it
        ds_describe = find_task(engine, worker_id)
        do_task(engine, ds_describe, gpu, lucky_string)


def main():
    workflow(args.name, args.gpu)


if __name__ == '__main__':
    main()
