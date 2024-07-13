import hashlib
import os
import pathlib
import random
import time
from importlib import import_module
from typing import List

import tensorflow as tf
from sqlalchemy.orm import Session

import job_recorder
from ea_code_tf import gene_to_code
from graph_seq import gene_graph_seq_with_info
from init_ea import create_db_engine


def get_hash_by_id(id):
    engine = create_db_engine()
    with Session(engine) as session:
        row = session.execute("select hash from hash where id={}".format(id)).first()
        if row is not None:
            return row[0]
        else:
            return None


def lite_convert(model, tflite_model_save_dir, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path(tflite_model_save_dir)
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    model_filename = model_name + ".tflite"

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_file = tflite_models_dir / model_filename
    tflite_model_file.write_bytes(tflite_model)


def request_evaluation(id_list, chip_list=None):
    if chip_list is None:
        chip_list = ["Qualcomm Snapdragon 865"]
    input = tf.ones([1, 180, 320, 3])
    for id in id_list:
        tflite_model_save_dir = "../RuntimeEvaluationHttpServer/temp_lite_save"
        hash = get_hash_by_id(id)
        if not os.path.exists(tflite_model_save_dir + "/" + hash + ".tflite"):
            model_class = getattr(import_module('models.model_{}'.format(id)), "GEN_SR_MODEL")
            model = model_class()
            model(input)
            lite_convert(model, tflite_model_save_dir, hash)
        for chip in chip_list:
            job_recorder.save_a_job(job_recorder.create_db_engine(), chip, hash)


# For DS
def request_evaluation_by_gene_list(gene_list: List[str], chip_list=None):
    if chip_list is None:
        chip_list = ["Qualcomm Snapdragon 865"]
    input = tf.ones([1, 180, 320, 3])
    hash_list = []
    for gene in gene_list:
        tflite_model_save_dir = "../RuntimeEvaluationHttpServer/temp_lite_save"
        gene_units = gene.split("-")[1:-1]
        seq, info = gene_graph_seq_with_info(gene_units)
        info_s = '-'.join(info)
        hash = hashlib.sha1((seq + '|' + info_s).encode(encoding='utf-8')).hexdigest()
        hash_list.append(hash)
        if not os.path.exists(tflite_model_save_dir + "/" + hash + ".tflite"):
            code, _ = gene_to_code([item.split(",") for item in gene.split("-")])
            file_head = '''import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model'''
            code = file_head + "\n\n\n" + code
            os.makedirs("te_model_temp", exist_ok=True)
            random.seed(time.time())
            fname = str(int(time.time())) + "-" + str(random.randint(0, 999999999))
            with open("te_model_temp/temp_{}.py".format(fname), "w") as f:
                f.write(code)
            model_class = getattr(import_module('te_model_temp.temp_{}'.format(fname)), "GEN_SR_MODEL")
            model = model_class()
            model(input)
            lite_convert(model, tflite_model_save_dir, hash)
            try:
                os.remove("te_model_temp/temp_{}.py".format(fname))
            except:
                pass
        for chip in chip_list:
            job_recorder.save_a_job(job_recorder.create_db_engine(), chip, hash)
    return hash_list


def look_evaluation(id_list, chip_list=None):
    if chip_list is None:
        chip_list = ["Qualcomm Snapdragon 865"]
    with Session(job_recorder.create_db_engine()) as session:
        res = []
        for id in id_list:
            re = []
            hash = get_hash_by_id(id)
            for chip in chip_list:
                time = None
                row = session.execute(
                    "select pkey from job_save where chip='{}' and job='{}'".format(chip, hash)).first()
                if row is not None:
                    pkey = row[0]
                    row = session.execute("select * from job_commit where pkey='{}'".format(pkey)).first()
                    if row is not None:
                        status = row[1]
                        if status == "OK":
                            time = row[2]
                        else:
                            time = -1
                re.append(time)
            res.append(re)
        session.commit()
    return res


# For DS
def look_evaluation_by_hash_list(hash_list, chip_list=None):
    if chip_list is None:
        chip_list = ["Qualcomm Snapdragon 865"]
    with Session(job_recorder.create_db_engine()) as session:
        res = []
        for hash in hash_list:
            re = []
            for chip in chip_list:
                time = None
                row = session.execute(
                    "select pkey from job_save where chip='{}' and job='{}'".format(chip, hash)).first()
                if row is not None:
                    pkey = row[0]
                    row = session.execute("select * from job_commit where pkey='{}'".format(pkey)).first()
                    if row is not None:
                        status = row[1]
                        if status == "OK":
                            time = row[2]
                        else:
                            time = 999.9
                re.append(time)
            res.append(re)
        session.commit()
    return res
