import hashlib
import os
import shutil
import time
from importlib import import_module
from typing import List

import tensorflow as tf
from sqlalchemy.orm import Session
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D

import dataset
from ea_code_tf import gene_to_code
from graph_seq import gene_graph_seq_with_info
from init_ea import Code, CodeFile, GeneExpressed, Hash, Length, Main, ResFile, Runtime, create_db_engine, Score, SR, \
    Generation
from runtime_evaluation import request_evaluation_by_gene_list, look_evaluation_by_hash_list
from score import score_sr
from train_tf import train

tf.random.set_seed(666666)


class DSCell(Model):
    @staticmethod
    def get_f_a_list():
        f_list = [1, 3, 5]
        a_list = [None, "relu", "sigmoid"]
        return f_list, a_list

    def __init__(self, channel_num: int, init_weight: List[float], time_diff: List[float]):
        super(DSCell, self).__init__()
        self.weight = tf.Variable(init_weight, trainable=True)
        f_list, a_list = DSCell.get_f_a_list()
        self.convs = []
        self.time_diff = time_diff
        for f in f_list:
            for a in a_list:
                self.convs.append(Conv2D(channel_num, f, padding="same", activation=a))

    def call(self, x):
        weight = tf.nn.softmax(self.weight)
        y = self.convs[0](x) * weight[0]
        for i in range(1, len(self.convs)):
            y += self.convs[i](x) * weight[i]
        return y

    def get_time_diff_sum(self):
        weight = tf.nn.softmax(self.weight)
        y = self.time_diff[0] * weight[0]
        for i in range(1, len(self.time_diff)):
            y += self.time_diff[i] * weight[i]
        return y

    def get_weight_list(self):
        weight = tf.nn.softmax(self.weight)
        return list(weight.numpy())

    def get_weight_penalty(self):
        weight = tf.nn.softmax(self.weight)
        return tf.math.reduce_sum(weight * weight)


class DSModel(Model):
    def __init__(self, ds_describe: str, init_gene: str, weight_a: float = 0.445):
        super(DSModel, self).__init__()
        self.ds_describe_items = [[int(i) for i in item.split(",")] for item in ds_describe.split("-")[:-1]]
        self.init_gene_params = [[int(i) for i in item.split(",")] for item in init_gene.split("-")[:-1]]
        self.cells = {}
        f_list, a_list = DSCell.get_f_a_list()
        lf = len(f_list)
        la = len(a_list)
        n = lf * la
        gene_list = [init_gene]
        for i in range(1, len(self.init_gene_params)):
            param = self.init_gene_params[i]
            if param[0] == 1:
                pre = "-".join([",".join([str(i) for i in p]) for p in self.init_gene_params[:i]])
                las = "-".join([",".join([str(i) for i in p]) for p in self.init_gene_params[i + 1:]] + ["255"])
                mod = [i for i in param]
                for f in f_list:
                    for a in range(len(a_list)):
                        mod[4] = f
                        mod[6] = a
                        mods = ",".join([str(i) for i in mod])
                        gene = "-".join([pre, mods, las])
                        gene_list.append(gene)
        hash_list = request_evaluation_by_gene_list(gene_list)
        k = True
        while k:
            res = look_evaluation_by_hash_list(hash_list)
            k = False
            for r in res:
                if r[0] is None:
                    k = True
                    time.sleep(60)
                    break
        res = look_evaluation_by_hash_list(hash_list)
        self.time_base = res[0][0]
        mat = max([i[0] for i in res])
        mit = min([i[0] for i in res])
        print(res[0][0], mat, mit, (mat - mit) / res[0][0])
        time_pos = 1
        for i in range(1, len(self.ds_describe_items)):
            item = self.ds_describe_items[i]
            param = self.init_gene_params[i]
            if item[0] == 1:
                p_index = f_list.index(param[4]) * lf + param[6]
                weight_a = max(weight_a, 0)
                weight_a = min(weight_a, 1)
                init_weight = [(1 - weight_a) / n for i in range(n)]
                init_weight[p_index] = 1 / n + (n - 1) / n * weight_a
                time_diff = [r[0] - self.time_base for r in res[time_pos:time_pos + n]]
                time_pos += n
                self.cells[str(i)] = DSCell(item[3], init_weight, time_diff)

    def call(self, x):
        mid = {0: x}
        for i in range(1, len(self.ds_describe_items)):
            item = self.ds_describe_items[i]
            if item[0] == 1:
                offset = item[1]
                in_pos = max(i - offset, 0)
                mid[i] = self.cells[str(i)](mid[in_pos])
            else:
                offset = (item[1], item[2])
                in_pos = [max(i - off, 0) for off in offset]
                if item[3] == 2:
                    mid[i] = mid[in_pos[0]] + mid[in_pos[1]]
                else:
                    mid[i] = mid[in_pos[0]] * mid[in_pos[1]]
        y = tf.nn.depth_to_space(mid[len(self.ds_describe_items) - 1], 4)
        return y

    def estimate_time(self):
        time = self.time_base + sum([cell.get_time_diff_sum() for cell in self.cells.values()])
        return time

    def get_result_gene(self):
        f_list, a_list = DSCell.get_f_a_list()
        lf = len(f_list)
        la = len(a_list)
        params = ["0"]
        for i in range(1, len(self.init_gene_params)):
            param = self.init_gene_params[i]
            mod = [i for i in param]
            if param[0] == 1:
                cell = self.cells[str(i)]
                weight = cell.get_weight_list()
                wi = weight.index(max(weight))
                f = f_list[wi // lf]
                a = wi % la
                mod[4] = f
                mod[6] = a
                mods = ",".join([str(i) for i in mod])
                params.append(mods)
            else:
                mods = ",".join([str(i) for i in mod])
                params.append(mods)
        params.append("255")
        return "-".join(params)

    def get_cell_penalty(self):
        return tf.math.reduce_sum([cell.get_weight_penalty() for cell in self.cells.values()]) / len(
            self.cells.values())

    def print_weight_dist(self):
        for k, cell in self.cells.items():
            print(cell.get_weight_list())

    def get_weight_slight(self):
        res = []
        for k, cell in self.cells.items():
            w = cell.get_weight_list()
            m = max(w)
            res.append(w.index(m))
        return ",".join([str(i) for i in res])


def search(gpu_idx, iteration, ds_describe, init_id, init_gene, init_score, ckpt_dir, save_dir,
           log_dir, epochs_ds, epochs_re, batch_size=4):
    if gpu_idx >= 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
    ds_ckpt_dir = ckpt_dir + '/ds'
    ds_save_dir = save_dir + '/ds'
    ds_log_dir = log_dir + '/ds'
    final_gene = ds_train(ds_describe, init_gene, ds_ckpt_dir, ds_save_dir, ds_log_dir, epochs_ds, batch_size)
    if final_gene == init_gene:
        return -1, -1
    # Start scoring it
    hash_list = request_evaluation_by_gene_list([final_gene])
    re_ckpt_dir = ckpt_dir + '/re'
    re_save_dir = save_dir + '/re'
    re_log_dir = log_dir + '/re'
    code, gene_length = gene_to_code([item.split(",") for item in final_gene.split("-")])
    file_head = '''import tensorflow as tf

from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model'''
    code_f = file_head + "\n\n\n" + code
    os.makedirs("./ds_model", exist_ok=True)
    with open("./ds_model/model_{}.py".format(init_id), "w") as f:
        f.write(code_f)
    model_class = getattr(import_module('ds_model.model_{}'.format(init_id).format()), "GEN_SR_MODEL")
    final_model_dirname, best_psnr, the_ssim = train(0, model_class, re_ckpt_dir, re_save_dir, re_log_dir, epochs_re,
                                                     4, False)
    k = True
    while k:
        res = look_evaluation_by_hash_list(hash_list)
        k = False
        for r in res:
            if r[0] is None:
                k = True
                time.sleep(60)
                break
    res = look_evaluation_by_hash_list(hash_list)
    runtime = res[0][0]
    if runtime == 999.9:
        return -1, -2
    new_score = score_sr(best_psnr, the_ssim, runtime)
    print(new_score)
    if new_score <= init_score:
        return -1, -3
    # Add final gene to current generation
    engine = create_db_engine()
    with Session(engine) as session:
        row = session.execute("select * from main order by id desc limit 1").first()
        if row is None:
            new_id = 1
        else:
            new_id = row[0] + 1
        row = Main(id=new_id, gene=final_gene)
        session.merge(row)
        row = Score(id=new_id, score=new_score)
        session.merge(row)
        row = Runtime(id=new_id, runtime=runtime)
        session.merge(row)
        row = SR(id=new_id, psnr=best_psnr, ssim=the_ssim)
        session.merge(row)
        row = Generation(id=new_id, father=init_id, mother=init_id, iteration=iteration)
        session.merge(row)
        row = Length(id=new_id, length=gene_length, gene_code_length=gene_length)
        session.merge(row)
        row = Code(id=new_id, code=code)
        session.merge(row)
        row = CodeFile(id=new_id, file="./ds_model/model_{}.py".format(init_id), ckpt_dir=ckpt_dir, save_dir=save_dir,
                       log_dir=log_dir)
        session.merge(row)
        row = ResFile(id=new_id, save_name=final_model_dirname, save_dir=re_save_dir)
        session.merge(row)
        row = GeneExpressed(id=new_id, gene_expressed=final_gene)
        session.merge(row)
        gene_units = final_gene.split("-")[1:-1]
        seq, info = gene_graph_seq_with_info(gene_units)
        info_s = '-'.join(info)
        hash = hashlib.sha1((seq + '|' + info_s).encode(encoding='utf-8')).hexdigest()
        row = Hash(id=new_id, hash=hash, seq_info=seq + '|' + info_s)
        session.merge(row)
        session.commit()
    engine.dispose()
    return new_id, new_score


def ds_train(ds_describe, init_gene, ckpt_dir, save_dir, log_dir, epochs, batch_size=4):
    train_log_dir = log_dir + '/train'
    val_log_dir = log_dir + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    with tf.device('/device:GPU:0'):
        model: tf.keras.Model = DSModel(ds_describe, init_gene)
    loss_mse = tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    loss_mae = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(1e-3 * 6, 200, 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_psnr = tf.keras.metrics.Mean(name='train_psnr')
    train_ssim = tf.keras.metrics.Mean(name='train_ssim')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_psnr = tf.keras.metrics.Mean(name='val_psnr')
    val_ssim = tf.keras.metrics.Mean(name='val_ssim')
    train_ds, train_ds_len = dataset.get_reds_train_dset()
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=2)
    val_ds, val_ds_len = dataset.get_reds_val_dset()
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=2)
    train_ds_cache = [i for i in iter(train_ds)]
    val_ds_cache = [i for i in iter(val_ds)]

    @tf.function
    def train_step(images, labels):
        with tf.device('/device:GPU:0'):
            with tf.GradientTape() as tape:
                predictions = model(images)
                loss_l1 = loss_mae(labels, predictions)
                loss_l2 = loss_mse(labels, predictions)
                time = model.estimate_time()
                loss_psnr = tf.math.xlogy(10.0, 65025.0 / loss_l2) / tf.math.log(10.0)
                loss_score = - tf.math.pow(2.0, 2.0 * (loss_psnr - 26.0)) / time
                loss_final = loss_score * 10 + loss_l1
                gradients = tape.gradient(loss_final, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss_final)
                train_psnr(tf.image.psnr(labels, predictions, max_val=255))  # for int8
                train_ssim(tf.image.ssim(labels, predictions, max_val=255))  # for int8

    @tf.function
    def val_step(images, labels):
        with tf.device('/device:GPU:0'):
            predictions = model(images)
            loss_l1 = loss_mae(labels, predictions)
            loss_l2 = loss_mse(labels, predictions)
            time = model.estimate_time()
            loss_psnr = tf.math.xlogy(10.0, 65025.0 / loss_l2) / tf.math.log(10.0)
            loss_score = - tf.math.pow(2.0, 2.0 * (loss_psnr - 26.0)) / time
            loss_final = loss_score * 10 + loss_l1

            val_loss(loss_final)
            val_psnr(tf.image.psnr(labels, predictions, max_val=255))  # for int8
            val_ssim(tf.image.ssim(labels, predictions, max_val=255))  # for int8

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer=optimizer, model=model, best_psnr=tf.Variable(0.0),
                               the_ssim=tf.Variable(0.0))
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    start_path = ckpt_dir + "/start"
    if not os.path.exists(start_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(start_path, 'w', encoding='utf-8') as f:
            f.write("0")

    w_slight = []

    for epoch in range(epochs - ckpt.epoch):
        train_loss.reset_states()
        train_psnr.reset_states()
        val_loss.reset_states()
        val_psnr.reset_states()

        for images, labels in train_ds_cache:
            train_step(images, labels)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('psnr', train_psnr.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('ssim', train_ssim.result(), step=tf.cast(ckpt.epoch, tf.int64))

        for val_images, val_labels in val_ds_cache:
            val_step(val_images, val_labels)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('psnr', val_psnr.result(), step=tf.cast(ckpt.epoch, tf.int64))
            tf.summary.scalar('ssim', val_ssim.result(), step=tf.cast(ckpt.epoch, tf.int64))

        template = 'Epoch {}, Loss: {}, PSNR: {}, SSIM: {}, Val Loss: {}, Val PSNR: {}, Val SSIM: {}'
        if val_psnr.result() > ckpt.best_psnr:
            try:
                os.removedirs(save_dir + "/best_temp")
            except BaseException:
                pass
            model.save(save_dir + "/best_temp")
            ckpt.best_psnr.assign(val_psnr.result())
            ckpt.the_ssim.assign(val_ssim.result())
        ckpt.epoch.assign_add(1)
        manager.save()
        print(template.format(int(ckpt.epoch),
                              train_loss.result(),
                              train_psnr.result(),
                              train_ssim.result(),
                              val_loss.result(),
                              val_psnr.result(),
                              val_ssim.result()))
        w_slight.append(model.get_weight_slight())
    final_model_dirname = "best_" + '{}'.format(tf.cast(ckpt.best_psnr * 1e6, tf.int32))
    final_model_path = save_dir + "/" + final_model_dirname
    try:
        shutil.move(save_dir + "/best_temp", final_model_path)
    except BaseException:
        pass
    final_gene = model.get_result_gene()
    for i in range(len(w_slight)):
        print(i, w_slight[i])
    print("final", model.get_weight_slight())
    return final_gene
