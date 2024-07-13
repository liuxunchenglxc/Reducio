import sqlite3
import time

from tf_lite_converter import lite_convert
import os
import job_recorder

engine = job_recorder.create_db_engine()


def connect_db(project_root):
    con = sqlite3.connect(project_root + "/EA.db")
    cur = con.cursor()
    # create table main
    sql = "CREATE TABLE IF NOT EXISTS main(id INTEGER PRIMARY KEY, gene TEXT)"
    cur.execute(sql)
    # create table score
    sql = "CREATE TABLE IF NOT EXISTS score(id INTEGER PRIMARY KEY, score REAL)"
    cur.execute(sql)
    # create table runtime
    sql = "CREATE TABLE IF NOT EXISTS runtime(id INTEGER PRIMARY KEY, runtime REAL)"
    cur.execute(sql)
    # create table sr
    sql = "CREATE TABLE IF NOT EXISTS sr(id INTEGER PRIMARY KEY, psnr REAL, ssim REAL)"
    cur.execute(sql)
    # create table generation
    sql = "CREATE TABLE IF NOT EXISTS generation(id INTEGER PRIMARY KEY, father INTEGER, mother INTEGER, iteration INTEGER)"
    cur.execute(sql)
    # create table length
    sql = "CREATE TABLE IF NOT EXISTS length(id INTEGER PRIMARY KEY, length INTEGER, gene_code_length INTEGER)"
    cur.execute(sql)
    # create table number
    sql = "CREATE TABLE IF NOT EXISTS number(id INTEGER PRIMARY KEY, number INTEGER, avg_length INTEGER)"
    cur.execute(sql)
    # create table code
    sql = "CREATE TABLE IF NOT EXISTS code(id INTEGER PRIMARY KEY, code TEXT)"
    cur.execute(sql)
    # create table status
    sql = "CREATE TABLE IF NOT EXISTS status(id INTEGER PRIMARY KEY, status INTEGER, iteration INTEGER)"
    cur.execute(sql)
    # create table code_file
    sql = "CREATE TABLE IF NOT EXISTS code_file(id INTEGER PRIMARY KEY, file TEXT, ckpt_dir TEXT, save_dir TEXT, log_dir TEXT)"
    cur.execute(sql)
    # create table res_file
    sql = "CREATE TABLE IF NOT EXISTS res_file(id INTEGER PRIMARY KEY, save_name TEXT, save_dir TEXT)"
    cur.execute(sql)
    # create table bad_code
    sql = "CREATE TABLE IF NOT EXISTS bad_code(id INTEGER PRIMARY KEY, train_log TEXT)"
    cur.execute(sql)
    # create table gene_expressed
    sql = "CREATE TABLE IF NOT EXISTS gene_expressed(id INTEGER PRIMARY KEY, gene_expressed TEXT)"
    cur.execute(sql)
    # create table hash
    sql = "CREATE TABLE IF NOT EXISTS hash(id INTEGER PRIMARY KEY, hash TEXT)"
    cur.execute(sql)
    con.commit()
    return con, cur


def close_db(con, cur):
    con.commit()
    cur.close()
    con.close()


def select_db_all_no_params(project_root, sql: str):
    con, cur = connect_db(project_root)
    ok = 0
    while ok == 0:
        try:
            cur.execute(sql)
            con.commit()
            ok = 1
        except sqlite3.OperationalError:
            ok = 0
            time.sleep(10)
    rows = cur.fetchall()
    close_db(con, cur)
    return rows


def find_all_model(project_root):
    sql = "select sr.id, hash.hash from sr inner join hash on sr.id=hash.id " \
          "where sr.id in (select min(id) from hash group by hash)"
    rows = select_db_all_no_params(project_root, sql)
    for row in rows:
        try:
            id = row[0]
            hash = row[1]
            saved_model_dir = project_root + "/save/" + str(id) + "/"
            saved_model_dir += os.listdir(project_root + "/save/" + str(id))[0]
            tflite_model_save_dir = "./temp_lite_save"
            model_name = hash
            if not os.path.exists(tflite_model_save_dir + "/" + model_name + ".tflite"):
                lite_convert(saved_model_dir, tflite_model_save_dir, model_name)
                job_recorder.save_a_job(engine, "Qualcomm Snapdragon 865", hash)
                print(hash)
        except:
            pass

