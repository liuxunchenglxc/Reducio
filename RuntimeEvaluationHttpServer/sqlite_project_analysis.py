import sqlite_project_fetcher
import job_recorder
from sqlalchemy.orm import Session

engine = job_recorder.create_db_engine()


def check_all_model(project_root1, project_root2, project_root3):
    with Session(engine) as session:
        res = session.query(job_recorder.JobSave.job, job_recorder.JobCommit.time) \
            .join(job_recorder.JobCommit, job_recorder.JobSave.pkey == job_recorder.JobCommit.pkey).all()
    print("hash", "psnr", "time", "score", sep="\t")
    for hash, time in res:
        sql = "select sr.psnr from sr inner join hash on sr.id=hash.id " \
              "where hash.hash=\'" + hash + "\' order by sr.psnr desc limit 1"
        rows = sqlite_project_fetcher.select_db_all_no_params(project_root1, sql)
        if len(rows) == 0:
            sql = "select sr.psnr from sr inner join hash on sr.id=hash.id " \
                  "where hash.hash=\'" + hash + "\' order by sr.psnr desc limit 1"
            rows = sqlite_project_fetcher.select_db_all_no_params(project_root2, sql)
            if len(rows) == 0:
                sql = "select sr.psnr from sr inner join hash on sr.id=hash.id " \
                      "where hash.hash=\'" + hash + "\' order by sr.psnr desc limit 1"
                rows = sqlite_project_fetcher.select_db_all_no_params(project_root3, sql)
                row = rows[0]
            else:
                row = rows[0]
        else:
            row = rows[0]

        psnr = row[0]
        print(hash, psnr, time, (2**(2*(psnr-26)))/time, sep="\t")

