from flask import Flask, send_file
import job_recorder

app = Flask(__name__)
app.config['FLASK_ENV'] = 'development'
engine = job_recorder.create_db_engine()


@app.route('/get_a_lite/<job>')
def getALite(job):
    path = "./temp_lite_save/" + job + ".tflite"
    return send_file(path, as_attachment=True, attachment_filename=path)


@app.route('/get_a_job/<chip>')
def getAJob(chip):
    # find old job
    job = job_recorder.find_undo_job(engine, chip)
    if job is not None:
        return "HAS:" + job[0], 200
    # find a model
    # convert this model, if not re-find another
    # give this lite model
    return "NONE:JOB", 200


@app.route('/done_a_job/<chip>/<job>/<status>/<time>')
def doneAJob(chip, job, status, time):
    print(chip, job, status, time)
    # deal with erri and erro
    if status == "ERRO":  # err that may recover
        pass
    job_recorder.commit_a_job(engine, chip, job, status, time)
    # if not OK, request from client repeats.
    return "OK", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
