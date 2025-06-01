# **Differential NAS for Real-Time On-Mobile Video Super-Resolution**

Code of 'Leveraging differentiable NAS and abstract genetic algorithms for optimizing on-mobile VSR performance'. DOI: `10.1007/s10994-025-06801-5`

## RuntimeEvaluationHttpServer Directory

Main code of RuntimeEvaluation HTTP server for communicating with Android clients in Python.

### Prerequisite
#### Database
`PostgreSQL 15`

#### Python Environment
```
flask 2.1
SQLAlchemy 1.4
psycopg2-binary 2.8
```
#### Code Check
- `RuntimeEvaluationHttpServer/job_recorder.py` Line 32: Check the Database link.

### Run
0. Ensure PostgreSQL Database is OK.
1. Run `app.py` to start flask app.

## AndroidRuntimeEvaluation Directory

Main code of RuntimeEvaluation Android client in Java and C++.

### Build
This is an Android Studio project.

### Run
0. Ensure PostgreSQL Database and RuntimeEvaluationHttpServer is OK.
1. Run the Android app.

## Main Directory

Main code of 'Novel Differential NAS for Real-Time On-Mobile Video Super-Resolution' in Python.

### Prerequisite
#### Database
`PostgreSQL 15`
#### Python Environment
```
TensorFlow 2.8
SQLAlchemy 1.4
psycopg2-binary 2.8
```
#### Code Check
- `init_ea.py` Line 235: Check the Database link to record the process status.
- `job_recorder.py` Line 32: Check the Database link to RuntimeEvaluation.
- `dataset.py`: Check the Dataset path.

### Run
0. Ensure PostgreSQL Database, RuntimeEvaluationHttpServer, and AndroidRuntimeEvaluation is OK.
1. Run `ea_test.py` to start the main progress.
2. Run multi-`train_worker_tf.py` to start normal training workers.
3. Run multi-`ds_worker_tf.py` to start differential search workers. This is not in conflict with `train_worker_tf.py`.
