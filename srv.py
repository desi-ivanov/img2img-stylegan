from srv_predictor import Predictor
from flask import Flask, request, jsonify, make_response
import threading
import time
import os
EXIT_SECONDS_AFTER_INACTIVITY = 60 * 5

predictor = Predictor(
    os.environ.get('SG_SEG_DATASET_PATH'),
    os.environ.get('SG_REAL_DATASET_PATH'),
    os.environ.get('SG_MODEL_PATH'),
    os.environ.get('SG_SEG_COLORS_JSON'),
    os.environ.get('SG_CLF_PATH'),
)

app = Flask(__name__)

latest_request_ptr = [time.time()]

@app.route("/infer", methods=['POST', 'OPTIONS'])
def infer():
    latest_request_ptr[0] = time.time()
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        content = request.json
        in_img = content['img'].split(',')[-1]
        raw_z = content['z']
        return _corsify_actual_response(jsonify(predictor.infer(in_img, raw_z)))


@app.route("/example", methods=['POST', 'OPTIONS'])
def example():
    latest_request_ptr[0] = time.time()
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        content = request.json
        raw_z = content['z']
        return _corsify_actual_response(jsonify(predictor.example(raw_z)))

@app.route("/health", methods=['GET'])
def health():
    latest_request_ptr[0] = time.time()
    return "OK"

def _build_cors_preflight_response():
    response = make_response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def keep_checking_if_inactive():
    print(f'CHECKING {time.time() - latest_request_ptr[0]}s since last request')
    if time.time() - latest_request_ptr[0] > EXIT_SECONDS_AFTER_INACTIVITY:
        os._exit(0)
    t = threading.Timer(10, keep_checking_if_inactive)
    t.start()

keep_checking_if_inactive()