import random
from srv_predictor import Predictor
from flask import Flask, request, jsonify, make_response
import os
import time

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
        if content['action'] == 'infer':
            in_img = content['img'].split(',')[-1]
            raw_z = content['z']
            return _corsify_actual_response(jsonify(predictor.infer(in_img, raw_z)))
        elif content['action'] == 'example':
            raw_z = content['z'] if 'z' in content else [random.gauss(0, 1) for _ in range(512)]
            return _corsify_actual_response(jsonify(predictor.example(raw_z)))
        else:
            return _corsify_actual_response(jsonify({'error': 'unknown action'}))

@app.route("/health", methods=['GET'])
def ping():
    return "OK"

@app.route("/latest_request", methods=['GET'])
def latest_request():
    return str(latest_request_ptr[0])


def _build_cors_preflight_response():
    response = make_response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
