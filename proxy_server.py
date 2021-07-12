import sys
import os
import uuid
import requests
import logging
logging.basicConfig(level=logging.DEBUG)
import json
import click

from flask import Flask, request
app = Flask(__name__)
data_type = "application/json"

_log_dir = None
_mlflow_model_server_uri = None

def write_records(inp, out, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    columns = [ c for c in inp["columns"]]
    columns.insert(0, "prediction")
    opath = os.path.join(log_dir,str(uuid.uuid4())+".csv")
    with open(opath, "w") as f:
        f.write( ",".join(columns)+"\n")
        for pred, row in zip(out, inp["data"]):
            row.insert(0, pred)
            row = [ str(x) for x in row ]
            f.write( ",".join(row)+"\n")

def call_mlflow_model_server(data):
    headers = { "accept": data_type, "Content-Type": data_type }
    rsp = requests.post(url=_mlflow_model_server_uri, data=json.dumps(data), allow_redirects=True, headers=headers)
    return json.loads(rsp.text)

@app.route("/invocations", methods = [ "POST" ])
def process():
    inp = request.json
    out = call_mlflow_model_server(inp)
    write_records(inp, out, _log_dir)
    return json.dumps(out)


@click.command()
@click.option("--port", help="Port", type=int, required=True)
@click.option("--mlflow-model-server-uri", help="MLflow model server URI", type=str, required=True)
@click.option("--log-dir", help="Log directory", default="tmp", type=str)

def main(port, mlflow_model_server_uri, log_dir):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    global _log_dir, _mlflow_model_server_uri
    _mlflow_model_server_uri = mlflow_model_server_uri
    _log_dir = log_dir
    app.run(debug=True, port=port)

if __name__ == '__main__':
    main()
