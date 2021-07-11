import sys
import os
import uuid
import requests
import logging
logging.basicConfig(level=logging.DEBUG)
import json

from flask import Flask, request
app = Flask(__name__)
data_type = "application/json"

def write_records(inp, out, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    columns = [ c for c in inp["columns"]]
    columns.insert(0, "prediction")
    opath = os.path.join(out_dir,str(uuid.uuid4())+".csv")
    with open(opath, "w") as f:
        f.write( ",".join(columns)+"\n")
        for pred, row in zip(out, inp["data"]):
            row.insert(0, pred)
            row = [ str(x) for x in row ]
            f.write( ",".join(row)+"\n")

def call_mlflow_model_server(data):
    headers = { "accept": data_type, "Content-Type": data_type }
    rsp = requests.post(url=mlflow_model_server_uri, data=json.dumps(data), allow_redirects=True, headers=headers)
    return json.loads(rsp.text)

@app.route("/invocations", methods = [ "POST" ])
def process():
    inp = request.json
    out = call_mlflow_model_server(inp)
    write_records(inp, out, out_dir)
    return json.dumps(out)

if __name__ == '__main__':
    mlflow_model_server_uri = sys.argv[2] 
    out_dir = sys.argv[3] 
    logging.debug(f"mlflow_model_server_uri: {mlflow_model_server_uri}")
    logging.debug(f"out_dir: {out_dir}")
    app.run(debug=True, port=int(sys.argv[1]))
