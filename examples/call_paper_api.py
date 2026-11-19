"""Command-line client using only the Python standard library. Prints the response."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def main():
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://127.0.0.1:8765')
    parser.add_argument('--file',type=Path,help='CSV/TSV of examination records; omit for the built-in example')
    args=parser.parse_args()
    headers={}
    token=os.environ.get('CHRONIC_API_TOKEN')
    if token:headers['Authorization']='Bearer '+token
    if args.file is None:
        path='/api/v2/hypertension/predict'
        body=json.dumps({'records':[{'male':1,'age':50,'currentSmoker':0,'cigsPerDay':0,
              'BPMeds':0,'diabetes':0,'totChol':200,'sysBP':130,'diaBP':80,'BMI':25,
              'heartRate':70,'glucose':90}]}).encode()
        headers['Content-Type']='application/json'
    else:
        path='/api/v2/hypertension/predict-file'
        boundary='----PaperAPI'+uuid.uuid4().hex
        parts=[]
        # ASCII upload name; only the file content matters for inference.
        filename='input'+args.file.suffix.lower()
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: text/plain\r\n\r\n'.encode())
        parts.extend([args.file.read_bytes(),f'\r\n--{boundary}--\r\n'.encode()])
        body=b''.join(parts)
        headers['Content-Type']='multipart/form-data; boundary='+boundary
    request=urllib.request.Request(args.base.rstrip('/')+path,data=body,headers=headers,method='POST')
    try:
        with urllib.request.urlopen(request,timeout=120) as response:
            result=json.load(response)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except urllib.error.HTTPError as exc:
        print(f'HTTP {exc.code}: {exc.read().decode("utf-8",errors="replace")}')
        raise SystemExit(1)


if __name__=='__main__':main()
