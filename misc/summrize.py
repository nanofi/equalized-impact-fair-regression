#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
import glob
import hashlib
import json

import fire

import numpy as np
import pandas as pd

def main(eval_path='./results/evals/', result_path='./results/summary/result.pd'):
    results = {}
    params = {}
    for f in glob.glob(eval_path + "**/*.result", recursive=True):
        with open(f, "rb") as fp:
            result = pickle.load(fp)
        train_param_path = result['train_param']
        with open(train_param_path, "rb") as fp:
            train_param = pickle.load(fp)
        key_param = dict([(k, v) for (k, v) in train_param.items() if not k.endswith("seed") and not k.endswith("path")])

        ident = hashlib.sha512(json.dumps(key_param, sort_keys=True).encode('utf-8')).hexdigest()
        if ident not in results:
            results[ident] = {}
            params[ident] = key_param
        for k, v in result['result'].items():
            if k not in results[ident]:
                results[ident][k] = []
            results[ident][k].append(v)

    summary = {}
    for ident, ev in results.items():
        avg = {}
        for key, values in ev.items():
            avg[key] = np.mean(values)
            avg["{}_std".format(key)] = np.std(values)
        summary[ident] = avg
    sf = pd.DataFrame.from_dict(summary, orient='index')
    pf = pd.DataFrame.from_dict(params, orient='index')

    df = pf.join(sf)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    df.to_pickle(result_path)

if __name__ == '__main__':
    fire.Fire(main)