#!/usr/bin/env python3

import os
import pathlib
import fire
import copy
import hashlib
import json
import pickle

import numpy as np

def main(train_path='./results/dump/'):

    for mdir in pathlib.Path(train_path).iterdir():
        if not mdir.is_dir(): continue

        name = mdir.name

        target = os.path.join("./results/params/plot/dist/{}/".format(name))
        os.makedirs(target, exist_ok=True)
        result_path = "./results/plot/dist/{}/".format(name)
        os.makedirs(result_path, exist_ok=True)
        log_path = "./logs/plot/dist/{}/".format(name)
        os.makedirs(log_path, exist_ok=True)

        for path in mdir.iterdir():
            param = {
                'dump_path': str(path)
            }
            ident = hashlib.sha512(json.dumps(param, sort_keys=True).encode('utf-8')).hexdigest()
            param['result_path'] = os.path.join(result_path, "{}.result".format(ident))
            param['log_path'] = os.path.join(log_path, "{}.log".format(ident))

            outpath = os.path.join(target, "{}.param".format(ident))
            with open(outpath, "wb") as f:
                pickle.dump(param, f)

if __name__ == '__main__':
    fire.Fire(main)