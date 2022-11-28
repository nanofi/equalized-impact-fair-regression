#!/usr/bin/env python3

import os
import glob
import pickle
import subprocess

import fire

def main(path):
    for f in glob.glob(path + "**/*.param", recursive=True):
        name = os.path.splitext(os.path.basename(f))[0]
        with open(f, "rb") as fp:
            param = pickle.load(fp)

        submit_str = "./misc/test.py {}".format(f)
        try:
            stdout = subprocess.check_output(submit_str, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.returncode)
            print(e.output)
            raise
        print(stdout, submit_str)

if __name__ == '__main__':
    fire.Fire(main)