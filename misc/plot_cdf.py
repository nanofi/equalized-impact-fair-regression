#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pickle
import importlib
import glob
import hashlib
import json
import math
import fire
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#mpl.rc('text', usetex=True)


lims = {
    'comm_crime': (-2, 2),
    'nlsy79': (-3, 3)
}

methods = ["hams", "jcn", "ot_ei"]

tradeoff = {
    'hams': ("kappa", False),
    'lin-hams': ("kappa", False),
    'jcn': ("lambda", True),
    'ot_ei': ("eta", True),
}

method_names = {
    "hams": "EOpair",
    "lin-hams": "EOpair (linear)",
    "jcn": "EO$\chi^2$",
    "unconstrained": "LS",
    "ot_ei": "EI"
}

def plot_cdf(df, lim, srow=3):
    g = df.groupby("name")
    
    ncol = len(methods)
    nrow = int(math.ceil(g["eps"].nunique().max() / srow))

    fig, axes = plt.subplots(nrow, ncol, sharex=True, sharey=True, figsize=(7, 6))

    legend_data = {}

    pad = 5
    for ax, col in zip(axes[0], methods):
        method = method_names[col]
        ax.annotate(method, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for col, method in enumerate(methods):
        dfg = g.get_group(method)
        key, asd = tradeoff[method]
        for irow, (eps, dfi) in enumerate(dfg.groupby(key)):
            if irow % srow != 0:
                continue
            irow = irow // srow
            row = irow if asd else nrow - irow - 1

            ax = axes[row, col]

            if dfi["prob"].isna().all():
                sns.ecdfplot(data=dfi, x='upsilon', hue='s', ax=ax, legend=True, palette="deep")
                ax.set_xlabel(r"$\Upsilon$")
                ax.set_ylabel("CDF")
                ax.set_xlim(lim)
            else:
                sns.ecdfplot(data=dfi, x='upsilon', hue='s', weights='prob', ax=ax, legend=True, palette="deep")
                ax.set_xlabel(r"$\Upsilon$")
                ax.set_ylabel("CDF")
                ax.set_xlim(lim)
            
            ax.text(0.01, 0.99, "$\\{}$={:.2E}".format(key, eps),
                horizontalalignment='left',
                verticalalignment='top',
                size = "small",
                transform = ax.transAxes)

            data = {}
            if ax.legend_ is not None:
                handles = ax.legend_.legendHandles
                labels = [t.get_text() for t in ax.legend_.texts]
                data.update({l: h for h, l in zip(handles, labels)})
            handles, labels = ax.get_legend_handles_labels()
            data.update({l: h for h, l in zip(handles, labels)})
            legend_data.update(data)
            ax.get_legend().remove()
    
    label_order = list(legend_data.keys())
    blank_handle = mpl.patches.Patch(alpha=0, linewidth=0)
    handles = [legend_data.get(l, blank_handle) for l in label_order]
    labels = label_order
    #fig.legend(handles, labels, title="groups", loc = 'center right')        
    
    return fig

def main(dump_path='./results/dump/', result_dir='./results/plot/dist/', skip_row=3):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    results = {}
    params = {}
    for f in glob.glob(dump_path + "**/*.result", recursive=True):
        with open(f, "rb") as fp:
            result = pickle.load(fp)
        train_param_path = result['train_param']
        with open(train_param_path, "rb") as fp:
            train_param = pickle.load(fp)
        
        key_param = {}
        for k, v in train_param.items():
            if k.endswith("seed"):
                key_param[k] = v
        key_param['dataset'] = train_param['dataset']

        ident = hashlib.sha512(json.dumps(key_param, sort_keys=True).encode('utf-8')).hexdigest()
        if ident not in results:
            results[ident] = []
            params[ident] = key_param
        results[ident].append({'result': result, 'train_param': train_param})
    
    for ident, ps in results.items():
        key_param = params[ident]
        dataset_path = key_param['dataset']
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        dataset_seed = key_param['dataset_seed']

        dfs = []
        for p in ps:
            dumps = p['result']['dumps']
            prob = p['result']['prob']
            train_param = p['train_param']

            if len(prob.shape) > 0:
                for i, df in enumerate(dumps):
                    df["prob"] = 1.0 if prob.shape == () else prob[i]
                df = pd.concat(dumps)
            else:
                df = dumps[0]

            df["name"] = train_param['name']
            df["Method"] = method_names[train_param['name']]
            df["eps"] = train_param['eps'] if 'eps' in train_param else None
            df["upsilon"] = df["y"] - df["z"]

            if train_param["name"] in ["hams", "lin-hams"]:
                df["kappa"] = train_param["eps"]
            elif train_param["name"] == "jcn":
                eps = train_param["eps"]
                df["lambda"] = (1 - eps) / eps
            elif train_param["name"] == "ot_ei":
                df["eta"] = 1 - train_param["eps"]
            else:
                df["eps"] = None
            
            dfs.append(df)
        
        df = pd.concat(dfs)

        lim = lims[dataset_name]

        fig = plot_cdf(df, lim, srow=skip_row)
        fig.savefig(os.path.join(result_dir, "dist_{}_{}.pdf".format(dataset_name, dataset_seed)), bbox_inches="tight")
        plt.clf()

if __name__ == '__main__':
    fire.Fire(main)