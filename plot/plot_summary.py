#!/usr/bin/env python3
import os
from functools import partial
import fire

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks")

def main():
    df = pd.read_pickle('result.pd')
    datasets = [
        "../data/comm_crime/comm_crime.pd",
        "../data/nlsy79/nlsy79.pd"
    ]

    df["kappa"] = df["eps"]
    df["lambda"] = (1 - df["eps"]) / df["eps"]
    df["eta"] = 1 - df["eps"]
    method_names = {
        "hams": "EOpair",
        "lin-hams": "EOpair (linear)",
        "jcn": r"EO$\chi^2$",
        "unconstrained": "LS",
        "ot_ei": "EI"
    }
    def make_method_name(x):
        return method_names[x["name"]]
    df["Method"] = df.apply(make_method_name, axis=1)

    method_order = ["LS", "EOpair", r"EO$\chi^2$", "EI"]
    


    for dataset in datasets:
        for name in ["lin-hams", "hams"]:
            di = df[(df["dataset"] == dataset) & (df["name"] == name)].sort_values(["kappa"])

            di.plot(x="kappa", y="unfair_odds", yerr="unfair_odds_std", legend=False)
            plt.xlabel("$\kappa$")
            plt.ylabel("Unfair Pairwise")
            plt.xscale("log")

            dataname = os.path.splitext(os.path.basename(dataset))[0]
            plt.savefig("{}-{}-vs-kappa.pdf".format(dataname, name), bbox_inches="tight")
            plt.clf()
    
    for dataset in datasets:
        name = "jcn"
        di = df[(df["dataset"] == dataset) & (df["name"] == name)].sort_values(["lambda"])

        di.plot(x="lambda", y="unfair_chi2", yerr="unfair_chi2_std", legend=False)
        plt.xlabel("$\lambda$")
        plt.ylabel("Unfair Chi2")
        plt.xscale("log")

        dataname = os.path.splitext(os.path.basename(dataset))[0]
        plt.savefig("{}-{}-vs-lambda.pdf".format(dataname, name), bbox_inches="tight")
        plt.clf()
    
    for dataset in datasets:
        name = "ot_ei"
        di = df[(df["dataset"] == dataset) & (df["name"] == name)].sort_values(["eta"])

        di.plot(x="eta", y="unfair_impact", yerr="unfair_impact_std", legend=False, figsize=(4,3))
        plt.xlabel("$\eta$")
        plt.ylabel("Unfair Impact")

        dataname = os.path.splitext(os.path.basename(dataset))[0]
        plt.savefig("{}-{}-vs-eta.pdf".format(dataname, name), bbox_inches="tight")
        plt.clf()

    df = df[df["name"] != "lin-hams"]

    def td_plot(*args, **kwargs):
        data = kwargs.pop("data").sort_values(["eps"], ascending=False)
        kwargs["alpha"] = 0.8
        x = args[0]
        y = args[1]
        if len(data) > 1:
            plt.plot(data[x][:1], data[y][:1], marker=">", **kwargs)
            plt.plot(data[x][-1:], data[y][-1:], marker="<", **kwargs)
            sns.lineplot(data=data, x=x, y=y, marker="o", markevery=slice(1,-1), sort=False, **kwargs)
        else:
            sns.scatterplot(data=data, x=x, y=y, marker="o", zorder=3, **kwargs)

    for dataset in datasets:
        di = df[df["dataset"] == dataset]

        g = sns.FacetGrid(di, hue="Method", legend_out=False, height=3, aspect=4/3, hue_order=method_order)
        g.map_dataframe(td_plot, "unfair_odds", "mse")
        g.add_legend()
        plt.xlabel("Unfair odds")
        plt.ylabel("MSE")
        plt.xscale("log")
        plt.yscale("log")

        dataname = os.path.splitext(os.path.basename(dataset))[0]
        plt.savefig("{}-odds-mse.pdf".format(dataname), bbox_inches="tight")
        plt.clf()
    for dataset in datasets:
        di = df[df["dataset"] == dataset]

        g = sns.FacetGrid(di, hue="Method", legend_out=False, height=3, aspect=4/3, hue_order=method_order)
        g.map_dataframe(td_plot, "unfair_chi2", "mse")
        g.add_legend()
        plt.xlabel("Unfair chi2")
        plt.ylabel("MSE")
        plt.xscale("log")
        plt.yscale("log")

        dataname = os.path.splitext(os.path.basename(dataset))[0]
        plt.savefig("{}-chi2-mse.pdf".format(dataname), bbox_inches="tight")
        plt.clf()

    for dataset in datasets:
        di = df[df["dataset"] == dataset]

        g = sns.FacetGrid(di, hue="Method", legend_out=False, height=3, aspect=4/3, hue_order=method_order)
        g.map_dataframe(td_plot, "unfair_impact", "mse")
        g.add_legend()
        plt.xlabel("Unfair Impact")
        plt.ylabel("MSE")

        dataname = os.path.splitext(os.path.basename(dataset))[0]
        plt.savefig("{}-impact-mse.pdf".format(dataname), bbox_inches="tight")
        plt.clf()

    for dataset in datasets:
        di = df[(df["dataset"] == dataset) & ((df["name"] == "unconstrained") | (df["name"] == "ot_ei"))].sort_values(["eps"])

        g = sns.FacetGrid(di, hue="Method", legend_out=False, height=3, aspect=4/3, hue_order=method_order)
        g.map_dataframe(td_plot, "unfair_impact", "mse")
        g.add_legend()
        plt.xlabel("Unfair Impact")
        plt.ylabel("MSE")

        dataname = os.path.splitext(os.path.basename(dataset))[0]
        plt.savefig("{}-impact-mse-prop.pdf".format(dataname), bbox_inches="tight")
        plt.clf()

if __name__ == '__main__':
    fire.Fire(main)