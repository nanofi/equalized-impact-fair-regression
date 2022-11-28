#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd

def main(path):
    os.makedirs(path, exist_ok=True)
    datapath = os.path.join(path, "data.csv")
    if not os.path.exists(datapath):
        raise "You must place 'data.csv' in {0}".format(datapath)
    
    data = pd.read_csv(datapath, sep=",")
    age = data["R0000600"]
    race = data["R0009600"] #categorical
    gender = data["R0214800"] #binary
    grade90 = data["R3401501"] 
    income06 = data["T0912400"]
    income96 = data["R5626201"]
    income90 = data["R3279401"]
    partner = data["R2734200"] #binary
    height = data["R0481600"] 
    weight = data["R1774000"] 
    famsize = data["R0217502"] 
    genhealth = data["H0003400"] 
    illegalact = data["R0304900"] #categorical
    charged = data["R0307100"] 
    jobsnum90 = data["R3403500"] 
    afqt89 = data["R0618300"] 
    typejob90 = data["R3127300"] 
    #data = data[data.R3127500 >= 0] 
    #classjob90 = data["R3127500"] 
    jobtrain90 = data["R3146100"] 
    #data = data[data.R0304900 >= 0]

    attrs = [gender,income90,genhealth,illegalact,age,charged,grade90,jobsnum90,afqt89,jobtrain90]
    data = pd.concat(attrs, axis=1)
    data["job_agri"] = [int(10 <= j <= 39) for j in typejob90]
    data["job_mining"] = [int(40 <= j <= 59) for j in typejob90]
    data["job_construction"] = [int(60 <= j <= 69) for j in typejob90]
    data["job_manuf"] = [int(100 <= j <= 399) for j in typejob90]
    data["job_transp"] = [int(400 <= j <= 499) for j in typejob90]
    data["job_wholesale"] = [int(500 <= j <= 579) for j in typejob90]
    data["job_retail"] = [int(580 <= j <= 699) for j in typejob90]
    data["job_fin"] = [int(700 <= j <= 712) for j in typejob90]
    data["job_busi"] = [int(721 <= j <= 760) for j in typejob90]
    data["job_personal"] = [int(761 <= j <= 791) for j in typejob90]
    data["job_enter"] = [int(800 <= j <= 811) for j in typejob90]
    data["job_pro"] = [int(812 <= j <= 892) for j in typejob90]
    data["job_pub"] = [int(900 <= j <= 932) for j in typejob90]
    data = data.rename(columns={"R0000600":"age"})
    data = data.rename(columns={"R0214800":"gender"})
    data["gender"] = data["gender"]-1 #1,2->0,1
    data = data.rename(columns={"R3279401":"income"})
    data = data[data.income >= 0]
    data = data.rename(columns={"R3401501":"grade90"})
    data = data[data.grade90 >= 0]
    data = data.rename(columns={"H0003400":"genhealth"})
    data = data[data.genhealth >= 0]
    data = data.rename(columns={"R0304900":"illegalact"})
    data = data[data.illegalact >= 0]
    data = data.rename(columns={"R0307100":"charged"})
    data = data[data.charged >= 0]
    data = data.rename(columns={"R3403500":"jobsnum90"})
    data = data[data.jobsnum90 >= 0]
    data = data.rename(columns={"R0618300":"afqt89"})
    data = data[data.afqt89 >= 0]
    data = data.rename(columns={"R3146100":"jobtrain90"})
    data = data[data.jobtrain90 >= 0]

    data = data.rename(columns={'income': 'outcome'})
    data["outcome"] = data["outcome"] / 10000.0
    data = data.rename(columns={'gender': 'sensitive'})

    save = {
        "num_groups": 2,
        "num_train": 0.7,
        "dataset": data
    }
    processedpath = os.path.join(path, "nlsy79.pd")
    with open(processedpath, "wb") as f:
        pickle.dump(save, f)

if __name__ == '__main__':
    main('../data/nlsy79')