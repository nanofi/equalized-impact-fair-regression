
# Setup
We use pip to setup the environment for running our codes. With the existence of `pip3` command, we can setup all the dependencies by executing the following command:
```
./jax-setup.sh
```

# Datasets

Download the NLSY79 dataset from the NLS investigator: https://www.nlsinfo.org/investigator/pages/login.jsp , and put it as `../data/nlsy79/data.csv`. Then, execute the following command:
 ```
./dataset.sh
```
The command preprocesses the datasets. The community and crime dataset is automatically downloaded. The resultant datasets can be found in the `../data` folder.

# Training
We first generate all the training parameters by executing the following commands:
```
./param.sh
```
Then, we can run the training processes with all the parameters by the following commands:
```
./train.py results/params/train/
```
Then, the resultant models will be written in the `results/models/` folder.

# Evaluation
We can evaluate the trained model by the following commands:
```
./misc/make_test_param.py
./test.py results/params/test/
./misc/summrize.py
```
We can then find `results/summary/result.pd` file, the pickled pandas DataFrame, composed of values of the evaluated measures.

We can reproduce Fig. 2 and Fig. 3 by executing the following commands:
```
mv plot
cp ../results/summary/result.pd .
./plot_summary.py
```

# Render CDFs

We can reproduce Fig. 4 by executing the following commands:
```
./misc/make_dump_param.py
./dump.py results/params/dump
./misc/plot_cdf.py
```
We can find the rendered CDFs in the `/results/plot/dist/` folder.

# License

The only purpose of providing this code is to make the experimental results reproducible. Under the review process, it prohibits using this code for any purpose except to confirm the experimental results. After acceptance, we'll open this code and specify its license.