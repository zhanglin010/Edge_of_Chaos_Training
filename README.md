# Code for Edge of Chaos as A Guiding Principle for Modern Neural Network Training

## Description

This repository is to accompany paper: [Edge of Chaos as A Guiding Principle for Modern Neural Network Training](https://arxiv.org/abs/2107.09437).

## Prerequisites

Note that the code in Folder1 is MATLAB script. Other folders are written in Python. The Prerequisites for Python are in the Dockerfile. In our group DGX server, use the docker image: dgx/linzhang/tensorflow:20.03-tf2-py3.

## Usage

This repo contains four file folders.

### Folder1 (Fig.1(b))

Calculate the theoretical boundary between the spin-glass phase and paramagnetic/ferromagnetic phases.

Command to run:

```bash
run_tanh.m
# For relu: run_relu.m
```

After running the above command, a result file named 'data_100bins_tanh.mat' will shown in current directory. Then we need to modify the file by replacing all 1 with 0 in -1 < J0/J < 1 except J = 1, and rename the variable and file as 'integrals_modified' and 'data_100bins_tanh_modified.mat'. Lastly, copy the file 'data_100bins_tanh_modified.mat' to next program './2.Phase_diagram/rawdata/tanh'.

### Folder2 (Fig.1(b))

Numerical result for the Phase diagram in Fig.1(b) in the paper.

Command to run:

```bash
nohup python main.py --activation-func tanh --num-neurons 100 --num-bins 100 --num-iter 500 &> out/tanh/100_100_500 &
# For relu: nohup python main.py --activation-func relu --num-neurons 100 --num-bins 100 --num-iter 100 &> out/relu/100_100_100 &
```

After running this program, a result image named '100_100_500.png' will shown in directory 'results/separation/tanh', and this is Fig.1(b) in the paper. Moreover, a data file named '100_100_500' will be generated at directory 'rawdata/tanh'. For next program, we need -1 < J0/J < 1 and 0 < 1/J < 3. So we need modify the calculate_separation.py a little bit. I ran the simulation and get the data file '50_100_100_tanh'. We need to copy this data file to next program '3.Model_evolution/rawdata/'.

### Folder3 (All other figures)

This folder is the main program. Many commands are needed, so please check the file './3.Model_evolution/results/commands.md'.

## Note

This is a simplified version of the code, I did not managed to test all the programs. Thus, please feel free to contact me(linzhang_010@outlook.com) if you have any problems or questions.
