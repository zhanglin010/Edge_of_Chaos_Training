# Commands used to run this program

This program splits the process of generating raw data and plotting the results. The code before line 108(main.py) is about generating raw data, and all the raw data will be saved at directory './rawdata/'. After running the following commands, if you want to modify the result figures, you can just uncomment line 58(main.py) and run the code about plotting only.

## Fig.2 and Fig.3 in the paper

```bash
# Fig.2 and Fig.3(a)
nohup python main.py --dataset fashion_mnist --activation-func tanh --epochs 501 --optimizer SGD --lr 0.02 --batch-size 32 --momentum 0.5 --num-repeats 1 --weight-decay 0. &> out/fashion_mnist/normal/sgd_501_1_decay0.out &
# For Fig.3(b)-(d), you need to change the command-line arguments. For example,
# if you want to get the slope for (lr=0.01, batch_size=32, momentum=0.5),
# the command is:
nohup python main.py --dataset fashion_mnist --activation-func tanh --epochs 501 --optimizer SGD --lr 0.01 --batch-size 32 --momentum 0.5 --num-repeats 1 --weight-decay 0. &> out/fashion_mnist/normal/sgd_501_1_decay0.out &
# I already recorded all the slopes for Figure.3 in file './coefficient(figure3).py'.
# Runing coefficient(figure3).py will generate Fig.3(b)-(d).
```

## Fig.4-5 in the paper

```bash
# Fig.4
nohup python main.py --dataset fashion_mnist --activation-func tanh --epochs 501 --optimizer SGD --lr 0.02 --batch-size 32 --momentum 0.5 --num-repeats 1 --weight-decay 0. &> out/fashion_mnist/scale/sgd.out &
# Other parameter combinations (epochs, lr, batch_size, momentum) needed to run:
# (501, 0.005, 8, 0.5), (501, 0.01, 16, 0.5),
# (501, 0.04, 64, 0.5), (501, 0.08, 128, 0.5).
# After runing all these parameter combinations, uncomment line 118(main.py)
# and comment line 58 to plot Fig.4.

# Fig.5
nohup python main.py --dataset fashion_mnist --activation-func tanh --epochs 501 --optimizer SGD --lr 0.02 --batch-size 32 --momentum 0.5 --num-repeats 1 --weight-decay 0. &> out/fashion_mnist/scale/sgd.out &
# Other parameter combinations (epochs, lr, batch_size, momentum) needed to run:
# (1001, 0.01, 32, 0.5), (2001, 0.01, 32, 0),
# (4001, 0.005, 32, 0), (8001, 0.0025, 32, 0).
# After runing all these parameter combinations, uncomment line 119(main.py)
# and comment line 58 to plot Fig.5.
```

## Fig.6 in the paper

```bash
nohup python main.py --dataset fashion_mnist --activation-func tanh --epochs 501 --optimizer SGD --lr 0.02 --batch-size 32 --momentum 0.5 --num-repeats 10 --weight-decay 0 &> out/fashion_mnist/normal/sgd_501_10_decay0.out &
# The above command is for weight decay strength 0. Other weight decay strength
# needed to run: 0.00001, 0.00005, 0.00009, 0.00015, 0.0003, 0.0005, 0.001.
# After runing all these weight decay strength, uncomment line 122-125(main.py)
# and comment line 58 to plot Fig.6.
```

## For relu+Adam

```bash
# A similar figure as Fig.2
nohup python main.py --dataset fashion_mnist --activation-func relu --epochs 201 --optimizer Adam --lr 0.001 --batch-size 32 --num-repeats 1 --weight-decay 0. &> out/fashion_mnist/relu/adam_201_1_decay0.out &

# A similar figure as Fig.6
nohup python main.py --dataset fashion_mnist --activation-func relu --epochs 201 --optimizer Adam --lr 0.001 --batch-size 32 --num-repeats 10 --weight-decay 0. &> out/fashion_mnist/relu/adam_201_10_decay0.out &
## Other weight decay strength: 0.000001, 0.00001, 0.00004, 0.0001, 0.0003, 0.0005, 0.001
# After runing all these weight decay strength, uncomment line 122-125(main.py)
# and comment line 58 to plot the figure. (Note you need to change all the 0.00009
# and 0.0005 in plot_figure6.py to 0.00004 and 0.0003, and replace line 24 as:
# weight_decay_values = [0, 0.000001, 0.00001, 0.00004, 0.0001, 0.0003, 0.0005, 0.001].)
```
