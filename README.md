*Update: Minor updates to code. Added integration to tensorboard so you can can log and create graphs of training, see graph of model, and visualize your weights and biases distributions as they update during training. 

* A3G A NEW GPU/CPU ARCHITECTURE OF A3C FOR SUBSTANTIALLY ACCELERATED TRAINING!!
*Training with A3G benefits training speed most when using larger models i.e using raw pixels for observations such as training in atari environments that have raw pixels for state representation*

# RL A3C Pytorch Continuous

![A3C LSTM playing BipedalWalkerHardcore-v2](https://github.com/dgriff777/a3c_continuous/blob/master/demo/BPHC.gif)

This repository includes my implementation with reinforcement learning using Asynchronous Advantage Actor-Critic (A3C) in Pytorch an algorithm from Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."

# A3G!!
New implementation of A3C that utilizes GPU for speed increase in training. Which we can call **A3G**. A3G as opposed to other versions that try to utilize GPU with A3C algorithm, with A3G each agent has its own network maintained on GPU but shared model is on CPU and agent models are quickly converted to CPU to update shared model which allows updates to be frequent and fast by utilizing Hogwild Training and make updates to shared model asynchronously and without locks. This new method greatly increase training speed and models and can be see in my [rl_a3c_pytorch][55] repo that training that use to take days to train can be trained in as fast as 10minutes for some Atari games!

[55]: https://github.com/dgriff777/rl_a3c_pytorch

### A3C LSTM

This is continuous domain version of my other a3c repo. Here I show A3C can solve BipedalWalker-v3 but also the much harder BipedalWalkerHardcore-v3 version as well. "Solved" meaning to train a model capable of averaging reward over 300 for 100 consecutive episodes

Added trained model for BipedWalkerHardcore-v3

## Requirements

- Python 2.7+
- openai gym<=0.19.0
- Pytorch
- setproctitle

## Training
*When training model it is important to limit number of worker processes to number of cpu cores available as too many processes (e.g. more than one process per cpu core available) will actually be detrimental in training speed and effectiveness*

To train agent in BipedalWalker-v3 environment with 6 different worker processes:
*On a MacPro 2014 laptop traing typically takes 10mins to get to a winning solution*

```
python main.py --workers 6 --env BipedalWalker-v3 --save-max --amsgrad --model MLP --stack-frames 1
```

![Graph of training run for BipedalWalker-v3](https://github.com/dgriff777/a3c_continuous/blob/master/demo/BW3_Rewards_graph.jpg)
Graph showing training a BipedalWalker-v3 agent with the above command on Macbook pro. Train a successful model in 10mins on your laptop!
 
To train agent in BipedalWalkerHardcore-v3 environment with 64 different worker processes:
*BipedalWalkerHardcore-v3 is much harder environment compared to normal BipedalWalker*
*On a 72 cpu AWS EC2 c5.18xlarge instance training with 64 worker processes takes up to 24hrs to get to model that could solve the environment but can get to a model that averages 280+ takes just 2-3 hrs. Using enhanced A3G design, training model takes only 4-6hrs*

```
python main.py --workers 64 --env BipedalWalkerHardcore-v3 --save-max --amsgrad --model CONV --stack-frames 4
```

#A3C-GPU

To train agent in BipedalWalkerHardcore-v3 environment with 32 different worker processes with new A3C-GPU:

```
python main.py --env BipedalWalkerHardcore-v3 --workers 32 --gpu-ids 0 1 2 3 --amsgrad --model CONV --stack-frames 4
```


Hit Ctrl C to end training session properly

![A3C LSTM playing BipedalWalkerHardcore-v2](https://github.com/dgriff777/a3c_continuous/blob/master/demo/BPHC3.gif)

## Evaluation
To run a 100 episode gym evaluation with trained model
```
python gym_eval.py --env BipedalWalkerHardcore-v3 --num-episodes 100 --stack-frames 4 --model CONV --new-gym-eval
```

## Project Reference

- https://github.com/ikostrikov/pytorch-a3c
- https://github.com/andrewliao11/pytorch-a3c-mujoco


## README STILL UNDER CONSTRUCTION
