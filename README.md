*Update: Major update providing large training performance gains as well as code working with latest versions of pytorch and gym libraries. With updated code now possible to train a successful model that can avg 300+ on BipedalWalkerHardcore-v3 env in just 20-40mins using just CPU!!

* A3G A NEW GPU/CPU ARCHITECTURE OF A3C FOR SUBSTANTIALLY ACCELERATED TRAINING!!
*Training with A3G benefits training speed most when using larger models i.e using raw pixels for observations such as training in atari environments that have raw pixels for state representation*

# RL A3C Pytorch Continuous

![A3C LSTM playing BipedalWalkerHardcore-v3](https://github.com/dgriff777/a3c_continuous/blob/master/demo/BPHC.gif)

This repository includes my implementation with reinforcement learning using Asynchronous Advantage Actor-Critic (A3C) in Pytorch an algorithm from Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."

# A3G!!
New implementation of A3C that utilizes GPU for speed increase in training. Which we can call **A3G**. A3G as opposed to other versions that try to utilize GPU with A3C algorithm, with A3G each agent has its own network maintained on GPU but shared model is on CPU and agent models are quickly converted to CPU to update shared model which allows updates to be frequent and fast by utilizing Hogwild Training and make updates to shared model asynchronously and without locks. This new method greatly increase training speed and models and can be see in my [rl_a3c_pytorch][55] repo that training that use to take days to train can be trained in as fast as 10minutes for some Atari games!

[55]: https://github.com/dgriff777/rl_a3c_pytorch

### A3C LSTM

This is continuous domain version of my other a3c repo. Here I show A3C can solve BipedalWalker-v3 but also the much harder BipedalWalkerHardcore-v3 version as well. "Solved" meaning to train a model capable of averaging reward over 300 for 100 consecutive episodes

## Requirements

- Python 3.7+
- openai gym==0.26.2
- Pytorch
- spdlog (Is a much faster logging library than the standard python logging library)
- setproctitle

## Training
*When training model it is important to limit number of worker processes to number of cpu cores available as too many processes (e.g. more than one process per cpu core available) will actually be detrimental in training speed and effectiveness*

To train agent in BipedalWalker-v3 environment with 8 different worker processes:
*On a MacPro 2014 laptop traing typically takes less than 5mins to converge to a winning solution*

```
python main.py --env BipedalWalker-v3 --optimizer Adam --shared-optimizer --workers 8 --amsgrad --stop-when-solved --model-300-check --tensorboard-logger
```

![Graph of training run for BipedalWalker-v3](https://github.com/dgriff777/a3c_continuous/blob/master/demo/BW3_Rewards_graph.jpg)
Graph showing training a BipedalWalker-v3 agent with the above command on Macbook pro. Train a successful model in 10mins on your laptop!

To tail training log for above command use the following command:
```
tail -f logs/BipedalWalker-v3_log
```
 
To train agent in BipedalWalkerHardcore-v3 environment with 18 different worker processes:
*BipedalWalkerHardcore-v3 is much harder environment compared to normal BipedalWalker*
*Training a successful model than can achieve a 300+ avg reward on 100 episode test typical takes 20-40mins*

```
python main.py --env BipedalWalkerHardcore-v3 --optimizer Adam --shared-optimizer --workers 18 --amsgrad --stop-when-solved --model-300-check --tensorboard-logger
```

![Graph of training run for BipedalWalkerHardcore-v3](https://github.com/dgriff777/a3c_continuous/blob/master/demo/BWH3_Rewards_graph.jpg)
Graph showing training a BipedalWalkerHardcore-v3 agent with above command to train succesful model in under 30mins!


To tail training log for above command use the following command:
```
tail -f logs/BipedalWalkerHardcore-v3_log
```

Hit Ctrl C to end training session properly

## Evaluation
To run a 100 episode gym evaluation with trained model
```
python gym_eval.py --env BipedalWalkerHardcore-v3 --num-episodes 100
```

## Project Reference

- https://github.com/ikostrikov/pytorch-a3c
- https://github.com/andrewliao11/pytorch-a3c-mujoco

