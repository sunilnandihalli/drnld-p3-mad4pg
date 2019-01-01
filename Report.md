# Introduction
The goal of this project is to train agents such that can play tennis by bouncing the ball over the net. It is a collaborative setting.

# Implemention

Multi-Agent Distributed Distributional Deep Deterministic Policy Gradients (MAD4PG) has been implemented.

## Distributed Distributional Deep Deterministic Policy Gradients (D4PG)

This is a off-policy actor-critic method based on DDPG with critic maintaining a belief over the ActionValue function instead of a single value for each action state pair.

![alt text](https://github.com/d4pg/drnld-p3-mad4pg/blob/master/d4pg.jpg)

## Multi-Agent Deep Deterministic Policy Gradients (MADDPG)

![alt text](https://github.com/d4pg/drnld-p3-mad4pg/blob/master/maddpg.jpg)



## Hyperparamters

Various hyper parameters were tried before settling on the following values
Hyperparameter | Value
--- | ---    
batch_size | 64
gamma | 0.99
tau | 1e-2
lr_actor | 1e-4
lr_critic | 1e-4
num_mc_steps | 5 
update_every | 4
Vmax | 0.7
Vmin | -0.7
N_ATOMS | 51

## Network Structures

### Actor

Layer | Dimension
--- | ---
Input | N x 24
Linear Layer, Leaky Relu | N x 256
Linear Layer, Leaky Relu | N x 2
Batchnormalization1D | N x 2
Tanh Output | N x 2

### Critic

Layer | Dimension
--- | ---
Input | N x 24
Linear Layer, Leaky Relu | N x 128
Linear Layer + Actor Output, Leaky Relu | N x (128 + 2)
Linear Layer, Leaky Relu | N x 128
Linear Layer | N x 51

## Training Results
The goal was achieved in 6220 episodes. The graph is embedded in the ipython notebook.


# Ideas for future work
1. The current model discards the training data once the length of the rest of the episode is less than the number-montecarlo-steps. This seems wasteful. I tried to fix it but was not successful. 
2. Try using lambda return instead of n-step bootstrapping. This can fix the above problem too.
