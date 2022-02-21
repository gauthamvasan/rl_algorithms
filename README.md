# [Deprecated]: As of Feb 20th, 2022 I've switched to using [gauthamvasan/rl_suite](https://github.com/gauthamvasan/rl_suite) for my own research work.

# rl_algorithms
Simple, easy to use implementations of Reinforcement Learning (RL) algorithms.

## Implemented Algorithms
- REINFORCE
- Batch Actor Critic
- Proximal Policy Optimization (PPO)
- Soft Actor Critic (SAC)

## How to use?
All the training scripts are stored under the `rl_algorithms` subfolder. I use yaml files to 
specify the hyper-paramers of an algorithm. 

Example usage:

```python
python ppo_train.py --seed 1 --config "config/ppo.yml" 
python sac_train.py --seed 1 --config "config/sac.yml" 
python reinforce_train.py --seed 1 --config "config/reinforce.yml" 
python batchac_train.py --seed 1 --config "config/batchac.yml"  
```
