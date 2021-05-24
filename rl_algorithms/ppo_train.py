import sys
import argparse
import gym
import torch
import yaml

import numpy as np
import matplotlib as mpl
mpl.use("TKAgg")
import matplotlib.pyplot as plt

from rl_algorithms.agent.learner import PPO
from rl_algorithms.agent.mlp_policies import Actor, Critic
from rl_algorithms.agent.buffer import SimpleBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Please provide a config.yml file")
    parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")
    args = parser.parse_args()

    seed = args.seed
    cfg = yaml.load(open(args.config))

    # Task setup block starts
    # Do not change
    env = gym.make('CartPole-v1')
    env.seed(seed)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    # Task setup block end

    # Learner setup block
    torch.manual_seed(seed)
    ####### Start
    cfg["obs_dim"] = o_dim
    cfg["action_dim"] = a_dim
    np.random.seed(seed)
    buffer = SimpleBuffer()
    pi = Actor(cfg=cfg, device=device)
    critic = Critic(cfg=cfg, device=device)
    learner = PPO(cfg=cfg, pi=pi, critic=critic, buffer=buffer, device=device)
    ####### End

    # Experiment block starts
    ret = 0
    rets = []
    avgrets = []
    o = env.reset()
    num_steps = 500000
    checkpoint = 10000
    for steps in range(num_steps):

        # Select an action
        ####### Start
        # Replace the following statement with your own code for
        # selecting an action
        # a = np.random.randint(a_dim)

        a, lprob = pi.compute_action(torch.as_tensor(o.astype(np.float32)))
        ####### End

        # Observe
        op, r, done, infos = env.step(a)

        # Learn
        ####### Start
        learner.push_and_update(obs=o, action=a, reward=r, log_prob=lprob, done=done, next_obs=op)
        print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}".format(steps, op[:5], a, r, done))
        o = op
        ####### End

        # Log
        ret += r
        if done:
          rets.append(ret)
          ret = 0
          o = env.reset()

        if (steps+1) % checkpoint == 0:
          avgrets.append(np.mean(rets))
          rets = []
          plt.clf()
          plt.plot(range(checkpoint, (steps+1)+checkpoint, checkpoint), avgrets)
          plt.pause(0.001)
    name = sys.argv[0].split('.')[-2].split('_')[-1]
    data = np.zeros((2, len(avgrets)))
    data[0] = range(checkpoint, num_steps+1, checkpoint)
    data[1] = avgrets
    np.savetxt(name+str(seed)+".txt", data)
    # plt.show()


if __name__ == "__main__":
    main()
