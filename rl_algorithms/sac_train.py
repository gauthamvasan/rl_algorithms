import sys
import argparse
import gym
import torch
import yaml

import numpy as np
import matplotlib as mpl
mpl.use("TKAgg")
import matplotlib.pyplot as plt

from rl_algorithms.agent.mlp_policies import SACPolicy
from rl_algorithms.agent.replay_buffer import SACReplayBuffer
from rl_algorithms.agent.agents import SACAgent
from rl_algorithms.plot import smoothed_curve

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
    env = gym.make(cfg["env"])
    # env = gym.make('Pendulum-v0')
    env.seed(seed)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    # Task setup block end

    # Learner setup block
    torch.manual_seed(seed)
    ####### Start
    cfg["obs_dim"] = o_dim
    cfg["action_dim"] = a_dim
    np.random.seed(seed)
    buffer = SACReplayBuffer(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0],
                             size=cfg["n_buffer"])
    ac = SACPolicy(cfg=cfg, device=device)
    agent = SACAgent(cfg=cfg, actor_critic=ac, buffer=buffer, device=device)
    ####### End

    # Experiment block starts
    ret = 0
    n_episodes = 0
    epi_steps = 0
    rets = []
    all_epi_steps = []
    avgrets = []
    name = sys.argv[0].split('.')[-2].split('_')[-1]
    o = env.reset()
    num_steps = 500000
    checkpoint = 1000
    for steps in range(num_steps):

        # Select an action
        ####### Start
        # Replace the following statement with your own code for
        # selecting an action

        if steps > cfg["n_warmup"]:
            a = agent.compute_action(torch.as_tensor(o.astype(np.float32)))
        else:
            a = env.action_space.sample()
        ####### End

        # Observe
        op, r, done, infos = env.step(a)
        print("Step: {}, Obs: {}, Action: {}, Reward: {:.2f}, Done: {}".format(steps, op[:5], a, r, done))

        # Learn
        ####### Start
        agent.push_and_update(obs=o, action=a, reward=r, done=done, next_obs=op)
        o = op
        ####### End

        # Log
        ret += r
        epi_steps += 1
        if done: # or (ep_len == 1000)
            rets.append(ret)
            all_epi_steps.append(epi_steps)
            ret = 0
            o = env.reset()
            n_episodes += 1
            epi_steps = 0

            print("-" * 50)
            print("Episode {} complete. Return: {}, # steps: {}. Total steps: {}".format(
                n_episodes, rets[-1], all_epi_steps[-1], steps))

        if (steps+1) % checkpoint == 0:
            plot_rets = smoothed_curve(np.array(rets), np.array(all_epi_steps), x_tick=checkpoint,
                                       window_len=checkpoint)
            plt.clf()
            plt.plot(np.arange(1, len(plot_rets) + 1) * checkpoint, plot_rets)
            plt.pause(0.001)
            # torch.save(ac.state_dict(), "./sac_baseline_model.pt")

            data = np.zeros((2, len(rets)))
            data[0] = all_epi_steps
            data[1] = rets
            np.savetxt(name + str(seed) + ".txt", data)

    data = np.zeros((2, len(rets)))
    data[0] = all_epi_steps
    data[1] = rets
    np.savetxt(name + str(seed) + ".txt", data)
    plt.show()


if __name__ == "__main__":
    main()