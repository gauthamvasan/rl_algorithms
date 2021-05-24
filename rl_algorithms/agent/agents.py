import torch
import time


from copy import deepcopy
from rl_algorithms.agent.sac import SAC


class SACAgent(SAC):
    def __init__(self, cfg, actor_critic, buffer, device=torch.device('cpu')):
        """ Simple, Synchronous Agent
        Args:
            cfg (dict): Contains hyper-parameters and configs
            learner (object): An instance of project.algos.sac.SAC
            pi (object): Policy. This should be a callable function
        """
        super(SACAgent, self).__init__(cfg=cfg, actor_critic=actor_critic, device=device)
        self.buffer = buffer
        self.steps = 0
        self.n_updates = 0

    def compute_action(self, obs, deterministic=False):
        with torch.no_grad():
            action = self.actor_critic.compute_action(obs, deterministic=deterministic)
            return action

    def push_and_update(self, obs, action, reward, done, next_obs):
        self.buffer.store(obs, action, reward, done, next_obs)
        self.steps += 1
        if self.steps >= self.n_warmup and self.steps % self.steps_between_updates == 0:
            tic = time.time()
            for i in range(self.n_epochs):
                observations, actions, rewards, dones, next_observations = self.buffer.sample_batch(self.batch_size)
                self.update(observations, actions, rewards, dones, next_observations)
                self.n_updates += 1
            if self.n_updates % 20 == 0:
                print("Update {} took {}".format(self.n_updates, time.time() - tic))

    def state_dict(self):
        model = {'actor_critic': self.actor_critic.state_dict()}
        return model

    def load_state_dict(self, model):
        self.actor_critic.load_state_dict(model['actor_critic'])
        self.pi = deepcopy(self.actor_critic.pi)
        for p in self.pi.parameters():
            p.requires_grad = False
