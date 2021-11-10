import torch
import argparse
import os
import time
import json

import rl_algorithms.agent.utils as utils
import numpy as np

from rl_algorithms.agent.sac_rad import SacRadAgent
from rl_algorithms.logger import Logger
from rl_algorithms.envs.visual_mujoco_reacher import VisualMujocoReacher2D

config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],

    'latent': 50,

    'mlp': [
        [-1, 1024],
        [1024, 1024],
        [1024, -1]
    ],
}

def save_returns_np(fname, rets, ep_steps):
    data = np.zeros((2, len(rets)))
    data[0] = np.array(rets)
    data[1] = np.array(ep_steps)
    np.savetxt(fname, data)

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--target_type', default='visual_reacher', type=str)
    parser.add_argument('--ip', default='localhost', type=str)
    parser.add_argument('--image_height', default=125, type=int)
    parser.add_argument('--image_width', default=200, type=int)
    parser.add_argument('--stack_frames', default=3, type=int)
    parser.add_argument('--tol', required=True, type=float)
    parser.add_argument('--image_period', required=True, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', required=True, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=False, action='store_true')
    parser.add_argument('--max_updates_per_step', default=10, type=int)
    parser.add_argument('--episode_length', required=True, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=1, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # misc
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--work_dir', default='./results', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    # parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=10000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    parser.add_argument('--freeze_cnn', default=False, type=bool)
    # Number of updates
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not 'conv' in config:
        image_shape = (0, 0, 0)
    else:
        image_shape = (3 * args.stack_frames, args.image_height, args.image_width)

    env = VisualMujocoReacher2D(args.tol, image_shape, args.image_period)
    utils.set_seed_everywhere(args.seed, env)

    if not args.async_mode:
        version = 'SAC_sync'
    elif args.async_mode and args.lock:
        version = 'SACv1'
    elif args.async_mode:
        version = 'SAC_async'
    else:
        raise NotImplementedError('Not a supported mode!')

    args.work_dir += f'/{version}_' \
                     f'seed={args.seed}_tol={args.tol}_' \
                     f'image_period={args.image_period}_' \
                     f'image_shape={image_shape}/'

    utils.make_dir(args.work_dir)

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # Numpy file with returns and corresponding episode lengths
    fname_rets = os.path.join(args.work_dir, 'returns.txt')

    if args.device == '':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    agent = SacRadAgent(
        image_shape=env.image_space.shape,
        proprioception_shape=env.proprioception_space.shape,
        action_shape=env.action_space.shape,
        device=device,
        net_params=config,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        actor_lr=args.actor_lr,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_tau=args.encoder_tau,
        rad_offset=args.rad_offset,
        async_mode=args.async_mode,
        replay_buffer_capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        max_updates_per_step=args.max_updates_per_step,
        init_steps=args.init_steps,
        freeze_cnn=args.freeze_cnn,
        update_every=args.update_every,
        update_epochs=args.update_epochs,
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, ret, episode_step, done = 0, 0, 0, True
    rets = []
    ep_steps = []
    image, state = env.reset()
    start_time = time.time()
    for step in range(args.env_steps + args.init_steps):
        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(image, state)

        # step in the environment
        next_image, next_state, reward, done, _ = env.step(action)

        ret += reward
        episode_step += 1

        agent.push_sample(image, state, action, reward, next_image, next_state, done)

        if done or (episode_step == args.episode_length):  # set time out here
            L.log('train/duration', time.time() - start_time, step)
            L.log('train/return', ret, step)
            L.dump(step)
            rets.append(ret)
            ep_steps.append(episode_step)
            next_image, next_state = env.reset()
            done = False
            ret = 0
            episode_step = 0
            episode += 1
            L.log('train/episode', episode, step)
            save_returns_np(fname_rets, rets, ep_steps)
            '''
            if args.save_model and step > 0 and step % args.save_model_freq == 0:
                agent.save(model_dir, step)
            '''
            start_time = time.time()

        stat = agent.update_networks(step)
        for k, v in stat.items():
            L.log(k, v, step)

        image = next_image
        state = next_state

    # save the last model
    if args.save_model:
        agent.save(model_dir, step)

    if not done:
        # N.B: We're adding a partial episode to have a cumulative experience of length 'env_steps'.
        #   But this partial data point shouldn't be used for plotting!
        rets.append(ret)
        ep_steps.append(step)
        save_returns_np(fname_rets, rets, ep_steps)

    # Clean up
    agent.close()
    env.close()


if __name__ == '__main__':
    main()
