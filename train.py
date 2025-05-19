import os
import argparse
import gym
import minedojo
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from gym.envs.registration import register  # 👈 Needed to register custom task

# 👇 Register a simple PvP custom environment
register(
    id='MineDojoPvPCustom-v0',
    entry_point='minedojo.sim.mc_meta:make',
    kwargs={
        'env_spec_path': None,
        'task_id': 'harvest-wool',  # Dummy base task, gets overridden
        'mod_loader': 'fabric',     # Important: match your loader
        'world_seed': 123,          # Optional
        'image_size': 160,
        'use_voxel': False,
        'record_hero_path': False,
        'no_log': True
    }
)

def make_env(rank, map_path, server_ip, server_port):
    def _init():
        os.environ['MINEDOJO_SERVER_IP'] = server_ip
        os.environ['MINEDOJO_SERVER_PORT'] = str(server_port)
        return gym.make('MineDojoPvPCustom-v0')  # 👈 Updated
    return _init


def main():
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str, default='localhost')
    parser.add_argument('--server_port', type=int, default=25565)
    parser.add_argument('--map_path', type=str, default='server/world/PvPPractice_1.20')
    parser.add_argument('--num_envs', type=int, default=2)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--checkpoint_freq', type=int, default=500_000)
    args = parser.parse_args()

    env_fns = [make_env(i, args.map_path, args.server_ip, args.server_port) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    model = PPO("CnnPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.num_envs,
        save_path="./checkpoints",
        name_prefix="pvp_minedojo"
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    model.save("pvp_minedojo_final")


if __name__ == '__main__':
    main()
