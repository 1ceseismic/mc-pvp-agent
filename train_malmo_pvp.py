import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import gym
import malmo_pvp_env  # registers MalmoPvPEnv

def make_env(role, mission_file, port):
    def _init():
        return gym.make(
            'MalmoPvPCustom-v0',
            mission_file=mission_file,
            agent_role=role,
            port=port
        )
    return _init

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mission', type=str, default='missions/pvp_arena.xml')
    parser.add_argument('--port', type=int, default=10000)
    parser.add_argument('--hosts', type=int, default=1,
                        help='Number of parallel servers to spawn')
    parser.add_argument('--timesteps', type=int, default=2_000_000)
    args = parser.parse_args()

    roles = ['agent_1', 'agent_2']
    envs = []
    for i in range(args.hosts):
        for role in roles:
            envs.append(make_env(role, args.mission, args.port + i))

    vec_env = SubprocVecEnv(envs)
    model = PPO('CnnPolicy', vec_env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save('malmo_pvp_selfplay')
