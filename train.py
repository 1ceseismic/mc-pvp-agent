import gym
import minedojo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing as mp # Import multiprocessing

# Number of parallel environments
num_envs = 2

def make_env(rank):
    def _init():
        env = minedojo.make(
            task_id='harvest-wool',
            image_size=160,
            use_voxel=False,
            record_hero_path=False
        )
        return env
    return _init

# Add this block to protect the main execution part
if __name__ == '__main__':
    # On Windows, you might need to set the start method
    # explicitly if you encounter further issues, although spawn is default.
    # mp.set_start_method("spawn", force=True) # Uncomment if needed

    # Create vectorized environment
    # This line and everything after needs to be inside the if block
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Train the agent
    model = PPO('CnnPolicy', env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save('ppo_harvest_wool')

    # Close the environment properly
    env.close()