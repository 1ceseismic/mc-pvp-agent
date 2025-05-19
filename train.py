import gym
import minedojo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing as mp
from minedojo.sim import InventoryItem
# Number of parallel environments
num_envs = 2
inv_config= [
    InventoryItem(slot=0, name="diamond_sword", variant=None, quantity=1),
    InventoryItem(slot=36, name="diamond_boots", variant=None, quantity=1),
    InventoryItem(slot=37, name="diamond_leggings", variant=None, quantity=1),
    InventoryItem(slot=38, name="diamond_chestplate", variant=None, quantity=1),
    InventoryItem(slot=39, name="diamond_helmet", variant=None, quantity=1),
    InventoryItem(slot=40, name="shield", variant=None, quantity=1),
]

def make_env(rank):
    def _init():
        # Create a CombatMeta environment using task_id='combat'
        # Pass parameters from the CombatMeta documentation as keyword arguments
        env = minedojo.tasks.CombatMeta(
            target_names=['zombie'],
            target_quantities=[1],
            allow_mob_spawn=True,
            fast_reset=False,
            image_size=(160, 256),
            start_at_night=False,
            initial_inventory=inv_config
        );
        # env = minedojo.make(
        #     task_id='combat',
        #     image_size=160,
        #     use_voxel=False,
        #     # --- CombatMeta specific parameters ---
        #     # REQUIRED arguments as per documentation:
        #     target_names=['zombie', 'skeleton'], # Example: The agent needs to defeat zombies and skeletons
        #     target_quantities=[5, 3],          # Example: Defeat 5 zombies and 3 skeletons

        #     # Optional parameters you might want:
        #     initial_mobs=['zombie', 'skeleton', 'spider'], # Example: start with some common hostile mobs present
        #     start_at_night=True, # Hostile mobs spawn more at night
        #     allow_mob_spawn=True, # Allow more mobs to spawn naturally over time
        #     fast_reset=True, # Use fast reset for quicker restarts
        #     # initial_inventory=[minedojo.types.InventoryItem('diamond_sword', 1), minedojo.types.InventoryItem('cooked_beef', 10)],
        #     # start_health=20,
        #     # start_food=20,
        #     # specified_biome='plains' # Example: fight in a specific biome
        #     # ------------------------------------
        # )
        return env
    return _init

# Add this block back to protect the main execution part for multiprocessing
if __name__ == '__main__':
    # It's good practice to include the freeze_support() or set_start_method if distributing,
    # but often not strictly needed for simple script execution on Windows.
    # mp.freeze_support() # Uncomment if you plan to freeze the script into an executable
    # mp.set_start_method("spawn", force=True) # Uncomment if you have issues on non-Windows platforms or complex setups

    print(f"Creating {num_envs} combat environments...")

    # Create vectorized environment
    try:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
        print("Environments created. Starting training...")

        # Train the agent
        # You might need a different policy or network architecture depending on the observation space
        # 'CnnPolicy' is suitable if 'image_size' is specified and it's the main observation.
        model = PPO('CnnPolicy', env, verbose=1)
        model.learn(total_timesteps=1_000_000) # Use a larger number of timesteps for meaningful training
        model.save('ppo_combat_agent')

        print("Training finished. Model saved.")

    except Exception as e:
        print(f"An error occurred during environment creation or training: {e}")
    finally:
        # Ensure environment is closed even if training fails
        if 'env' in locals() and env is not None:
             print("Closing environment...")
             env.close()
             print("Environment closed.")