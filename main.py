from util_RL import *
from TD3agent import *
from env import *
import config
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from scipy.optimize import minimize # Assuming this is used elsewhere if not directly here

# Ensure the plot directory exists early
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)
TRAJECTORY_DIR = "./trajectory/explore"
os.makedirs(TRAJECTORY_DIR, exist_ok=True)


#parameters
N = config.num_bins
kT = 0.5981
sim_time = 2000 # Make sure this is an integer, not a tuple
window_size = 100 # Changed from 10 to 100 based on previous EWA context
max_action = 20 # Max number of actions per episode (which corresponds to config.propagation_step)
# max_traj_length is now determined by env.py based on config.propagation_step and config.dcdfreq_mfpt
# state_size should be max_traj_length or a derived value
# Assuming state_size should be the length of the padded trajectory, which is max_traj_length from env.py
# This will be updated after env initialization.

discrete_size = N

random.seed(1245)
np.random.seed(1232)
torch.manual_seed(2431)

# Initialize environment - max_traj_length is now derived inside Propagate_1D
env = Propagate_1D(N=N, kT=kT, sim_time=sim_time, window_size=window_size)

# The state_size needs to match env.max_traj_length which is set in env.__init__
state_size = env.max_traj_length # Get the derived max_traj_length from the environment
agent = TD3Agent( alpha=0.0003, beta=0.0003, state_size=state_size, action_size=1, hidden_size=128, gamma=0.99, tau=0.005, action_noise=0.1,
                 policy_noise=0.2, policy_noise_clip=0.5, delay_time=2, max_size=10000,
                 batch_size=256, max_action=20, learning_rate=1e-4)

total_rewards = []
# mfpts = [] # Commented out if not actively calculated/used

def train(num_episodes):
    for episode in range(num_episodes):
        state = env.reset() # Reset environment at the beginning of each episode
        agent.reset_action_counter()
        done = False
        total_reward = 0
        start_epsilon = 0.8
        final_epsilon = 0.05
        epsilon = (-(start_epsilon-final_epsilon)/num_episodes)*episode +start_epsilon #epsilon decay

        # Inner loop for a single episode
        episode_steps = 0 # Track steps within this episode
        while not done:
            # Check if we've exceeded the configured propagation steps for an episode
            if episode_steps >= config.propagation_step:
                print(f"DEBUG: Episode {episode+1} reached max propagation steps ({config.propagation_step}). Forcing done.")
                done = True # Force end of episode if steps exceed config
                continue # Skip to the next iteration (which will then break out of while loop)

            action = agent.choose_action(state, epsilon, train=True)


            # Ensure action is not None before proceeding
            if action is None:
                print(f"Warning: Agent returned None action at episode {episode+1}, step {episode_steps}. Breaking episode.")
                done = True
                continue
            _,b,_=action
            # env.step now returns `done` as the last element
            next_state, reward, reached_flag, traj, coor_x, done = env.step(action)
            agent.remember(state, b, reward, next_state, done)

            # Print the last 10 states of the trajectory (for debugging)
            print(f"Episode {episode+1}, Step {episode_steps}: Trajectory last 10 x-coords: {coor_x[-10:].flatten()}")

            # The plotting logic for trajectory is now handled inside util_RL.py
            # and is triggered by propagate_ld_wrapper. No need for this block in main.py
            # if False:
            #     # ... (original plotting block removed) ...
            agent.learn()
            total_reward += reward
            state = next_state
            episode_steps += 1 # Increment episode step counter

            if done: # Break from inner while loop if episode is done (either reached target or max steps)
                print(f"Episode {episode+1} finished. Reached: {'Yes' if reached_flag == 1 else 'No'}. Total Reward for episode: {total_reward:.2f}")
                break

        # This block was previously inside the `while not done` loop and caused too many plots
        # Moving it outside ensures one final trajectory plot per episode.
        # However, plotting per `prop_index` is handled by util_RL.py.
        # If you want one overarching trajectory plot for the entire episode, keep this or adapt.
        # Given the request to plot 'after each iteration', the util_RL.py solution is better.
        # This original plot logic might be for a different type of visualization.
        # Temporarily commenting out or adjusting the main.py plotting logic to avoid redundancy.
        # if True: # This condition was always True, leading to many overlapping plots
        #     traj_arr = np.array(coor_x) # coor_x is from the *last* step of the episode
        #     timestep = np.arange(len(traj_arr))
        #     plt.figure() # Create a new figure for each plot
        #     plt.plot(timestep , traj_arr, '-')
        #     plt.title(f'trajectory episode: {episode+1}')
        #     plt.xlabel('time step')
        #     plt.ylabel('state value')
        #     filename = os.path.join(PLOT_DIR, f'RL_LD_trajectory_episode_{episode+1}_final.png')
        #     plt.savefig(filename)
        #     plt.close()


        #ep_mfpt = env.get_mfpt(state) # Uncomment if mfpt calculation is active
        total_rewards.append(total_reward)
        #mfpts.append(ep_mfpt) # Uncomment if mfpt calculation is active

        # This print statement can be kept for episode summary
        print(f"Episode Summary {episode+1}: Total Reward: {total_reward:.2f}, Average Reward per Action: {total_reward/agent.action_counter:.2f}, Reached: {'Yes' if reached_flag == 1 else 'No'}")

        # if episode in [100, 200, 400, 508, 1000, 2000, 3000, num_episodes-1]: # Added more save points
        #     torch.save(agent.model.state_dict(), f'./model_checkpoint_episode_{episode+1}_N{N}.pt')
        #     print(f"Saved model checkpoint at episode {episode+1}")


num_episodes = 200
# num_episodes = 100 # For faster testing


train(num_episodes)

# Save final model
# torch.save(agent.model.state_dict(), f'./model_final_N{N}.pt')
# print("Training done!")
np.savetxt(os.path.join(PLOT_DIR, 'total_rewards.txt'), total_rewards)


# Plot overall rewards after training is complete
episodes = range(1, num_episodes + 1)
plt.figure(figsize=(12, 7)) # Larger figure for rewards plot
plt.plot(episodes, total_rewards, '-', color='blue', linewidth=1.5, label='Total Reward per Episode')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', color='blue', fontsize=12)
plt.title('Reward over Training Episodes', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
final_rewards_plot_path = os.path.join(PLOT_DIR, 'RL_training_rewards.png')
plt.savefig(final_rewards_plot_path)
plt.show()
print(f"Saved final rewards plot to {final_rewards_plot_path}")


