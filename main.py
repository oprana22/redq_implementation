from redq import REDQ
import gymnasium as gym
import numpy as np
import torch
import csv
import os
import multiprocessing


def evaluate_agent(agent, test_env, num_episodes=5):
    """runs the agent with the current policy for num_episodes
    returns the average reward over num_episodes
    """
    returns = []
    for _ in range(num_episodes):
        obs, _ = test_env.reset() #reset env
        done = False
        ep_ret = 0
        while not done:
            action = agent.select_action(obs, deterministic=True)  # deterministic --> no exploration noise
            obs, reward, terminated, truncated, _ = test_env.step(action) #take action and observe env
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)

    avg_return = np.mean(returns)
    print(f"\n---> Evaluation (Avg Return over {num_episodes} episodes): {avg_return:.2f}\n")
    return avg_return


def run_single_seed(seed):
    """This function contains everything that used to be inside your 'for' loop."""
    env_name = 'Hopper-v4'
    csv_filename = f"{env_name}_redq_results.csv"

    print(f"--> Starting Seed {seed} on a separate CPU core...")

    # Set global seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    test_env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = REDQ(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit)

    # REDQ parameters (restored to paper standard for science run)
    start_steps = 5000  # warm up steps
    utd_ratio = 20  # neural network updates per agent step in the env
    max_steps = 1_000_000  # Total steps for the academic run
    eval_interval = 5000  # How often to evaluate and log

    # to track episodes (injecting the seed into the first reset)
    obs, info = env.reset(seed=seed)
    test_env.reset(seed=seed)  # Seed the test env too
    ep_ret, ep_len = 0, 0

    # TRAINING
    for step in range(max_steps):

        # ask agent for action
        action = agent.select_action(obs, env=env, current_steps=step, start_steps=start_steps)

        # act on the environment and save observed reward
        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward
        ep_len += 1  # increment step size

        # store in agent's buffer
        agent.replay_buffer.store(obs, action, reward, next_obs, float(terminated))

        if step >= start_steps:
            for _ in range(utd_ratio):
                agent.train()  # when warm up is done, update utd_ratio times the neural nets

        obs = next_obs  # set new state

        # end of episode changes
        if terminated or truncated:
            # Optionally quiet this print statement if it gets too noisy during a 1M step run
            print(f"Seed: {seed} | Step: {step} | Episode Return: {ep_ret:.2f} | Length: {ep_len}")

            # reset for next episode
            obs, info = env.reset()
            ep_ret, ep_len = 0, 0

        # every 5000 steps, test the agent pure performance without exploration
        if (step + 1) % eval_interval == 0:
            avg_return = evaluate_agent(agent, test_env)

            # 4. Open the CSV in 'append' mode ('a') and log the data
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([env_name, seed, step + 1, avg_return])

def main():
    # 1. Setup the CSV Logger header
    env_name = 'Hopper-v4'
    csv_filename = f"{env_name}_redq_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Environment", "Seed", "Step", "Average_Return"])

    # 2. Define the seeds
    seeds = [10, 20, 30]

    # 3. The Magic: Launch all seeds simultaneously!
    # This maps each seed in your list to its own CPU process.
    with multiprocessing.Pool(processes=len(seeds)) as pool:
        pool.map(run_single_seed, seeds)

    print("All seeds have finished training!")


if __name__ == "__main__":
    main()