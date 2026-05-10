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
    """This function contains the training loop and is build for parallel training
    Each seed will be executed in a separate cpu thread
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1) #set the threads to execute this funciton
    env_name = 'Hopper-v4' #the environment
    csv_filename = f"{env_name}_redq_results.csv"

    print(f"--> Starting Seed {seed} on a separate CPU core...")

    #set global seeds for reproducibility (the current seed we are training)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name) #init environment
    test_env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0] #pass environment dimensions
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    agent = REDQ(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit) #create the REDQ object

    #define REDQ parameters
    start_steps = 5000  #warm up steps
    utd_ratio = 20  #neural network updates per agent step in the env
    max_steps = 200000  #total steps
    eval_interval = 5000  #how often to evaluate

    #to track episodes (injecting the seed into the first reset)
    obs, info = env.reset(seed=seed)
    test_env.reset(seed=seed)  #seed the test env too
    ep_ret, ep_len = 0, 0

    #TRAINING
    for step in range(max_steps):

        #ask agent for action
        action = agent.select_action(obs, env=env, current_steps=step, start_steps=start_steps)

        #act on the environment and save observed reward
        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward
        ep_len += 1  #increment step size

        #store in agent's buffer
        agent.replay_buffer.store(obs, action, reward, next_obs, float(terminated))

        if step >= start_steps:
            for _ in range(utd_ratio):
                agent.train()  #when warm up is done, update utd_ratio times the neural nets

        obs = next_obs  #set new state

        #end of episode changes
        if terminated or truncated:
            obs, info = env.reset() #reset for next episode
            ep_ret, ep_len = 0, 0

        #every 5000 steps, test the agent performance without exploration
        if (step + 1) % eval_interval == 0:
            avg_return = evaluate_agent(agent, test_env) #evaluate current policy

            with open(csv_filename, mode='a', newline='') as file: #log the data
                writer = csv.writer(file)
                writer.writerow([env_name, seed, step + 1, avg_return])

def main():
    #setup the CSV logger header
    env_name = 'Hopper-v4'
    csv_filename = f"{env_name}_redq_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Environment", "Seed", "Step", "Average_Return"])

    #set seeds
    seeds = [10, 20, 30]

    with multiprocessing.Pool(processes=len(seeds)) as pool:
        pool.map(run_single_seed, seeds) #execute each seed in a different thread

    print("All seeds have finished training!")


if __name__ == "__main__":
    main()