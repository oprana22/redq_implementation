from redq import REDQ
import gymnasium as gym
import numpy as np


def evaluate_agent(agent, test_env, num_episodes=5):
    """runs the agent with the current policy for num_episodes
    returns the average reward over num_episodes
    """
    returns = []
    for _ in range(num_episodes):
        obs, _ = test_env.reset()
        done = False
        ep_ret = 0
        while not done:
            action = agent.select_action(obs, deterministic=True) #deterministic --> no exploration noise
            obs, reward, terminated, truncated, _ = test_env.step(action)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)

    avg_return = np.mean(returns)
    print(f"\n---> Evaluation (Avg Return over {num_episodes} episodes): {avg_return:.2f}\n")
    return avg_return


def main():
    env_name = 'Hopper-v4' #change for the environment you want to train on
    env = gym.make(env_name)
    test_env = gym.make(env_name)  #keep a separate env for evaluation

    #extract dimensions of state and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]  #for the Tanh scaling

    #init the class object
    agent = REDQ(obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit)

    #REDQ parameters
    start_steps = 1000  #warm up steps
    utd_ratio = 5  #neural network updates per agent step in the env

    #to track episodes
    obs, info = env.reset()
    ep_ret, ep_len = 0, 0

    #TRAINING
    for step in range(3000):  # Start with 100k for testing, move to 1M later

        #ask agent for action
        action = agent.select_action(obs, env=env, current_steps=step)

        #act on the environment and save observed reward
        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward
        ep_len += 1 #increment step size

        #store in agent's buffer
        agent.replay_buffer.store(obs, action, reward, next_obs, float(terminated))

        if step >= start_steps:
            for _ in range(utd_ratio):
                agent.train() #when warm up is done, update utd_ratio times the neural nets

        obs = next_obs #set new state

        #end of episode changes
        if terminated or truncated:
            print(f"Step: {step} | Episode Return: {ep_ret:.2f} | Length: {ep_len}")

            #reset for next episode
            obs, info = env.reset()
            ep_ret, ep_len = 0, 0

        #every 5000 steps, test the agent pure performance without exploration
        if (step + 1) % 1000 == 0:
            evaluate_agent(agent, test_env)


if __name__ == "__main__":
    main()