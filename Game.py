# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
import pygame
import numpy as np
env = gym.make("CarRacing-v3", domain_randomize=True, render_mode="human")

# normal reset, this changes the colour scheme by default


# reset with no colour scheme change
observation, info = env.reset(options={"randomize": False})

episode_over = False
total_reward = 0
action_set = [0., 0., 0.]

observations = []
rewards = []

while not episode_over:
     #stearing, gass, break
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    keys = pygame.key.get_pressed()
    # gas
    if keys[pygame.K_w]:
        action_set[1] += 0.1
    else:
        action_set[1] -= 0.1

    # brake
    if keys[pygame.K_s]:
        action_set[2] += 0.1
    else:
        action_set[2] -= 0.1

    # steering
    if keys[pygame.K_LEFT]:
        action_set[0] -= 0.05
    elif keys[pygame.K_RIGHT]:
        action_set[0] += 0.05
    else:
        action_set[0] *= 0.9   # eases back toward center when neither is held

    if action_set[0] > 1.0:
        action_set[0] = 1.0
    if action_set[0] < -1.0:
        action_set[0] = -1.0
    if action_set[1] > 1.0:
        action_set[1] = 1.0
    if action_set[1] < 0:
        action_set[1] = 0
    if action_set[2] > 1.0:
        action_set[2] = 1.0
    if action_set[2] < 0:
        action_set[2] = 0
    print(action_set)
    action = np.array(action_set)

    observation, reward, terminated, truncated, info = env.step(action)

    observations.append(observation)
    rewards.append(reward)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()

observations = np.array(observations)
rewards = np.array(rewards)
np.savez("runs/run_9.npz", observations = observations, rewards=rewards)