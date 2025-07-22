import gymnasium as gym
from stable_baselines3 import PPO
import os
import argparse

from rl_env import DrawingEnv

def train_agent(image_path, total_timesteps=25000):
    """
    Trains a PPO agent on the DrawingEnv.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    abs_image_path = os.path.abspath(image_path)

    # Create the environment
    env = DrawingEnv(image_path=abs_image_path)

    # Instantiate the agent
    model = PPO("MultiInputPolicy", env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    # Save the agent
    model.save("ppo_drawing_agent")
    
    print("Training complete. Model saved to ppo_drawing_agent.zip")

    # --- Example of how to use the trained agent ---
    # obs, _ = env.reset()
    # for _ in range(500):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         obs, _ = env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a drawing agent.")
    parser.add_argument("image_path", help="Path to the reference image.")
    parser.add_argument("--timesteps", type=int, default=25000, help="Number of training timesteps.")
    args = parser.parse_args()

    train_agent(args.image_path, args.timesteps)
