import ray
from ray.rllib.algorithms.ppo import PPO

def train_simple():
    ray.init()
    
    config = {
        "env": "CartPole-v1",
        "framework": "torch",
        "num_workers": 1,
        "num_gpus": 0,
        "lr": 0.001,
    }
    
    algo = PPO(config=config)
    
    print("Starting training...")
    for i in range(10):
        result = algo.train()
        mean_reward = result["episode_reward_mean"]
        print(f"Iteration {i}: Mean reward = {mean_reward:.2f}")
        
        if mean_reward >= 195.0:
            print("Solved! Stopping early.")
            break
    
    algo.save("./trained_ppo")
    print("Training completed!")
    
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    train_simple()
