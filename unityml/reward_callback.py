from typing import Union
import gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common import logger


class RewardCallback(BaseCallback):
    def __init__(
            self,
            eval_env: Union[VecEnv, gym.Env],
            check_freq: int,
            reward_threshold: float,
            verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.env = eval_env
        self.check_freq = check_freq
        self.logger = None
        self.last_callback_step = 0
        self.reward_threshold = reward_threshold

    def _init_callback(self) -> None:
        self.logger = logger

    def _on_step(self) -> bool:
        if 0 < self.check_freq <= (self.num_timesteps - self.last_callback_step):
            rewards = [info['r'] for info in self.model.ep_info_buffer]
            if len(rewards) > 0:
                mean_100_reward = np.mean(rewards)
                self.logger.record('train/mean_100_reward', float(mean_100_reward))
                if mean_100_reward >= self.reward_threshold:
                    print(
                        f"Stopping training because the mean(100) reward {mean_100_reward:.2f}"
                        f" is above the threshold {self.reward_threshold}"
                    )
                    return False
            self.last_callback_step = self.num_timesteps
        return True
