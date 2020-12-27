"""This uses the facade pattern to interface with Unity ML
"""
from typing import Any, Tuple, Dict, Optional
import random
import gym
from gym.spaces import Box
import numpy as np
from unityagents import UnityEnvironment, BrainInfo, BrainParameters


class UnityMlFacade(gym.Env):
    def __init__(
            self,
            *args,
            executable_path: str,
            train_mode: bool = True,
            seed: Optional[int] = None,
            environment_port: Optional[int] = None,
            **kwargs):
        """A facade between Unity ML and OpenAI gym.

        Args:
            *args: arguments which are directly passed to the Unity environment. This is supposed to make the
                the initialization of the wrapper very similar to the initialization of the Unity environment.
            train_mode: toggle to set the unity environment to train mode
            seed: sets the seed of the environment - if not given, a random seed will be used
            environment_port: port of the environment, used to be able to run multiple environments concurrently
        """
        self.train_mode = train_mode
        self.unity_env, self.brain_name, self.brain = self._setup_unity(
            *args,
            path=executable_path,
            environment_port=environment_port,
            seed=seed,
            no_graphics=True,
            **kwargs)
        self.action_space, self.observation_space, self.reward_range = self._get_environment_specs(self.brain)
        self.episode_step = 0
        self.episode_reward = 0.0

    def step(self, action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        brain_info = self.unity_env.step(action)[self.brain_name]
        state, reward, done = self._parse_brain_info(brain_info)
        self.episode_reward += reward
        info = (
            dict(episode=dict(
                r=self.episode_reward,
                l=self.episode_step))
            if done else dict())
        self.episode_step += 1
        return state, reward, done, info

    def reset(self) -> np.ndarray:
        brain_info = self.unity_env.reset(train_mode=self.train_mode)[self.brain_name]
        self.episode_step = 0
        self.episode_reward = 0.0

        return self._parse_brain_info(brain_info)[0]

    def render(self, mode='human') -> None:
        pass

    def close(self):
        self.unity_env.close()

    @staticmethod
    def _parse_brain_info(info: BrainInfo) -> Tuple[Any, float, bool]:
        """Extract the state, reward and done information from an environment brain."""
        observation = info.vector_observations[0]
        reward = info.rewards[0]
        done = info.local_done[0]

        return observation, reward, done

    @staticmethod
    def _setup_unity(*args, path: str, environment_port: Optional[int], seed: Optional[int],
                     **kwargs) -> Tuple[UnityEnvironment, str, BrainParameters]:
        """Setup a Unity environment and return it and its brain."""
        kwargs['file_name'] = path
        kwargs['seed'] = random.randint(0, int(1e6)) if not seed else seed
        if environment_port:
            kwargs['base_port'] = environment_port

        unity_env = UnityEnvironment(*args, **kwargs)
        brain_name = unity_env.brain_names[0]
        brain = unity_env.brains[brain_name]

        return unity_env, brain_name, brain

    @staticmethod
    def _get_environment_specs(brain: BrainParameters) -> Tuple[Box, Box, Tuple[float, float]]:
        """Extract the action space, observation space and reward range info from an environment brain."""
        action_space_size = brain.vector_action_space_size
        observation_space_size = brain.vector_observation_space_size
        action_space = Box(
            low=np.array(action_space_size * [-1.0]),
            high=np.array(action_space_size * [1.0]))
        observation_space = Box(
            low=np.array(observation_space_size * [-float('inf')]),
            high=np.array(observation_space_size * [float('inf')]))
        reward_range = (0.0, float('inf'))

        return action_space, observation_space, reward_range
