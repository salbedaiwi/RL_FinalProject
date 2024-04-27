from typing import Dict
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import numpy as np
import pygame
import gymnasium as gym
from game import board, character, controller, doors, game, gates, level_select


# Define action and observation spaces
NUM_ACTIONS = 6  # Example number of actions
NUM_AGENTS = 2  # MagmaBoy and HydroGirl
OBSERVATION_SHAPE = (24, 32)  # Example observation shape

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env

class MagmaBoyHydroGirlEnv(ParallelEnv[str, np.ndarray, int]):
    metadata = {
        "render_modes": ["human"],
        "name": "magmaboy_hydrogirl_v0"
    }
    def __init__(self, render_mode=None):
        super().__init__()

        # Define observation and action spaces for each agent
        # each player needs to be aware of other player's location, but not location of other gate, or distinguish between enemy liquids and goo
        self.observation_spaces = {'magma_boy': gym.spaces.Box(low=0, high=255, shape=OBSERVATION_SHAPE, dtype=np.uint8),
                                   'hydro_girl': gym.spaces.Box(low=0, high=255, shape=OBSERVATION_SHAPE, dtype=np.uint8)}
        self.action_spaces = {'magma_boy': gym.spaces.Discrete(NUM_ACTIONS),
                              'hydro_girl': gym.spaces.Discrete(NUM_ACTIONS)}

        # Initialize other environment parameters
        self.agents = ['magma_boy', 'hydro_girl']
        self.possible_agents = self.agents
        self.render_mode = "human"

        self.board = Board('data/level1.txt')  # Example board initialization
        self.magma_boy = MagmaBoy((16, 336))  # Example MagmaBoy initialization
        self.hydro_girl = HydroGirl((35, 336))  # Example HydroGirl initialization
        self.clock = pygame.time.Clock()

        gate_location = (285, 128)
        plate_locations = [(190, 168), (390, 168)]
        self.gate = Gates(gate_location, plate_locations)
        self.gates = [gate]

        self.fire_door = FireDoor(64, 48)
        self.water_door = WaterDoor(128, 48)
        self.doors = [fire_door, water_door]


        self.reward = 0
        self.dones = {'magma_boy': False, 'hydro_girl': False}


    def reset(self, seed=None, options=None):
        # Reset environment state and return initial observations
        # Initialize game objects, boards, etc.
        self.board = Board('data/level1.txt')  # Example board initialization
        self.magma_boy = MagmaBoy((16, 336))  # Example MagmaBoy initialization
        self.hydro_girl = HydroGirl((35, 336))  # Example HydroGirl initialization
        self.clock = pygame.time.Clock()  # Example clock initialization
        # Other initialization code...

        observations = {
            'magma_boy': self.get_observation(self.magma_boy),
            'hydro_girl': self.get_observation(self.hydro_girl)
        }
        self.rewards = 0
        self.dones = {'magma_boy': False, 'hydro_girl': False}
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions: Dict[str, int]):
        # Perform actions, update environment state, and return next observations, rewards, dones, and infos
        self.move_players(actions)
        self.check_for_special_events()  # Example: check for gate press, door open, etc.

        observations = {
            'magma_boy': self.get_observation(self.magma_boy),
            'hydro_girl': self.get_observation(self.hydro_girl)
        }
        rewards = self.calculate_rewards()  # Example: calculate rewards based on game logic
        dones = self.check_for_episode_completion()  # Example: check if episode is done
        infos = {agent: {} for agent in self.agents}  # Empty info dictionaries for now

        return observations, rewards, dones, infos

    def get_observation(self, character):
        # Get observation for a character (e.g., MagmaBoy or HydroGirl)
        return character.get_observation()

    def move_players(self, actions):
        # Move players based on actions
        self.magma_boy.move(actions['magma_boy'])
        self.hydro_girl.move(actions['hydro_girl'])
        # Other movement logic...

    def calculate_rewards(self):
        # Calculate rewards based on game logic
        rewards = {'magma_boy': 0, 'hydro_girl': 0}  # Initialize with zero rewards
        # Calculate rewards based on game state, actions, etc.
        return rewards

    def check_for_episode_completion(self):
        # Check if episode is done based on game state
        done = {'magma_boy': False, 'hydro_girl': False}  # Example: not done by default
        # Check conditions for episode termination
        return done

# Assuming you have additional classes like Board, MagmaBoy, HydroGirl, etc., implement those as needed.

# Example usage:
env = MagmaBoyHydroGirlEnv()
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()