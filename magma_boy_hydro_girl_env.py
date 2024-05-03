from typing import Dict
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
import numpy as np
import pygame
import gymnasium as gym
from gymnasium.spaces import Discrete

from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility

import sys
from pygame.locals import *

from game.game import Game
from game.board import Board
from game.character import MagmaBoy, HydroGirl
from game.controller import ArrowsController, WASDController, GeneralController
from game.gates import Gates
from game.doors import FireDoor, WaterDoor
from game.level_select import LevelSelect

"""
Action-Space
| 0         | No operation |
| 1         | Move up |
| 2         | Move right |
| 3         | Move left |
| 4         | Move upright |
| 5         | Move upleft |
"""

# Define action and observation spaces
NUM_ACTIONS = 6  # Example number of actions
NUM_AGENTS = 2  # MagmaBoy and HydroGirl
OBSERVATION_SHAPE = (24, 32, 3)  # Example observation shape

__all__ = ["env", "parallel_env", "raw_env"]

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
        self.agents = ['magma_boy', 'hydro_girl']
        # Define observation and action spaces for each agent
        # each player needs to be aware of other player's location, but not location of other gate, or distinguish between enemy liquids and goo
        self.observation_spaces = dict( zip(self.agents, [gym.spaces.Box(low=0, high=255, shape=OBSERVATION_SHAPE, dtype=np.uint8)] * len(self.agents)))
                                  
        self.action_spaces = dict( zip(self.agents, [Discrete(NUM_ACTIONS)] * len(self.agents)))

        # Initialize other environment parameters
        
        self.possible_agents = self.agents
        self.render_mode = "human"

        self.board = Board('game/data/level1.txt')  
        self.magma_boy = None
        self.hydro_girl = None 
        self.clock = None

        gate_location = (285, 128)
        plate_locations = [(190, 168), (390, 168)]
        self.gate = Gates(gate_location, plate_locations)
        self.gates = [self.gate]

        fire_location = (64, 48)
        self.fire_door = FireDoor(fire_location)
        water_location = (128, 48)
        self.water_door = WaterDoor(water_location)
        self.doors = [self.fire_door, self.water_door]

        self.infos = None
        self.rewards = None
        self.terminateds = None

        self.controller = GeneralController()
        self.game = Game()

        #check if issue do to moving of init
        #self.events = pygame.event.get()
        
        
        self.agent_selection = 0

        self.reset()
                

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, *, seed=None, options=None):
        # Reset environment state and return initial observations
        # Initialize game objects, boards, etc.
        self.magma_boy = MagmaBoy((16, 336))  # Example MagmaBoy initialization
        self.hydro_girl = HydroGirl((35, 336))  # Example HydroGirl initialization
        self.clock = pygame.time.Clock()  # Example clock initialization
        self.clock.tick(60)
        # Other initialization code...
        
        observations = {
            'magma_boy': self.observe(self.magma_boy),
            'hydro_girl': self.observe(self.hydro_girl)
        }
        self.rewards = {'magma_boy': 0, 'hydro_girl': 0}
        self.terminateds = {'magma_boy': False, 'hydro_girl': False}
        self.infos = {agent: {} for agent in self.agents}
        self.render()
        self.agent_selection = 0
        return observations, self.infos

    def step(self, actions: Dict[str, int]):
        # Perform actions, update environment state, and return next observations, rewards, terminateds, and infos
        self.move_players(actions)
        self.check_for_special_events()  # Example: check for gate press, door open, etc.

        observations = {
            'magma_boy': self.observe(self.magma_boy),
            'hydro_girl': self.observe(self.hydro_girl)
        }
        self.rewards = self.calculate_rewards()  # Example: calculate rewards based on game logic
        self.terminateds = self.check_for_episode_completion()  # Example: check if episode is terminateds
        infos = {agent: {} for agent in self.agents}  # Empty info dictionaries for now

        
        self.render()
        truncateds = {'magma_boy': False, 'hydro_girl': False}
        self.agent_selection = 0
        return observations, self.rewards, self.terminateds, truncateds, infos

    def observe(self, agent):
        # Get observation for a character (e.g., MagmaBoy or HydroGirl)
        return self.state()

    def state(self):
        state = pygame.surfarray.pixels3d(self.game.get_display()).copy()
        state = np.rot90(state, k=3)
        state = np.fliplr(state)
        return state

    def move_players(self, actions):
        # Move players based on actions
        
        self.controller.agent_control(self.magma_boy, actions['magma_boy'])
        self.controller.agent_control(self.hydro_girl, actions['hydro_girl'])

        self.game.move_player(self.board, self.gates, [self.magma_boy, self.hydro_girl])

    def calculate_rewards(self):
        # Calculate rewards based on game logic
        reward = {'magma_boy': 0, 'hydro_girl': 0}  # Initialize with zero rewards
        if self.magma_boy.is_dead() or self.hydro_girl.is_dead():
            reward = {'magma_boy': -1, 'hydro_girl': -1}
        elif Game.level_is_done(self.doors):
            reward = {'magma_boy': 1, 'hydro_girl': 1}
        # Calculate rewards based on game state, actions, etc.
        return reward

    def check_for_episode_completion(self):
        # Check if episode is terminateds based on game state
        terminated = {'magma_boy': False, 'hydro_girl': False}  # Example: not terminateds by default
        # Check conditions for episode termination
        if self.magma_boy.is_dead() or self.hydro_girl.is_dead() or Game.level_is_done(self.doors):
            terminated = {'magma_boy': True, 'hydro_girl': True}
        return terminated

    def check_for_special_events(self):
        self.game.check_for_death(self.board, [self.magma_boy, self.hydro_girl])

        self.game.check_for_gate_press(self.gates, [self.magma_boy, self.hydro_girl])

        self.game.check_for_door_open(self.fire_door, self.magma_boy)
        self.game.check_for_door_open(self.water_door, self.hydro_girl)

    def render(self):
        pygame.init()
        self.game.draw_level_background(self.board)
        self.game.draw_board(self.board)
        if self.gates:
            self.game.draw_gates(self.gates)
        self.game.draw_doors(self.doors)
        self.game.draw_player([self.magma_boy, self.hydro_girl])
        self.game.refresh_window()
        return self.state()

    def last(self, observe: bool = True):
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        agent = self.agent_selection
        assert agent is not None
        observation = self.observe(agent) if observe else None
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def close(self):
        pygame.quit()


    
# Assuming you have additional classes like Board, MagmaBoy, HydroGirl, etc., implement those as needed.

# Example usage:
# env = MagmaBoyHydroGirlEnv()
# observations, infos = env.reset()

# while env.agents:
#     # this is where you would insert your policy
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}

#     observations, rewards, terminations, truncations, infos = env.step(actions)
# env.close()