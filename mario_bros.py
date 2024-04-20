# mario bros usage
from pettingzoo.atari import mario_bros_v3

# env = mario_bros_v3.parallel_env(render_mode="human")
# observations, infos = env.reset()

# while env.agents:
#     # this is where you would insert your policy
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}

#     observations, rewards, terminations, truncations, infos = env.step(actions)
# env.close()

# testing environment
from pettingzoo.test import parallel_api_test
env = mario_bros_v3.parallel_env()
parallel_api_test(env, num_cycles=1000)