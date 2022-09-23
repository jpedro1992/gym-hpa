from gym.envs.registration import register

register(
    id='Redis-v0',
    entry_point='gym_hpa.envs:Redis',
)