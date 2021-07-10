"""Mostly standalone file for collecting data.
"""

import numpy as np


def collect_data(config, env, seed):
    """Collect data in the given env. Return a set of transitions.
    """
    demos = env.get_demonstrations()
    random_data = _collect_negative_data(env, demos, seed, config.num_negatives)
    return demos, random_data


def _collect_negative_data(env, demos, seed, num_negatives):
    rng = np.random.RandomState(seed)
    episodes = []
    print("Collecting negative data...", flush=True)
    for demo in demos:
        states_and_goals = []
        for (state, _, _, goal) in demo:
            states_and_goals.append((state, goal))
        for _ in range(num_negatives//len(demos)):
            # Sample random state
            state_idx = rng.choice(len(states_and_goals))
            state, goal = states_and_goals[state_idx]
            act = env.get_random_action(state)
            next_state = env.get_next_state(state, act)
            episodes.append([(state, act, next_state, goal)])
    print("\ndone collecting negative data", flush=True)
    return episodes
