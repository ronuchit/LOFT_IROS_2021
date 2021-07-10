"""Top-level script.
"""

import os
import sys
import pickle as pkl
import time
from data_collection import collect_data
from args import parse_args
from envs import create_env
from approaches import LOFT, ApproachTimeout, ApproachFailed
from settings import create_config


def main():
    """Main function.
    """
    universe_start = time.time()
    # Create config, holds global and experiment-specific params
    config = create_config(parse_args())
    env = create_env(config)
    simulator = env.get_next_state
    state_preds = env.get_state_predicates()
    action_preds = env.get_action_predicates()
    if not os.path.exists(config.data_dir):
        os.mkdir(config.data_dir)
    data_fname = os.path.join(config.data_dir, f"{config.env}.p")
    if config.collect_data:
        print("Running data collection")
        env.set_seed(config.start_seed+999)
        data = collect_data(config, env, config.start_seed+999)
        print("Done in {:.5f} seconds, collected {} demos, {} random".format(
            time.time()-universe_start, len(data[0]), len(data[1])))
        with open(data_fname, "wb") as f:
            data = pkl.dump(data, f)
        sys.exit(0)
    assert os.path.exists(data_fname)
    with open(data_fname, "rb") as f:
        data = pkl.load(f)
    print(f"Loaded {len(data[0])} demos and {len(data[1])} "
          f"random from {data_fname}")
    # Run experiments
    for seed in range(config.start_seed, config.start_seed+config.num_seeds):
        print(f"\nRunning seed {seed}")
        # Seed env
        env.set_seed(seed)
        # Create & train approach
        approach = LOFT(config, simulator, state_preds, action_preds)
        approach.set_seed(seed)
        train_start = time.time()
        print("Training approach...", flush=True)
        approach.train(data)
        train_time = time.time()-train_start
        print("Done training approach in {:.5f} seconds".format(train_time))
        # Run the approach on test problems
        test_probs = env.get_test_problems()
        _run_testing(config, env, approach, test_probs)
    print(f"Total time elapsed: {time.time()-universe_start} sec")


def _run_testing(config, env, approach, test_problems):
    """Run testing.
    """
    num_solved = 0
    print("\n**TESTING START**")
    for i, (init_state, goal) in enumerate(test_problems):
        print(f"Running test episode {i+1} of {len(test_problems)}", flush=True)
        ep_start_time = time.time()
        try:
            plan = approach.plan(init_state, goal, config.approach_timeout)
        except (ApproachFailed, ApproachTimeout) as e:
            print(f"\tApproach failed: {e.args[0]}")
            continue
        state = init_state
        for j in range(config.max_test_episode_length):
            solved = goal.holds(state)
            if solved:
                print("\tSolved in {} steps, took {:.5f} seconds".format(
                    j, time.time()-ep_start_time))
                num_solved += 1
                break
            try:
                act = plan.pop(0)
            except IndexError:
                print("\tFailed to solve, reached end of plan")
                break
            next_state = env.get_next_state(state, act)
            state = next_state
        else:
            print("\tFailed to solve, reached max_test_episode_length")

    print(f"In total, solved {num_solved} / {len(test_problems)}")
    print("**TESTING END**\n")


if __name__ == "__main__":
    main()
