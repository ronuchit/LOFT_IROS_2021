"""Base class for an env.
"""

import abc
import numpy as np
from approaches import Oracle, ApproachFailed, ApproachTimeout
from structs import WORLD


class BaseEnv:
    """An env implements a simulator and a goal.
    """
    def __init__(self, config):
        self._cf = config
        self._seed = None
        self._rng = None

    def set_seed(self, seed):
        """Reset self._seed.
        """
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)

    @abc.abstractmethod
    def get_state_predicates(self):
        """Set of state predicates in this environment.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_action_predicates(self):
        """Set of action predicates.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _get_demo_problems(self, num):
        """Returns a list of planning problems for demo. Each is a tuple
        of (low-level initial state, goal). Goal is a LiteralConjunction that
        also implements a holds(state) method on low-level states.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_test_problems(self):
        """Returns a list of planning problems for evaluation. Each is a tuple
        of (low-level initial state, goal). Goal is a LiteralConjunction that
        also implements a holds(state) method on low-level states.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_next_state(self, state, action):
        """Transition function / simulator on low-level states/actions.
        Returns a next low-level state.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_random_action(self, state):
        """Get a random valid action from the given state.
        """
        raise NotImplementedError("Override me!")

    def get_demonstrations(self):
        """Returns a list of demonstrations.
        """
        print("Collecting demonstrations...", end="", flush=True)
        simulator = self.get_next_state
        state_preds = self.get_state_predicates()
        action_preds = self.get_action_predicates()
        oracle_approach = Oracle(self._cf, simulator, state_preds, action_preds)
        oracle_approach.set_seed(self._seed)
        oracle_approach.train(([], []))
        demos = []
        for init_state, goal in self._get_demo_problems(num=self._cf.num_demos):
            try:
                plan = oracle_approach.plan(init_state, goal,
                                            self._cf.approach_timeout)
            except (ApproachTimeout, ApproachFailed) as e:
                print(f"WARNING: demo collection failed with error: {e}")
                continue
            demo = []
            state = init_state
            state_seq = [init_state]
            for act in plan:
                next_state = self.get_next_state(state, act)
                demo.append((state, act, next_state, goal))
                state = next_state
                state_seq.append(state)
            if not goal.holds(state):
                print("WARNING: demonstration did not achieve goal, discarding")
                continue
            demos.append(demo)
        print("done collecting demonstrations", flush=True)
        return demos

    @staticmethod
    def _copy_state(state):
        """Copy a state, which is a nested dictionary
        of objects -> attributes -> values.
        """
        new_state = {}
        for obj in state:
            new_state[obj] = {}
            for attr, val in state[obj].items():
                if obj != WORLD and isinstance(val, tuple):
                    raise Exception("Tuples for objs not supported")
                new_state[obj][attr] = \
                    BaseEnv._copy_state_value(val)
        return new_state

    @staticmethod
    def _copy_state_value(val):
        """Helper for _copy_state
        """
        if val is None or isinstance(val, (float, bool, int, str)):
            return val
        if isinstance(val, (list, tuple, set)):
            return type(val)(BaseEnv._copy_state_value(v) for v in val)
        if hasattr(val, 'copy'):
            return val.copy()
        raise NotImplementedError(f"Unsupported type in state: {type(val)}")

    def _sample_ground_act(self, state, act_pred, discrete_args):
        cont_vals = list(act_pred.sample(self._rng, state, *discrete_args))
        act_args = []
        for var_type in act_pred.var_types:
            if var_type.is_continuous:
                act_args.append(var_type(
                    f"sampled_{var_type}", cont_vals.pop(0)))
            else:
                act_args.append(discrete_args.pop(0))
        assert not cont_vals
        assert not discrete_args
        return act_pred(*act_args)
