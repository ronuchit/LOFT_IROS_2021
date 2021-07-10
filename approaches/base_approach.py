"""Base class for an approach.
"""

import abc
import time
import heapq as hq
import numpy as np
from pddlgym.utils import get_object_combinations
from utils import compute_static_preds, compute_dr_reachable_lits, \
    PyperplanHAddHeuristic, ground_literal, EnvironmentFailure
from structs import LiteralConjunction, Operator, WORLD


class BaseApproach:
    """BaseApproach definition.
    """
    def __init__(self, config, simulator, state_preds, action_preds):
        """All approaches are initialized with a simulator and the predicates.
        The samplers are bundled within the action_preds.
        """
        self._cf = config
        self._simulator = simulator
        self._state_preds = state_preds
        self._action_preds = action_preds
        self._seed = None

        self._num_calls = 0
        self._operators = None
        self._ground_operators = None

    def set_seed(self, seed):
        """Reset self._seed.
        """
        self._seed = seed

    @abc.abstractmethod
    def _get_operators(self, data):
        raise NotImplementedError("Override me!")

    def train(self, data):
        """Train this approach on the given data.
        """
        demos, random_data = data
        data = (demos, random_data)
        self._operators = self._get_operators(data)
        self._ground_operators = None

    def plan(self, init_state, goal, timeout):
        """Get a plan (sequence of actions) for the given planning problem.
        Can raise ApproachFailed or ApproachTimeout as necessary.
        """
        start_time = time.time()
        skeleton_gen = self._skeleton_generator(init_state, goal, timeout)
        sampler_rng = np.random.RandomState(self._seed+self._num_calls)
        self._num_calls += 1
        while True:
            try:
                skeleton, expected_lits_sequence = next(skeleton_gen)
            except StopIteration:
                break
            plan = self._sample_continuous_values(
                init_state, goal, skeleton, expected_lits_sequence,
                sampler_rng, start_time, timeout)
            if plan is not None:
                print(f"success! found plan of length {len(plan)}: {plan}")
                return plan
        if time.time()-start_time > timeout:
            raise ApproachTimeout("Timed out in skeleton search!")
        raise ApproachFailed("Ran out of skeletons!")

    def _sample_continuous_values(self, init_state, goal, skeleton,
                                  expected_lits_sequence, rng,
                                  start_time, timeout):
        """Backtracking search over continuous values.
        """
        cur_idx = 0
        num_tries = [0 for _ in skeleton]
        idx_to_max_num_tries = [self._cf.backtracking_num_samples_per_step
                                if any(v.is_continuous for v in a.variables)
                                else 1 for a in skeleton]
        plan = [None for _ in skeleton]
        traj = [init_state]+[None for _ in skeleton]
        while cur_idx < len(skeleton):
            if time.time()-start_time > timeout:
                raise ApproachTimeout("Timed out in backtracking!")
            assert num_tries[cur_idx] < idx_to_max_num_tries[cur_idx]
            num_tries[cur_idx] += 1
            state = traj[cur_idx]
            skel_act = skeleton[cur_idx]
            act_args = self._sample_act_args(state, skel_act, rng)
            ground_act = skel_act.predicate(*act_args)
            plan[cur_idx] = ground_act
            try:
                traj[cur_idx+1] = self._simulator(state, ground_act)
            except EnvironmentFailure:
                return None
            cur_idx += 1
            # Check literal sequence constraint. Backtrack if failed.
            # If expected_lits_sequence is None, just keep going until we
            # reach the final timestep, then check the goal in the simulator.
            if expected_lits_sequence is None:
                if cur_idx == len(skeleton) and goal.holds(traj[cur_idx]):
                    return plan  # success!
                if cur_idx < len(skeleton):
                    continue  # all good, no need to backtrack
            else:
                assert len(traj) == len(expected_lits_sequence)
                assert len(skeleton) == len(expected_lits_sequence)-1
                for lit in expected_lits_sequence[cur_idx]:
                    # Found a mismatch between expected sequence
                    # and actual trajectory
                    if not lit.holds(traj[cur_idx]):
                        break
                else:
                    # Found no mismatches, no need to backtrack
                    if cur_idx == len(skeleton) and goal.holds(traj[cur_idx]):
                        return plan  # success!
                    if cur_idx < len(skeleton):
                        continue  # all good, no need to backtrack
            # Do backtracking.
            cur_idx -= 1
            while num_tries[cur_idx] == idx_to_max_num_tries[cur_idx]:
                num_tries[cur_idx] = 0
                plan[cur_idx] = None
                traj[cur_idx+1] = None
                cur_idx -= 1
                if cur_idx < 0:
                    return None  # backtracking exhausted
        # Should only get here if the skeleton was empty
        assert not skeleton
        if goal.holds(init_state):
            return []
        return None

    def _parser(self, state):
        lits = set()
        for pred in self._state_preds:
            lits |= pred.get_ground_literals(state)
        return lits

    def _extract_transition_data(self, data):
        transition_data = []
        for episode in data:
            for (state, action, next_state, _) in episode:
                hl_state = self._parser(state)
                hl_next_state = self._parser(next_state)
                transition_data.append((hl_state, action, hl_next_state))
        return transition_data

    def _skeleton_generator(self, init_state, goal, timeout):
        start_time = time.time()
        prio_rng = np.random.RandomState(self._seed+self._num_calls)
        self._num_calls += 1
        discrete_objs = sorted({o for o in init_state if \
                                (not o.is_continuous and o != WORLD)})
        lits = self._parser(init_state)
        self._prepare_data_structures(lits, discrete_objs)
        if not self._is_reachable(lits, goal):
            print(f"Goal {goal} not reachable")
            return  # don't yield any skeletons
        heuristic = self._create_heuristic(discrete_objs, goal)
        queue = []
        root_node = Node(lits=lits, skeleton=[], lits_sequence=[lits])
        hq.heappush(queue, (heuristic(root_node),
                            prio_rng.uniform(),
                            root_node))
        goal_lits = set(goal.literals)
        while queue and (time.time()-start_time < timeout):
            _, _, node = hq.heappop(queue)
            if node.lits is None or goal_lits.issubset(node.lits):
                # If this skeleton satisfies the high-level goal, yield it.
                yield node.skeleton, node.lits_sequence
            if node.lits is None or not goal_lits.issubset(node.lits):
                # Otherwise, generate successors.
                for child_node in self._get_successors(node):
                    priority = (len(child_node.skeleton)+heuristic(child_node))
                    hq.heappush(queue, (priority,
                                        prio_rng.uniform(),
                                        child_node))

    def _prepare_data_structures(self, lits, discrete_objs):
        assert self._operators is not None, "Did you train yet?"
        # Cache the operators ground with the discrete objects.
        # These will have placeholders for continuous objects that will be
        # filled dynamically during search (to add a timestep index).
        ground_operators = self._ground_ops(self._operators, discrete_objs)
        # Filter out ground operators that don't satisfy static facts.
        static_preds = compute_static_preds(
            ground_operators, {lit.predicate for lit in lits})
        static_facts = {lit for lit in lits
                        if lit.predicate in static_preds}
        ground_operators = [op for op in ground_operators
                            if self._static_facts_satisfied(
                                op, static_preds, static_facts)]
        self._ground_operators = ground_operators

    def _get_successors(self, node):
        # Check whether each ground operator has satisfied the
        # DISCRETE preconditions.
        for operator in self._get_applicable_operators(
                self._ground_operators, node.lits):
            child_lits = self._apply_operator(operator, node.lits)
            child_skeleton = node.skeleton+[operator.action]
            # Add the CONTINUOUS preconditions as constraints, to be
            # dealt with by the optimizer.
            child_constraints = set()
            for lit in operator.preconds.literals:
                if lit == operator.action:
                    continue
                if any(o.is_continuous for o in lit.variables):
                    child_constraints.add(lit)
            yield Node(lits=child_lits, skeleton=child_skeleton,
                       lits_sequence=node.lits_sequence+[child_lits],
                       parent=node)

    def _create_heuristic(self, discrete_objs, goal):
        return PyperplanHAddHeuristic(self._operators, discrete_objs, goal)

    def _is_reachable(self, lits, goal):
        dr_reachables = compute_dr_reachable_lits(lits, self._ground_operators)
        return set(goal.literals).issubset(dr_reachables)

    def _ground_ops(self, operators, discrete_objs):
        """Return a grounded version of the given lifted operators,
        using the given set of objects.
        """
        ground_operators = []
        for operator in operators:
            if len(operator.preconds.literals) <= 1:
                continue
            params = operator.params
            objects = discrete_objs+[p for p in params if p.is_continuous]
            choices = list(get_object_combinations(
                objects, len(params), var_types=[p.var_type for p in params],
                allow_duplicates=True))
            for ground_params in choices:
                sigma = dict(zip(params, ground_params))
                ground_operator = self._create_ground_operator(operator, sigma)
                # If the ground operator has non-unique action arguments, do not
                # consider (this is explicitly disallowed)
                action_args = ground_operator.action.variables
                if len(action_args) != len(set(action_args)):
                    continue
                ground_operators.append(ground_operator)
        return ground_operators

    @staticmethod
    def _create_ground_operator(operator, sigma):
        """Substitute the params of the operator.
        """
        assert set(operator.params) == set(sigma)
        ground_params = [sigma[p] for p in operator.params]
        ground_action = ground_literal(operator.action, sigma)
        ground_preconds = LiteralConjunction([ground_literal(lit, sigma) \
            for lit in operator.preconds.literals])
        ground_effects = LiteralConjunction([ground_literal(lit, sigma) \
            for lit in operator.effects.literals])
        ground_operator = Operator(ground_action, operator.name, ground_params,
                                   ground_preconds, ground_effects)
        return ground_operator

    @staticmethod
    def _static_facts_satisfied(op, static_preds, static_facts):
        """Return whether the given GROUND operator could possibly satisfy
        the given static facts.
        """
        for pre in op.preconds.literals:
            assert not pre.is_negative
            if pre.predicate in static_preds and pre not in static_facts:
                return False
        return True

    @staticmethod
    def _get_applicable_operators(ground_operators, lits):
        for operator in ground_operators:
            applicable = operator.discrete_preconds.issubset(lits)
            if applicable:
                yield operator

    @staticmethod
    def _apply_operator(operator, lits):
        add_effects, delete_effects = set(), set()
        for effect in operator.effects.literals:
            assert not any(o.is_continuous for o in effect.variables)
            if effect.is_anti:
                delete_effects.add(effect.inverted_anti)
            else:
                add_effects.add(effect)
        new_lits = lits.copy()
        new_lits -= delete_effects
        new_lits |= add_effects
        return new_lits

    @staticmethod
    def _sample_act_args(state, skel_act, rng):
        # Get discrete action arguments.
        discrete_args = []
        for var in skel_act.variables:
            if not var.is_continuous:
                discrete_args.append(var)
        # Call sampler.
        cont_vals = list(skel_act.predicate.sample(rng, state, *discrete_args))
        act_args = []
        for var in skel_act.variables:
            if var.is_continuous:
                act_args.append(var.var_type(
                    f"sampled_{var.name}", cont_vals.pop(0)))
            else:
                act_args.append(var)
        assert not cont_vals
        return act_args


class ApproachFailed(Exception):
    """Exception raised when something goes wrong in an approach.
    """


class ApproachTimeout(Exception):
    """Exception raised when approach.plan() times out.
    """


class Node:
    """A node for the search over skeletons.
    """
    def __init__(self, lits, skeleton, lits_sequence, parent=None):
        self.lits = lits
        self.skeleton = skeleton
        self.lits_sequence = lits_sequence  # expected sequence of literals
        self.parent = parent
