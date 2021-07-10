"""LOFT implementation.
"""

from itertools import count
from collections import defaultdict
import heapq as hq
import functools
import numpy as np
from ndr.ndrs import NDR, NDRSet, NOISE_OUTCOME
from utils import construct_effects, preconditions_covered, \
    transition_covered, unify, lift_lit_set, prune_redundancies, \
    ndrs_to_operators
from approaches import BaseApproach

VERBOSE = False


class LOFT(BaseApproach):
    """LOFT implementation.
    """
    def __init__(self, config, simulator, state_preds, action_preds):
        super().__init__(config, simulator, state_preds, action_preds)
        self._neg_cache = None

    def _get_operators(self, data):
        demos, random_data = data
        transition_data = self._extract_transition_data(demos+random_data)
        ndrs = self._learn_all_ndrs(transition_data)
        operators = ndrs_to_operators(ndrs)
        print("Learned operators:")
        for operator in operators:
            print(operator)
        return operators

    def _learn_all_ndrs(self, transitions):
        print(f"Running BUILP on {len(transitions)} transitions")

        # Set up data.
        transitions_by_action = defaultdict(list)
        for state, action, next_state in transitions:
            # Assume that action arguments are unique
            assert len(set(action.variables)) == len(action.variables), \
                "Action arguments are assumed to be unique"
            effects = construct_effects(state, next_state)
            transition = (frozenset(state), action, frozenset(effects))
            transitions_by_action[action.predicate].append(transition)

        # Learn NDRs.
        ndrs = {}
        for act_pred in sorted(transitions_by_action):
            act_transitions = transitions_by_action[act_pred]
            act_ndrs, lifted_act = self._learn_ndrs(
                act_pred, act_transitions)
            ndrs[lifted_act] = act_ndrs

        # Recover probabilities.
        if self._cf.builp_learn_probabilities:
            new_ndrs = {}
            for lifted_act, ndr_set in ndrs.items():
                act_pred = lifted_act.predicate
                act_transitions = transitions_by_action[act_pred]
                new_ndr_set = self._recover_ndr_probabilities(
                    ndr_set, act_transitions)
                # Determinize, filtering out low probability outcomes,
                # because the rest of our system doesn't use the probs.
                # The only real point of learning the probabilities is
                # to filter out low probs.
                new_ndr_set = self._determinize_ndrs(new_ndr_set)
                new_ndrs[lifted_act] = new_ndr_set

            ndrs = new_ndrs

        return ndrs

    def _learn_ndrs(self, act_pred, transitions):
        # Initialize lifted action (assumes unique arguments)
        action_args = [f"?x{i}" for i in range(act_pred.arity)]
        lifted_action = act_pred(*action_args)

        # Partition the data by lifted effects
        lifted_effects, partitioned_transitions = \
            self._partition_transitions_by_lifted_effects(
                transitions, lifted_action)

        # Learn preconditions
        ndrs = []
        score = 0.
        default_ndr = None
        for i, positive_transitions in enumerate(partitioned_transitions):
            # Make sure score always positive
            score -= self._get_min_possible_score(len(positive_transitions))
            # Maybe don't learn empty effect operators
            if not self._cf.builp_learn_empty_effects and not lifted_effects[i]:
                continue
            # One-vs-all (negative examples are all others)
            negative_transitions = tuple(e \
                for j, group in enumerate(partitioned_transitions) \
                if i != j for e in group)
            if VERBOSE:
                print(f"Learning preconditions for action: {lifted_action}")
                print(f"with effects: {lifted_effects[i]}")
                print(f"and with {len(positive_transitions)} positives")
                print(f"and with {len(negative_transitions)} negatives")
            lifted_preconditions, pre_score = self._learn_preconditions(
                lifted_action, lifted_effects[i],
                positive_transitions, negative_transitions)

            # Make the score always positive
            score += pre_score
            assert score >= 0.

            # If one of the precondition sets is empty, this is the default NDR
            default_ndr = None
            for pre in lifted_preconditions:
                # Add NDR
                ndr = NDR(lifted_action, list(pre), [1.0, 0.0],
                          [lifted_effects[i], {NOISE_OUTCOME}])
                if not pre:
                    assert default_ndr is None, \
                        "Cannot have multiple default NDRs"
                    default_ndr = ndr
                else:
                    ndrs.append(ndr)
                if VERBOSE:
                    print("Created new NDR:")
                    print(ndr)

        # Construct NDRSet
        ndr_set = NDRSet(lifted_action, ndrs, default_ndr=default_ndr)
        if VERBOSE:
            print("Final NDR set:")
            print(ndr_set)

        return ndr_set, lifted_action

    @staticmethod
    def _partition_transitions_by_lifted_effects(transitions, lifted_action):
        # Two transitions are in the same partition if their effects
        # can be unified
        lifted_effects = []
        partitions = []

        for transition in transitions:
            _, ground_action, effects = transition
            # Verify that action can be unified, otherwise this transition
            # shouldn't be in transitions!
            assert unify(frozenset({ground_action}),
                         frozenset({lifted_action}))[0]
            partition_index = None
            for i, lifted_eff in enumerate(lifted_effects):
                # Try to unify with lifted effects
                if unify(effects | {ground_action},
                         lifted_eff | {lifted_action})[0]:
                    # Add to this partition
                    partition_index = i
                    break
            # Create a new group
            if partition_index is None:
                new_partition = [transition]
                partitions.append(new_partition)
                # Get new lifted effects
                obj_to_var = dict(zip(ground_action.variables,
                                      lifted_action.variables))
                lifted_eff = frozenset(lift_lit_set(effects, obj_to_var))
                assert lifted_eff not in lifted_effects
                lifted_effects.append(lifted_eff)
            # Add to existing group
            else:
                partitions[partition_index].append(transition)

        # There should only ever be at most one trivial effect.
        assert sum(int(not eff) for eff in lifted_effects) <= 1
        return lifted_effects, partitions

    def _learn_preconditions(self, lifted_action, lifted_effects,
                             positive_transitions, negative_transitions):
        # Keep track of all preconditions
        all_preconditions = []

        # We'll remove positives as they get covered
        remaining_positives = list(positive_transitions)

        # Keep track of the overall score (lower is better)
        score = float("inf")

        for _ in range(self._cf.builp_max_preconditions_per_effect):

            new_preconditions, new_score = self._learn_single_preconditions(
                lifted_action, lifted_effects,
                remaining_positives, negative_transitions)

            # Update remaining positives
            if new_preconditions is not None:
                new_remaining_positives = []
                for t in remaining_positives:
                    if not transition_covered(t, new_preconditions,
                                              lifted_action,
                                              lifted_effects):
                        new_remaining_positives.append(t)
                    else:
                        negative_transitions = negative_transitions + (t,)

                # assert len(new_remaining_positives) < len(remaining_positives)
                remaining_positives = new_remaining_positives

            # Terminate with failure
            if new_preconditions is None or new_score == float("inf"):
                break

            score = new_score

            # Add new rule to clauses
            assert new_preconditions not in all_preconditions, \
                "Tried to add a precondition already in the preconditions set"
            all_preconditions.append(new_preconditions)

            # Terminate with success
            if not remaining_positives:
                break

        else:
            if VERBOSE:
                print("Training did not converge, giving up.")

        assert score != -float("inf")
        return all_preconditions, score

    def _learn_single_preconditions(self, lifted_action, lifted_effects,
                                    positive_transitions, negative_transitions):
        # Initialize search
        tiebreak = count()
        queue = []
        best_score = float("inf")
        best_preconditions = None
        visited = set()
        hq.heappush(queue, (None, None, None))
        self._neg_cache = {None: set()}

        # Run the search
        for _ in range(self._cf.builp_max_search_iters):
            # Check if we've exhausted the queue
            if not queue:
                break
            # Get a rule to extend
            _, _, preconditions = hq.heappop(queue)

            # Consider different extensions
            for child in self._get_precond_successors(
                    preconditions, positive_transitions,
                    lifted_action, lifted_effects):
                # No need to reconsider children
                if child in visited:
                    continue
                # Don't consider children that are too large
                if len(child) > self._cf.builp_max_rule_size:
                    continue
                # Score the child
                child_score = self._score_preconditions(
                    child, lifted_action, lifted_effects,
                    positive_transitions, negative_transitions,
                    preconditions)
                # Update best score
                if child_score < best_score and \
                    not self._preconditions_malformed(child, lifted_action,
                                                      lifted_effects):
                    best_score = child_score
                    best_preconditions = child
                    # if VERBOSE:
                    #     print(f"Updating best preconditions ")
                    #     print(best_preconditions)
                    #     print("score:", best_score)
                # Add to the queue
                hq.heappush(queue, (child_score, next(tiebreak), child))
                visited.add(child)

        # if VERBOSE:
        #     print("\nTerminated without perfect preconditions")
        #     print(f"Best preconditions found (score {best_score}):")
        #     print(best_preconditions)

        return best_preconditions, best_score

    def _get_initial_preconditions(self, positive_transitions,
                                   lifted_action, lifted_effects):
        initial_preconditions = set()
        for state, action, effects in positive_transitions:
            state = prune_redundancies(state)
            # Hack: Remove objects that are in neither the effects
            # nor the action
            if self._cf.builp_referenced_objects_only:
                referenced_objs = set(action.variables) | \
                    {o for lit in effects for o in lit.variables}
                state = {lit for lit in state if all(o in referenced_objs\
                                                     for o in lit.variables)}
            # NOTE: this result is cached due to previous call to utils.unify()
            obj_to_var = unify(
                effects | {action}, lifted_effects | {lifted_action})[1]
            lifted_preconditions = frozenset(
                lift_lit_set(state, obj_to_var))
            initial_preconditions.add(lifted_preconditions)
        return initial_preconditions

    def _get_precond_successors(self, preconditions, positive_transitions,
                                lifted_action, lifted_effects):
        if preconditions is None:
            # Special case; only happens once at the beginning of search.
            all_initial_preconditions = self._get_initial_preconditions(
                positive_transitions, lifted_action, lifted_effects)
            for initial_preconditions in all_initial_preconditions:
                self._neg_cache[initial_preconditions] = set()
            return all_initial_preconditions
        successors = []
        preconditions = sorted(preconditions)
        for i in range(len(preconditions)):
            successor = preconditions[:i]+preconditions[i+1:]
            successors.append(frozenset(successor))
        return successors

    def _get_min_possible_score(self, num_examples):
        return -self._cf.builp_true_pos_weight*num_examples

    def _score_preconditions(self, preconditions, lifted_action, lifted_effects,
                             positive_transitions, negative_transitions,
                             parent):
        assert parent in self._neg_cache
        self._neg_cache[preconditions] = set()
        # If the rule is malformed, it explains nothing
        if self._preconditions_malformed(preconditions, lifted_action,
                                         lifted_effects):
            return 0.

        num_true_positives = 0
        num_false_positives = 0
        # Tally true positives
        for transition in positive_transitions:
            _, assignments = preconditions_covered(
                transition, preconditions, lifted_action, ret_assignments=True)
            if len(assignments) != 1:
                continue
            if transition_covered(transition, preconditions,
                                  lifted_action, lifted_effects):
                num_true_positives += 1
            else:
                num_false_positives += 1

        # Tally false positives
        num_false_positives += self._tally_false_positives(
            preconditions, negative_transitions, lifted_action, parent)

        # Get score
        score = self._cf.builp_false_pos_weight*num_false_positives - \
                self._cf.builp_true_pos_weight*num_true_positives

        # Penalize number of variables in the preconditions
        all_vars = {v for lit in preconditions for v in lit.variables}
        score += self._cf.builp_var_count_weight*len(all_vars)

        # Penalize number of preconditions
        score += self._cf.builp_rule_size_weight*len(preconditions)

        return score

    @functools.lru_cache(maxsize=100000)
    def _tally_false_positives(self, preconditions, negative_transitions,
                               lifted_action, parent):
        """Negative transitions do not change within a learning problem
        """
        num_false_positives = 0
        for i, transition in enumerate(negative_transitions):
            if i in self._neg_cache[parent]:
                num_false_positives += 1
                self._neg_cache[preconditions].add(i)
            else:
                # Here we use preconditions_covered instead of
                # transition_covered because negative examples don't care
                # whether the effects match.
                if preconditions_covered(transition, preconditions,
                                         lifted_action):
                    num_false_positives += 1
                    self._neg_cache[preconditions].add(i)
        return num_false_positives

    @staticmethod
    def _preconditions_malformed(preconditions, lifted_action, lifted_effects):
        # We require all variables in the effects to appear in
        # the preconditions / action
        effect_vars = {v for eff in lifted_effects for v in eff.variables}
        action_vars = set(lifted_action.variables)
        precondition_vars = {v for pre in preconditions for v in pre.variables}
        return not effect_vars.issubset(precondition_vars | action_vars)

    def _recover_ndr_probabilities(self, ndr_set, act_transitions):
        """Post-hoc determine probabilities for the NDRs in the
        NDR set. This involves grouping together the NDRs with
        the same preconditions and looking at the fraction of
        data where the outcomes held.
        """
        # All ndrs share the same lifted action
        lifted_action = ndr_set.action
        # Keep track of transitions that are never covered;
        # these will be the purview of the default transition
        never_covered_transitions = set(act_transitions)
        # Get a map from preconditions to all associated effects
        preconditions_to_effects = defaultdict(set)
        for ndr in ndr_set:
            pre = frozenset(ndr.preconditions)
            for eff in ndr.effects:
                # We have no use for the noise outcome
                if NOISE_OUTCOME in eff:
                    continue
                preconditions_to_effects[pre].add(frozenset(eff))
        new_ndrs = []
        # For each precondition set, we'll create one NDR
        for pre in sorted(preconditions_to_effects):
            # We'll treat the default NDR separately
            if len(pre) == 0:
                continue
            # Collect the transitions where the preconditions hold
            pre_transitions = []
            for transition in act_transitions:
                if preconditions_covered(transition, pre, lifted_action):
                    pre_transitions.append(transition)
                    never_covered_transitions.discard(transition)
            # For each possible outcome, calculate fraction
            # of data where the outcome held.
            outcomes = sorted(preconditions_to_effects[pre])
            probs = self._recover_single_ndr_probabilities(
                pre_transitions, outcomes, pre, lifted_action)
            # Note that these probabilities do not need to sum to 1
            # (example: if one outcome set is a superset of another)
            # And I don't really see any problem with just leaving
            # this unnormalized?
            new_ndr = NDR(lifted_action, list(pre), probs, outcomes,
                          require_noise_outcome=False)
            new_ndrs.append(new_ndr)
        # Handle the default NDR
        pre = frozenset()
        pre_transitions = never_covered_transitions
        # If none, default to empty
        if len(pre_transitions) == 0:
            outcomes = [frozenset()]
            probs = [1.0]
        else:
            outcomes = sorted(preconditions_to_effects[pre])
            probs = self._recover_single_ndr_probabilities(
                pre_transitions, outcomes, pre, lifted_action)
        default_ndr = NDR(lifted_action, list(pre), probs, outcomes,
                          require_noise_outcome=False)
        # Create the new set
        new_ndr_set = NDRSet(lifted_action, new_ndrs, default_ndr=default_ndr)
        return new_ndr_set

    @staticmethod
    def _recover_single_ndr_probabilities(pre_transitions, outcomes,
                                          pre, lifted_action):
        """Helper for _recover_ndr_probabilities
        """
        probs = []
        for outcome in outcomes:
            num_covered = 0
            num_not_covered = 0
            for transition in pre_transitions:
                covered, assigns = transition_covered(
                    transition, pre, lifted_action, outcome,
                    ret_assignments=True)
                if not covered or len(assigns) > 1:
                    num_not_covered += 1
                else:
                    num_covered += 1
            probs.append(num_covered / (num_covered + num_not_covered))
        return probs

    def _determinize_ndrs(self, ndr_set):
        """Filter out low probability outcomes and determinize
        remaining NDRs.
        """
        new_ndrs = []
        # All ndrs share the same lifted action
        lifted_action = ndr_set.action

        default_ndr = None
        for ndr in ndr_set:
            split_ndrs = self._determinize_ndr(ndr)
            if len(ndr.preconditions) == 0:
                assert len(split_ndrs) == 1
                assert default_ndr is None
                default_ndr = split_ndrs[0]
            else:
                new_ndrs.extend(split_ndrs)
        assert default_ndr is not None
        new_ndr_set = NDRSet(lifted_action, new_ndrs, default_ndr=default_ndr)
        return new_ndr_set

    def _determinize_ndr(self, ndr):
        """Helper for _determinize_ndrs
        """
        pre = ndr.preconditions
        lifted_action = ndr.action
        # Treat default NDR separately
        if len(pre) == 0:
            max_prob_idx = np.argmax(ndr.effect_probs)
            max_prob_outcome = ndr.effects[max_prob_idx]
            new_ndr = NDR(lifted_action, list(pre), [1.0, 0.0],
                          [max_prob_outcome, {NOISE_OUTCOME}])
            return [new_ndr]
        # Split the ndrs
        if VERBOSE:
            print("Determinizing NDR")
            print(ndr)
        split_ndrs = []
        for prob, outcome in zip(ndr.effect_probs, ndr.effects):
            # Filter out noisy things
            if prob < self._cf.effect_prob_threshold:
                if VERBOSE:
                    print("Filtering out outcome")
                    print(outcome)
                continue
            if VERBOSE:
                print("Keeping outcome")
                print(outcome)
            new_ndr = NDR(lifted_action, list(pre), [1.0, 0.0],
                          [outcome, {NOISE_OUTCOME}])
            split_ndrs.append(new_ndr)
        return split_ndrs
