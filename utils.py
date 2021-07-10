"""General utilities.
"""

import abc
import os
import functools
import tempfile
import itertools
from pyperplan.pddl.parser import Parser as PyperplanParser
from pyperplan.grounding import ground as pyperplan_ground
from pyperplan.planner import HEURISTICS as PYPERPLAN_HEURISTICS
from pyperplan.search import searchspace as pyperplan_searchspace
from pddlgym.parser import PDDLProblemParser, PDDLDomain
from pddlgym.inference import find_satisfying_assignments
from pddlgym.structs import Anti, Predicate
from pddlgym.utils import get_object_combinations
from ndr.ndrs import NOISE_OUTCOME
import structs


def angle(val1, val2):
    """Return angle between val1 and val2 in degrees.
    """
    while val1 < 0:
        val1 += 360
    while val2 < 0:
        val2 += 360
    assert 0 <= val1 <= 360
    assert 0 <= val2 <= 360
    return 180-abs(abs(val1-val2)-180)

def get_asset_path(asset_name):
    """Get absolute path to environment assets directory
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    asset_dir_path = os.path.join(dir_path, 'envs', 'assets')
    return os.path.join(asset_dir_path, asset_name)

def negate_predicate(pred):
    """Return a negation of the given predicate.
    """
    return structs.Predicate(
        "NOT-"+pred.name, pred.arity, pred.var_types,
        holds=lambda state, *args: not pred.holds(state, *args))


@functools.lru_cache(maxsize=None)
def process_lit_for_arity(lit, dummy, dummy_type):
    """Return an equivalent set of literals to `lit` that are all arity 1 or 2.
    """
    # Start by removing continuous objects
    continuous_var_idxs = [i for i, o in enumerate(lit.variables) \
                           if o.is_continuous]
    if continuous_var_idxs:
        kept_idxs = [i for i in range(len(lit.variables)) \
                     if i not in continuous_var_idxs]
        # Create new predicate with continuous removed
        kept_types = [lit.predicate.var_types[i] for i in kept_idxs]
        pred = Predicate(lit.predicate.name, len(kept_types),
                         kept_types, is_anti=lit.predicate.is_anti)
        # Create new lit
        kept_objs = [lit.variables[i] for i in kept_idxs]
        lit = pred(*kept_objs)

    if lit.predicate.arity == 0:
        # Create new predicate with dummy object
        pred = Predicate(lit.predicate.name, 1, [dummy_type],
                         is_anti=lit.predicate.is_anti)
        return {pred(dummy)}
    if lit.predicate.arity <= 2:
        return {lit}
    processed_lits = set()
    for i in range(lit.predicate.arity-1):
        obj_i = lit.variables[i]
        for j in range(i+1, lit.predicate.arity):
            obj_j = lit.variables[j]
            pred_ij = Predicate(f"{lit.predicate.name}_{i}_{j}",
                                2, [obj_i.var_type, obj_j.var_type],
                                is_anti=lit.predicate.is_anti)
            lit_ij = pred_ij(obj_i, obj_j)
            processed_lits.add(lit_ij)
    return processed_lits

def ground_literal(lifted_lit, assignments):
    """Given a lifted literal, create a ground
    literal with the assignments mapping vars to
    objects.
    Parameters
    ----------
    lifted_lit : Literal
    assignments : { TypedEntity : TypedEntity }
        Vars to objects.
    Returns
    -------
    ground_lit : Literal
    """
    ground_vars = []
    for v in lifted_lit.variables:
        arg = assignments[v]
        ground_vars.append(arg)
    return lifted_lit.predicate(*ground_vars)

def wrap_goal_literal(x):
    """Helper for converting a state to required input representation
    """
    if isinstance(x, Predicate):
        return Predicate("WANT"+x.name, x.arity, var_types=x.var_types,
                         is_negative=x.is_negative, is_anti=x.is_anti)
    new_predicate = wrap_goal_literal(x.predicate)
    return new_predicate(*x.variables)


def reverse_binary_literal(x):
    """Helper for converting a state to required input representation
    """
    if isinstance(x, Predicate):
        assert x.arity == 2
        return Predicate("REV"+x.name, x.arity, var_types=x.var_types,
                         is_negative=x.is_negative, is_anti=x.is_anti)
    new_predicate = reverse_binary_literal(x.predicate)
    variables = list(x.variables)
    assert len(variables) == 2
    return new_predicate(*variables[::-1])


def get_all_discrete_literals(pred, objects):
    """Get all ways to instantiate the discrete arguments of the
    given predicate with the given objects. Is type-aware.
    """
    discretes = [var for var in pred.var_types
                 if not var.is_continuous]
    choices = list(get_object_combinations(
        objects, len(discretes), allow_duplicates=True,
        var_types=discretes))
    for grounding in choices:
        sigma = dict(zip(discretes, grounding))
        action_vars = []
        for var in pred.var_types:
            if var in sigma:
                assert not var.is_continuous
                action_vars.append(sigma[var])
            else:
                assert var.is_continuous
                action_vars.append(var(f"?{var}"))
        yield pred(*action_vars)


def compute_static_preds(operators, predicates):
    """Compute the static predicates under the given operators.
    """
    static_preds = set()
    for pred in predicates:
        if any(op_changes_predicate(op, pred) for op in operators):
            continue
        static_preds.add(pred)
    return static_preds


def op_changes_predicate(op, pred):
    """Helper method for _compute_static_preds.
    """
    for lit in op.effects.literals:
        assert not lit.is_negative
        if lit.is_anti:
            eff_pred = lit.inverted_anti.predicate
        else:
            eff_pred = lit.predicate
        if eff_pred == pred:
            return True
    return False


def compute_dr_reachable_lits(lits, operators):
    """Compute the literals reachable from the given ones under the
    given operators.
    """
    reachable_lits = set(lits)
    while True:
        fixed_point_reached = True
        for op in operators:
            positive_preconditions = {lit for lit in op.preconds.literals
                                      if not lit.is_negative
                                      and lit != op.action
                                      and not any(v.is_continuous
                                                  for v in lit.variables)}
            if positive_preconditions.issubset(reachable_lits):
                positive_effects = {lit for lit in op.effects.literals
                                    if not lit.is_anti}
                for new_reachable_lit in positive_effects - reachable_lits:
                    fixed_point_reached = False
                    reachable_lits.add(new_reachable_lit)
        if fixed_point_reached:
            break
    return reachable_lits

def construct_effects(state, next_state):
    """Create effects from a transition of states, ignoring continuous lits
    """
    effects = set()
    for lit in state - next_state:
        if not any(v.is_continuous for v in lit.variables):
            effects.add(Anti(lit))
    for lit in next_state - state:
        if not any(v.is_continuous for v in lit.variables):
            effects.add(lit)
    return effects


@functools.lru_cache(maxsize=100000)
def preconditions_covered(transition, preconditions, lifted_action,
                          ret_assignments=False):
    """Checks whether the preconditions hold in the given transition
    (ignoring effects/next state)
    """
    state, action, _ = transition
    kb = state | {action}
    conditions = preconditions | {lifted_action}
    assignments = find_satisfying_assignments(
        kb, conditions, allow_redundant_variables=True,
        max_assignment_count=float("inf"))
    covered = (len(assignments) > 0)
    if ret_assignments:
        return covered, assignments
    return covered


def effects_covered(lifted_effects, ground_effects, pre_assignments,
                    ret_assignments=False):
    """Checks whether the effects in the transition hold
    pre_assignments should come from calling preconditions_covered() with
    ret_assignments=True
    """
    valid_assignments = []
    for assignment in pre_assignments:
        # Substitute variables of lifted effects.
        poss_ground_effects = substitute(lifted_effects, assignment)
        # Cancel out Anti's.
        to_remove = set()
        for eff in poss_ground_effects:
            if eff.is_anti and eff.inverted_anti in poss_ground_effects:
                to_remove.add(eff)
                to_remove.add(eff.inverted_anti)
        for rem in to_remove:
            poss_ground_effects.remove(rem)
        # Check if predicted effects match true ones.
        if poss_ground_effects == ground_effects:
            if not ret_assignments:
                return True
            valid_assignments.append(assignment)
    if not ret_assignments:
        return False
    return len(valid_assignments) > 0, valid_assignments


@functools.lru_cache(maxsize=100000)
def transition_covered(transition, preconditions, lifted_action,
                       lifted_effects, ret_assignments=False):
    """Checks whether the lifted rule covers the transition
    """
    covered, assignments = preconditions_covered(
        transition, preconditions, lifted_action, ret_assignments=True)
    if not covered:
        return False
    _, _, ground_effects = transition
    result = effects_covered(lifted_effects, ground_effects, assignments)
    if ret_assignments:
        return result, assignments
    return result


def substitute(literals, assignments):
    """Substitute variables in literals with given dict of assignments.
    """
    new_literals = set()
    for lit in literals:
        new_vars = []
        for var in lit.variables:
            assert var in assignments
            new_vars.append(assignments[var])
        new_literals.add(lit.predicate(*new_vars))
    return new_literals


@functools.lru_cache(maxsize=100000)
def unify(lits1, lits2):
    """Return a tuple of (whether the given frozensets lits1 and lits2 can be
    unified, the mapping if the first return value is True).
    Also returns the mapping.
    """
    # Terminate quickly if there is a mismatch between lits
    predicate_order1 = [lit.predicate for lit in sorted(lits1)]
    predicate_order2 = [lit.predicate for lit in sorted(lits2)]
    if predicate_order1 != predicate_order2:
        return False, None

    assignments = find_satisfying_assignments(
        lits2, lits1, allow_redundant_variables=False)
    if not assignments:
        return False, None
    return True, assignments[0]

def lift_lit_set(literal_set, obj_to_var):
    """Lift the given literal set. obj_to_var is a dict of variable
    assignments that you want to force. It can be empty.
    """
    if obj_to_var:
        next_var_id = max(int(v[2:]) for v in obj_to_var.values())+1
    else:
        next_var_id = 0
    var_count = itertools.count(next_var_id)
    for lit in sorted(literal_set):
        for obj in sorted(lit.variables):
            if obj not in obj_to_var:
                obj_to_var[obj] = obj.var_type(f"?x{next(var_count)}")
    return {ground_literal(lit, obj_to_var) for lit in literal_set}

def prune_redundancies(formula):
    """Relies heavliy on the assumption that redundant variables are allowed.
    """
    # Get representations to distinguish variables
    all_variables = {v for lit in formula for v in lit.variables}
    var_to_lits = {v: sorted([lit for lit in formula if v in lit.variables])
                   for v in all_variables}
    var_to_lifted_id = {v: _compute_lifted_variable_id(v, var_to_lits)
                        for v in all_variables}
    # Keep only one variable per group
    vars_to_keep = set()
    kept_lifted_ids = set()
    for v in sorted(all_variables):
        lifted_id = var_to_lifted_id[v]
        if lifted_id not in kept_lifted_ids:
            vars_to_keep.add(v)
            kept_lifted_ids.add(lifted_id)
    # Prune literals
    pruned_formula = {lit for lit in formula if all(v in vars_to_keep
                                                    for v in lit.variables)}
    return pruned_formula


def _compute_lifted_variable_id(main_v, var_to_lits):
    """Helper for prune_redundancies.
    """
    var_to_num = {main_v : 0}
    queue = sorted(var_to_lits[main_v])
    all_visited_lits = set()
    while queue:
        lit = queue.pop()
        all_visited_lits.add(lit)
        for v in lit.variables:
            if v not in var_to_num:
                var_to_num[v] = max(var_to_num.values()) + 1
                for new_lit in var_to_lits[v]:
                    if any(v_prime not in var_to_num
                           for v_prime in new_lit.variables):
                        queue.append(new_lit)
    lifted_lits = set()
    for lit in all_visited_lits:
        lifted_lit = (lit.predicate, tuple(var_to_num[a]
                                           for a in lit.variables))
        lifted_lits.add(lifted_lit)
    return frozenset(lifted_lits)


def extract_preds_and_types_from_ops(operators,
                                     ignore_action_lits=True):
    """Pull out predicates and types used in operators
    """
    preds = {}
    types = {}
    # Extract the predicates and types from the operators.
    for op in operators:
        for pre in op.preconds.literals:
            assert not pre.is_negative
            if ignore_action_lits and pre == op.action:
                # Ignore action literal.
                continue
            if any(var_type.is_continuous
                   for var_type in pre.predicate.var_types):
                # Ignore continuous preconditions.
                continue
            for var_type in pre.predicate.var_types:
                types[str(var_type)] = var_type
            preds[pre.predicate.name] = pre.predicate
        for eff in op.effects.literals:
            if eff.is_anti:
                eff = eff.inverted_anti
            assert not any(var_type.is_continuous
                           for var_type in eff.predicate.var_types)
            for var_type in eff.predicate.var_types:
                types[str(var_type)] = var_type
            preds[eff.predicate.name] = eff.predicate
    return preds, types


def make_domain(operators, operators_as_actions=False,
                ignore_action_lits=True):
    """Create a PDDLDomain object from the given operators.
    """
    # Extract preds and types
    preds, types = extract_preds_and_types_from_ops(
        operators, ignore_action_lits=ignore_action_lits)
    # Remove continuous predicates and possibly actions from ops
    new_operators = []
    for op in operators:
        to_delete = set()
        for pre in op.preconds.literals:
            assert not pre.is_negative
            if ignore_action_lits and pre == op.action:
                # Ignore action literal.
                to_delete.add(pre)
                continue
            if any(var_type not in types.values()
                   for var_type in pre.predicate.var_types):
                # Ignore continuous preconditions.
                to_delete.add(pre)
                continue
        precond_lits = op.preconds.literals.copy()
        for pre in to_delete:
            # Remove continuous preconditions and action literal.
            precond_lits.remove(pre)
        for eff in op.effects.literals:
            if eff.is_anti:
                eff = eff.inverted_anti
            assert not any(var_type not in types.values()
                           for var_type in eff.predicate.var_types)
        # Remove operator continuous parameters.
        params = [p for p in op.params if p.var_type in types.values()]
        new_operators.append(structs.Operator(
            op.action, op.name, params,
            structs.LiteralConjunction(precond_lits), op.effects))
    # Make domain object.
    operators_dict = {o.name: o for o in new_operators}
    actions = {o.action.predicate for o in new_operators}
    domain = PDDLDomain(domain_name="dummydomain", types=types,
                        predicates=preds, operators=operators_dict,
                        type_hierarchy={}, actions=actions,
                        operators_as_actions=operators_as_actions)
    return domain


def ndrs_to_operators(all_ndrs, include_empty_effects=False,
                      effect_threshold=0.):
    """Determinize the given NDRs into a list of operators.
    """
    operators = []
    for action in sorted(all_ndrs):
        ndr_set = all_ndrs[action]
        cnt = 0
        for ndr in ndr_set:
            for effects, effect_prob in zip(ndr.effects, ndr.effect_probs):
                # We'll never want to use an operator with noisy effects
                if NOISE_OUTCOME in effects:
                    continue
                if not include_empty_effects and \
                    len(effects) == 0:
                    continue
                if effect_prob < effect_threshold:
                    continue
                # Create new operator
                name = f"{ndr.action.predicate.name}{cnt}"
                cnt += 1
                preconds = structs.LiteralConjunction(
                    sorted(ndr.preconditions)+[action])
                params = sorted({v for lit in preconds.literals
                                 for v in lit.variables})
                effects = structs.LiteralConjunction(sorted(effects))
                operator = structs.Operator(action, name, params,
                                            preconds, effects)
                operators.append(operator)
    return operators


class PyperplanBaseHeuristic:
    """Base class for Pyperplan heuristics.
    """
    def __init__(self, operators, objects, goal):
        self._operators = operators
        self._objects = objects
        self._goal = goal
        # Remember predicates and types used in operators
        self._preds, self._types = extract_preds_and_types_from_ops(
            operators)
        # Filter out objects with unrecognized types
        objects = {o for o in objects if o.var_type in self._types}
        # Raise error if goal has unknown predicate
        assert all(lit.predicate.name in self._preds for lit in goal.literals)
        self._heuristic_name = self._get_heuristic_name()
        self._heuristic = None
        domain = make_domain(self._operators,
                             operators_as_actions=False)
        self._domain_fname = tempfile.NamedTemporaryFile(delete=False).name
        domain.write(self._domain_fname)

    @abc.abstractmethod
    def _get_heuristic_name(self):
        raise NotImplementedError("Override me!")

    def __call__(self, node):
        if self._heuristic is None:
            self._heuristic = self._get_heuristic_fn(node.lits)
        return self._heuristic(frozenset(node.lits))

    def _get_heuristic_fn(self, lits, cache_maxsize=10000):
        # Make problem file and set up Pyperplan objects.
        lits = {lit for lit in lits if lit.predicate.name in self._preds}
        try:
            problem_fname = tempfile.NamedTemporaryFile(delete=False).name
            PDDLProblemParser.create_pddl_file(
                problem_fname, self._objects, lits, "dummyproblem",
                "dummydomain", self._goal, fast_downward_order=True)
            parser = PyperplanParser(self._domain_fname, problem_fname)
            pyperplan_domain = parser.parse_domain()
            pyperplan_problem = parser.parse_problem(pyperplan_domain)
        finally:
            try:
                os.remove(self._domain_fname)
                os.remove(problem_fname)
            except FileNotFoundError:
                pass
        task = pyperplan_ground(pyperplan_problem)
        heuristic = PYPERPLAN_HEURISTICS[self._heuristic_name](task)

        @functools.lru_cache(cache_maxsize)
        def _call_heuristic(cur_lits):
            cur_lits = {lit for lit in cur_lits
                        if lit.predicate.name in self._preds}
            cur_objects = {obj for lit in cur_lits for obj in lit.variables}
            assert cur_objects.issubset(self._objects), \
                "If your object set changes, make a new heuristic object"
            state = task.facts & {lit.pddl_str().lower() for lit in cur_lits}
            node = pyperplan_searchspace.make_root_node(state)
            h = heuristic(node)
            return h
        return _call_heuristic


class PyperplanHAddHeuristic(PyperplanBaseHeuristic):
    """Pyperplan's hadd heuristic.
    """
    def _get_heuristic_name(self):
        return "hadd"


class PyperplanHFFHeuristic(PyperplanBaseHeuristic):
    """Pyperplan's hff heuristic.
    """
    def _get_heuristic_name(self):
        return "hff"


class EnvironmentFailure(Exception):
    """Exception raised when something goes wrong in an environment.
    """


class BiRRT:
    """Bidirectional rapidly-exploring random tree.
    """
    def __init__(self, sample_fn, extend_fn, collision_fn, distance_fn,
                 rng, num_attempts=10, num_iters=100, smooth_amt=50):
        self._sample_fn = sample_fn
        self._extend_fn = extend_fn
        self._collision_fn = collision_fn
        self._distance_fn = distance_fn
        self._rng = rng
        self._num_attempts = num_attempts
        self._num_iters = num_iters
        self._smooth_amt = smooth_amt

    def query(self, pt1, pt2):
        """Query the BiRRT, to get a collision-free path from pt1 to pt2.
        """
        if self._collision_fn(pt1) is not None or \
           self._collision_fn(pt2) is not None:
            return None
        direct_path = self._try_direct_path(pt1, pt2)
        if direct_path is not None:
            return direct_path
        for _ in range(self._num_attempts):
            path = self._rrt_connect(pt1, pt2)
            if path is not None:
                return self._smooth_path(path)
        return None

    def query_ignore_collisions(self, pt1, pt2):
        """Query the BiRRT but ignore collisions, thus returning a direct path.
        Also return the information for the first collision encountered.
        """
        path = [pt1]
        collision_info = self._collision_fn(pt1)
        if collision_info is None:
            collision_info = self._collision_fn(pt2)
        for newpt in self._extend_fn(pt1, pt2):
            if collision_info is None:
                collision_info = self._collision_fn(newpt)
            path.append(newpt)
        return path, collision_info

    def _try_direct_path(self, pt1, pt2):
        path = [pt1]
        for newpt in self._extend_fn(pt1, pt2):
            if self._collision_fn(newpt) is not None:
                return None
            path.append(newpt)
        return path

    def _rrt_connect(self, pt1, pt2):
        root1, root2 = TreeNode(pt1), TreeNode(pt2)
        nodes1, nodes2 = [root1], [root2]
        for _ in range(self._num_iters):
            if len(nodes1) > len(nodes2):
                nodes1, nodes2 = nodes2, nodes1
            samp = self._sample_fn(pt1)
            nearest1 = min(nodes1, key=lambda n, samp=samp:
                           self._distance_fn(n.data, samp))
            for newpt in self._extend_fn(nearest1.data, samp):
                if self._collision_fn(newpt) is not None:
                    break
                nearest1 = TreeNode(newpt, parent=nearest1)
                nodes1.append(nearest1)
            nearest2 = min(nodes2, key=lambda n, nearest1=nearest1:
                           self._distance_fn(n.data, nearest1.data))
            for newpt in self._extend_fn(nearest2.data, nearest1.data):
                if self._collision_fn(newpt) is not None:
                    break
                nearest2 = TreeNode(newpt, parent=nearest2)
                nodes2.append(nearest2)
            else:
                path1 = nearest1.path_from_root()
                path2 = nearest2.path_from_root()
                if path1[0] != root1:
                    path1, path2 = path2, path1
                path = path1[:-1]+path2[::-1]
                return [node.data for node in path]
        return None

    def _smooth_path(self, path):
        for _ in range(self._smooth_amt):
            if len(path) <= 2:
                return path
            i = self._rng.randint(0, len(path)-1)
            j = self._rng.randint(0, len(path)-1)
            if abs(i-j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = list(self._extend_fn(path[i], path[j]))
            if len(shortcut) < j-i and \
               all(self._collision_fn(pt) is None for pt in shortcut):
                path = path[:i+1]+shortcut+path[j+1:]
        return path


class TreeNode:
    """TreeNode definition.
    """
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent

    def path_from_root(self):
        """Return the path from the root to this node.
        """
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]
