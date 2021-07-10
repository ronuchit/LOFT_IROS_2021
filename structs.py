"""Define structs that are useful throughout the code.
"""

import itertools
import numpy as np
from pddlgym import structs as pddlgym_structs
from pddlgym.parser import Operator as PDDLGymOperator


class Type(pddlgym_structs.Type):
    """Like a PDDLGym type, but entities contain a value in addition to a name.
    """
    is_continuous = False

    def __call__(self, entity_name, entity_value=None):
        assert entity_value is None, "Discrete entities can't have a value"
        return TypedEntity.__new__(TypedEntity, entity_name, self, entity_value)  # pylint:disable=too-many-function-args


class ContinuousType(Type):
    """A continuous type
    """
    is_continuous = True
    _anonymous_count = itertools.count()

    def set_sampler(self, sampler):
        """Sets sampler function. Sampler takes in a rng and returns a value
        in the domain of this type. For example, if this type takes values
        between 0 and 1, then a sampler is `lambda rng : rng.uniform(0, 1)`.
        """
        self._sampler = sampler  # pylint:disable=attribute-defined-outside-init

    def sample(self, rng, entity_name=None):
        """Draw a sample from the sampler
        """
        assert hasattr(self, "_sampler"), "Must set sampler before sampling"
        if not entity_name:
            entity_name = f"{str(self)}{next(self._anonymous_count)}"
        return self(entity_name, self._sampler(rng))

    @property
    def dim(self):
        """Return the dimensionality of this type.
        """
        val = self.sample(np.random).value
        if isinstance(val, (list, np.ndarray)):
            return len(val)
        return 1

    def __call__(self, entity_name, entity_value=None):
        if not entity_name.startswith("?") and entity_value is None:
            raise Exception("Continuous entities must have a value, unless "
                            "they're lifted expressions (name starts with a ?)")
        return TypedEntity.__new__(TypedEntity, entity_name, self, entity_value,  # pylint:disable=too-many-function-args
                                   True)

    def __getstate__(self):
        """For pickling"""
        state = self.__dict__.copy()
        # Don't pickle sampler
        if hasattr(self, "_sampler"):
            del state["_sampler"]
        return state

    def __setstate__(self, state):
        """For pickling"""
        self.__dict__.update(state)
        # Sampler must be reset by calling set_sampler


class TypedEntity(pddlgym_structs.TypedEntity):
    """Like a PDDLGym entity, but contains a value in addition to a name.
    """
    def __new__(cls, name, var_type, value=None, is_continuous=False):
        obj = pddlgym_structs.TypedEntity.__new__(cls, name, var_type)  # pylint:disable=too-many-function-args
        if value is None:  # use name as value
            obj.value = obj.name
        else:
            obj.value = value
        obj.is_continuous = is_continuous
        return obj


class Predicate(pddlgym_structs.Predicate):
    """PDDLGym predicates but with some methods to support classification.
    """
    def __init__(self, name, arity, var_types=None, is_anti=False,
                 is_action_pred=False, sampler=None,
                 get_satisfying_args=None, holds=None):
        """The arguments that are different from pddlgym's Predicate class are:
        is_action_pred, sampler, get_satisfying_args, holds.
        - if is_action_pred=True, then don't provide get_satisfying_args / holds
        - else, get_satisfying_args is a function from state to a set of tuples
          of all arguments of this predicate that hold in that state; holds is a
          function from state and arguments to whether this predicate holds
          with those arguments in that state
          -- if get_satisfying_args is not provided, default implementation used
        - if is_action_pred=True, must provide a sampler function. A sampler is
        a function from (rng, state, *args) to a tuple of continuous argument
        values, where *args are the discrete argument values.
        """
        super().__init__(name, arity, var_types=var_types, is_anti=is_anti)
        assert is_action_pred in (True, False)
        self._is_action_pred = is_action_pred
        self._sampler = sampler
        if is_action_pred:
            assert get_satisfying_args is None and holds is None
            assert sampler is not None
            self._holds = None
            self._get_satisfying_args = None
        else:
            assert holds is not None
            assert sampler is None
            self._holds = holds
            if get_satisfying_args is not None:
                self._get_satisfying_args = get_satisfying_args
            else:
                self._get_satisfying_args = self._default_get_satisfying_args

    def __call__(self, *variables):
        var_list = list(variables)
        assert len(var_list) == self.arity
        return Literal(self, var_list)

    def get_ground_literals(self, state):
        """Get all ground literals for this predicate that hold in the
        given low-level state.
        """
        lits = set()
        for args in self._get_satisfying_args(state):
            assert self.holds(state, *args)
            lits.add(self(*args))
        return lits

    def holds(self, state, *args):
        """Return whether the ground literal self(*args) holds in the
        given low-level state.
        """
        return self._holds(state, *args)

    def _default_get_satisfying_args(self, state):
        if any(var_type.is_continuous for var_type in self.var_types):
            raise Exception("Can't use default get_satisfying_args for "
                            "predicates with continuous arguments!")
        domains = []
        for var_type in self.var_types:
            domains.append([obj for obj in state
                            if obj != WORLD and obj.var_type == var_type])
        satisfying_args = set()
        for choice in itertools.product(*domains):
            if len(choice) != len(set(choice)):
                continue  # ignore duplicate arguments
            if self.holds(state, *choice):
                satisfying_args.add(choice)
        return satisfying_args

    @property
    def positive(self):
        return self.__class__(self.name, self.arity, self.var_types,
                              is_anti=self.is_anti,
                              is_action_pred=self._is_action_pred,
                              sampler=self._sampler,
                              get_satisfying_args=self._get_satisfying_args,
                              holds=self._holds)

    @property
    def negative(self):
        raise Exception("Should never get here")

    @property
    def inverted_anti(self):
        return self.__class__(self.name, self.arity, self.var_types,
                              is_anti=(not self.is_anti),
                              is_action_pred=self._is_action_pred,
                              sampler=self._sampler,
                              get_satisfying_args=self._get_satisfying_args,
                              holds=self._holds)

    def sample(self, rng, state, *args):
        """Run the sampler. Only usable on action predicates.
        """
        assert self._is_action_pred
        return self._sampler(rng, state, *args)


class Literal(pddlgym_structs.Literal):
    """Just add a small convenience method for checking holds().
    """
    def holds(self, state):  # pylint:disable=arguments-differ
        """Return whether this ground literal holds in the given
        low-level state.
        """
        return self.predicate.holds(state, *self.variables)


class LiteralConjunction(pddlgym_structs.LiteralConjunction):
    """Just add a small convenience method for checking holds().
    """
    def holds(self, state):  # pylint:disable=arguments-differ
        """Return whether this ground literal holds in the given
        low-level state.
        """
        return all(lit.holds(state) for lit in self.literals)


class LiteralDisjunction(pddlgym_structs.LiteralDisjunction):
    """Just add a small convenience method for checking holds().
    """
    def holds(self, state):  # pylint:disable=arguments-differ
        """Return whether this ground literal holds in the given
        low-level state.
        """
        return any(lit.holds(state) for lit in self.literals)


class Operator(PDDLGymOperator):
    """Include in the operator a reference to the associated action.
    """
    def __init__(self, action, name, params, preconds, effects):
        self.action = action
        super().__init__(name, params, preconds, effects)
        # Cache discrete preconditions
        self.discrete_preconds = set()
        for lit in self.preconds.literals:
            # Ignore all preconditions involving continuous arguments
            if any(o.is_continuous for o in lit.variables):
                continue
            # Ignore action precondition
            if lit == self.action:
                continue
            self.discrete_preconds.add(lit)

# Global world object for environments
WORLD = Type("world")("world")
