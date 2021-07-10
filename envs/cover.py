"""Cover environment.
"""

import itertools
import numpy as np
import structs
from structs import WORLD
from envs import BaseEnv


class Cover(BaseEnv):
    """Cover environment.
    """
    def __init__(self, config):
        super().__init__(config)
        self._pose_count = itertools.count()
        # Types
        self.block_type = structs.Type("block")
        self.targ_type = structs.Type("targ")
        self.pose_type = structs.ContinuousType("pose")
        self.pose_type.set_sampler(lambda rng: rng.rand())
        # Predicates
        self.IsBlock = structs.Predicate(
            "IsBlock", 1, is_action_pred=False,
            get_satisfying_args=self._IsBlock_get_satisfying_args,
            holds=self._IsBlock_holds, var_types=[self.block_type])
        self.IsTarget = structs.Predicate(
            "IsTarget", 1, is_action_pred=False,
            get_satisfying_args=self._IsTarget_get_satisfying_args,
            holds=self._IsTarget_holds, var_types=[self.targ_type])
        self.Covers = structs.Predicate(
            "Covers", 2, is_action_pred=False,
            holds=self._Covers_holds,
            var_types=[self.block_type, self.targ_type])
        self.HandEmpty = structs.Predicate(
            "HandEmpty", 0, is_action_pred=False,
            holds=self._HandEmpty_holds, var_types=[])
        self.Holding = structs.Predicate(
            "Holding", 1, is_action_pred=False,
            holds=self._Holding_holds, var_types=[self.block_type])
        # Action predicates
        self.Pick = structs.Predicate(
            "Pick", 2, is_action_pred=True,
            var_types=[self.block_type, self.pose_type],
            sampler=self._pick_sampler)
        self.Place = structs.Predicate(
            "Place", 2, is_action_pred=True,
            var_types=[self.targ_type, self.pose_type],
            sampler=self._place_sampler)
        # Objects
        self._blocks = []
        self._targets = []
        for i in range(self._cf.cover_num_blocks):
            self._blocks.append(self.block_type(f"block{i}"))
        for i in range(self._cf.cover_num_targets):
            self._targets.append(self.targ_type(f"targ{i}"))

    def get_state_predicates(self):
        return {self.IsBlock, self.IsTarget,
                self.Covers, self.HandEmpty, self.Holding}

    def get_action_predicates(self):
        return {self.Pick, self.Place}

    def _get_demo_problems(self, num):
        return self._get_problems(num=num)

    def get_test_problems(self):
        return self._get_problems(num=self._cf.cover_num_test_problems)

    def _get_problems(self, num):
        problems = []
        goal1 = structs.LiteralConjunction(
            [self.Covers(self._blocks[0], self._targets[0])])
        goal2 = structs.LiteralConjunction(
            [self.Covers(self._blocks[1], self._targets[1])])
        goal3 = structs.LiteralConjunction(
            [self.Covers(self._blocks[0], self._targets[0]),
             self.Covers(self._blocks[1], self._targets[1])])
        goals = [goal1, goal2, goal3]
        for i in range(num):
            problems.append((self._create_initial_state(), goals[i%len(goals)]))
        return problems

    def _create_initial_state(self):
        state = {}
        assert len(self._cf.cover_block_widths) == len(self._blocks)
        for block, width in zip(self._blocks, self._cf.cover_block_widths):
            block_state = {}
            block_state["block"] = True
            block_state["target"] = False
            block_state["width"] = width
            block_state["grasp"] = -1
            while True:
                block_state["pose"] = self._rng.uniform(width/2, 1.0-width/2)
                if not self._any_intersection(block_state, state):
                    break
            state[block] = block_state
        assert len(self._cf.cover_target_widths) == len(self._targets)
        for targ, width in zip(self._targets, self._cf.cover_target_widths):
            targ_state = {}
            targ_state["block"] = False
            targ_state["target"] = True
            targ_state["width"] = width
            while True:
                targ_state["pose"] = self._rng.uniform(width/2, 1.0-width/2)
                if not self._any_intersection(
                        targ_state, state, larger_gap=True):
                    break
            state[targ] = targ_state
        world_state = {}
        world_state["hand"] = 0
        world_state["hand_regions"] = []
        for block in self._blocks:
            world_state["hand_regions"].append(
                (state[block]["pose"]-state[block]["width"]/2,
                 state[block]["pose"]+state[block]["width"]/2))
        for targ in self._targets:
            world_state["hand_regions"].append(
                (state[targ]["pose"]-state[targ]["width"]/10,
                 state[targ]["pose"]+state[targ]["width"]/10))
        # flat_hand_regions = np.array(world_state["hand_regions"]).flatten()
        holding_something = any(obj.var_type == "block"
                                and state[obj]["grasp"] != -1
                                for obj in state)
        world_state["flat"] = np.hstack([
            world_state["hand"],
            # flat_hand_regions,
            holding_something,
        ])
        world_state["flat_names"] = np.hstack([
            "flat:hand",
            # [f"flat:hand_regions{i}" for i in range(len(flat_hand_regions))],
            "flat:holding_something",
        ])
        state[WORLD] = world_state
        return state

    def get_next_state(self, state, action):
        assert action.predicate in [self.Pick, self.Place]
        pose = action.variables[1].value
        next_state = {k: v.copy() for k, v in state.items()}
        if action.predicate.var_types != [var.var_type for var
                                          in action.variables]:
            return next_state
        if any(hand_lb <= pose <= hand_rb
               for hand_lb, hand_rb in state[WORLD]["hand_regions"]):
            # Identify which block we're holding and which block we're above.
            held_block = None
            above_block = None
            for block in self._blocks:
                if state[block]["grasp"] != -1:
                    assert held_block is None
                    held_block = block
                block_lb = state[block]["pose"]-state[block]["width"]/2
                block_ub = state[block]["pose"]+state[block]["width"]/2
                if state[block]["grasp"] == -1 and block_lb <= pose <= block_ub:
                    assert above_block is None
                    above_block = block
            # If we're not holding anything and we're above a block, grasp it.
            if held_block is None and above_block is not None:
                grasp = pose-state[above_block]["pose"]
                next_state[WORLD]["hand"] = pose
                next_state[above_block]["pose"] = -1000  # out of the way
                next_state[above_block]["grasp"] = grasp
            # If we are holding something, place it.
            # Disallow placing on another block or in free space.
            if held_block is not None and above_block is None:
                new_pose = pose-state[held_block]["grasp"]
                dummy = {"pose": new_pose, "width": state[held_block]["width"]}
                if not self._any_intersection(
                        dummy, state, block_only=True) and \
                    any(state[targ]["pose"]-state[targ]["width"]/2
                        <= pose <=
                        state[targ]["pose"]+state[targ]["width"]/2
                        for targ in self._targets):
                    next_state[WORLD]["hand"] = pose
                    next_state[held_block]["pose"] = new_pose
                    next_state[held_block]["grasp"] = -1
        holding_something = any(obj.var_type == "block"
                                and next_state[obj]["grasp"] != -1
                                for obj in next_state)
        # Update flat world state
        next_state[WORLD]["flat"] = np.hstack([
            next_state[WORLD]["hand"],
            # np.array(next_state[WORLD]["hand_regions"]).flatten(),
            holding_something,
        ])
        next_state[WORLD]["flat_names"] = state[WORLD]["flat_names"]
        return next_state

    @staticmethod
    def _pick_sampler(rng, state, *args):
        assert len(args) == 1
        block = args[0]
        if block.var_type != "block":
            return (0.0,)
        block_state = state[block]
        world_state = state[WORLD]
        # Fail early if we're already grasping the object.
        if block_state["grasp"] != -1:
            return (0.0,)
        lb = block_state["pose"]-block_state["width"]/2
        ub = block_state["pose"]+block_state["width"]/2
        counter = 0
        while True:
            counter += 1
            if counter > 1000:
                print("WARNING: Grasp generator failed, "
                      "returning possibly invalid grasp")
                break
            cand = rng.uniform(lb, ub)
            if any(hand_lb <= cand <= hand_ub for hand_lb, hand_ub
                   in world_state["hand_regions"]):
                break
        return (cand,)

    @staticmethod
    def _place_sampler(rng, state, *args):
        assert len(args) == 1
        targ = args[0]
        if targ.var_type != "targ":
            return (0.0,)
        targ_state = state[targ]
        world_state = state[WORLD]
        lb = targ_state["pose"]-targ_state["width"]/2
        ub = targ_state["pose"]+targ_state["width"]/2
        counter = 0
        while True:
            counter += 1
            if counter > 1000:
                print("WARNING: Place generator failed, "
                      "returning possibly invalid grasp")
                break
            cand = rng.uniform(lb, ub)
            if any(hand_lb <= cand <= hand_ub for hand_lb, hand_ub
                   in world_state["hand_regions"]):
                break
        return (cand,)

    def get_random_action(self, state):
        if self._rng.choice(2):
            blocks = [o for o in state if o.var_type == self.block_type]
            block = blocks[self._rng.choice(len(blocks))]
            return self._sample_ground_act(state, self.Pick, [block])
        targs = [o for o in state if o.var_type == self.targ_type]
        targ = targs[self._rng.choice(len(targs))]
        return self._sample_ground_act(state, self.Place, [targ])

    @staticmethod
    def _IsBlock_get_satisfying_args(state):
        return {(obj,) for obj in state if obj.var_type == "block"}

    @staticmethod
    def _IsBlock_holds(state, block):
        return block in state

    @staticmethod
    def _IsTarget_get_satisfying_args(state):
        return {(obj,) for obj in state if obj.var_type == "targ"}

    @staticmethod
    def _IsTarget_holds(state, targ):
        return targ in state

    @staticmethod
    def _Covers_holds(state, block, targ):
        block_pose, block_width = state[block]["pose"], state[block]["width"]
        targ_pose, targ_width = state[targ]["pose"], state[targ]["width"]
        return (block_pose-block_width/2 <= targ_pose-targ_width/2) and \
               (block_pose+block_width/2 >= targ_pose+targ_width/2)

    @staticmethod
    def _HandEmpty_holds(state):
        for obj in state:
            if obj.var_type == "block" and state[obj]["grasp"] != -1:
                return False
        return True

    @staticmethod
    def _Holding_holds(state, block):
        return state[block]["grasp"] != -1

    def _any_intersection(self, obj_state, state, block_only=False,
                          exclude=None, larger_gap=False):
        mult = 1.5 if larger_gap else 0.5
        for other in state:
            if other == WORLD:
                continue
            if block_only and other not in self._blocks:
                continue
            if exclude is not None and other in exclude:
                continue
            other_state = state[other]
            distance = abs(other_state["pose"]-obj_state["pose"])
            if distance <= obj_state["width"]*mult+other_state["width"]*mult:
                return True
        return False
