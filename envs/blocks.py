"""Blocks environment.
"""

import numpy as np
import pybullet as p
import structs
from structs import WORLD
from envs import BaseEnv
from envs.pybullet_utils import get_kinematic_chain, inverse_kinematics, \
    get_asset_path, aabb_overlap


class Blocks(BaseEnv):
    """Blocks environment.
    """
    block_size = 0.06
    table_height = 0.2
    x_lb = 1.3
    x_ub = 1.4
    y_lb = 0.15
    y_ub = 0.85

    def __init__(self, config):
        super().__init__(config)
        # Types
        self.block_type = structs.Type("block")
        self.pose_type = structs.ContinuousType("pose")
        self.pose_type.set_sampler(lambda rng: rng.uniform(-10, 10, size=3))
        # Predicates
        self.IsBlock = structs.Predicate(
            "IsBlock", 1, is_action_pred=False,
            holds=self._IsBlock_holds, var_types=[self.block_type])
        self.On = structs.Predicate(
            "On", 2, is_action_pred=False,
            holds=self._On_holds,
            var_types=[self.block_type, self.block_type])
        self.OnTable = structs.Predicate(
            "OnTable", 1, is_action_pred=False,
            holds=self._OnTable_holds, var_types=[self.block_type])
        self.Clear = structs.Predicate(
            "Clear", 1, is_action_pred=False,
            holds=self._Clear_holds, var_types=[self.block_type])
        self.Holding = structs.Predicate(
            "Holding", 1, is_action_pred=False,
            holds=self._Holding_holds, var_types=[self.block_type])
        self.HandEmpty = structs.Predicate(
            "HandEmpty", 0, is_action_pred=False,
            holds=self._HandEmpty_holds, var_types=[])
        # Action predicates
        self.Pick = structs.Predicate(
            "Pick", 1, is_action_pred=True,
            sampler=self._pick_sampler,
            var_types=[self.block_type])
        self.PutOnTable = structs.Predicate(
            "PutOnTable", 1, is_action_pred=True,
            sampler=self._put_on_table_sampler,
            var_types=[self.pose_type])
        self.Stack = structs.Predicate(
            "Stack", 1, is_action_pred=True,
            sampler=self._stack_sampler,
            var_types=[self.block_type])
        self._obj_to_pybullet_obj = None
        self._initial_pybullet_setup(with_gui=False)

    def get_state_predicates(self):
        return {self.IsBlock, self.On, self.OnTable, self.Clear, self.Holding,
                self.HandEmpty}

    def get_action_predicates(self):
        return {self.Pick, self.PutOnTable, self.Stack}

    def _get_problems(self, num_problems, all_num_objs):
        problems = []
        for i in range(num_problems):
            j = i % len(all_num_objs)
            num_objs = all_num_objs[j]
            problems.append(self._create_problem(num_objs))
        return problems

    def _get_demo_problems(self, num):
        return self._get_problems(
            num, self._cf.blocks_demo_num_objs)

    def get_test_problems(self):
        return self._get_problems(
            self._cf.blocks_num_test_problems,
            self._cf.blocks_test_num_objs)

    def get_random_action(self, state):
        objs = list(sorted(state.keys()))
        objs.remove(WORLD)
        act_pred = self._rng.randint(3)
        if act_pred == 0:  # Pick
            obj = objs[self._rng.choice(len(objs))]
            return self._sample_ground_act(state, self.Pick, [obj])
        if act_pred == 1:  # Put on table
            return self._sample_ground_act(state, self.PutOnTable, [])
        if act_pred == 2:  # Stack
            obj = objs[self._rng.choice(len(objs))]
            return self._sample_ground_act(state, self.Stack, [obj])
        raise Exception("Can never reach here")

    def _create_problem(self, num_objs):
        # Sample piles
        piles = self._sample_initial_piles(num_objs)
        # Create state from piles
        state = self._sample_state_from_piles(piles)
        while True: # repeat until goal is not satisfied
            # Sample goal
            goal = self._sample_goal_from_piles(num_objs, piles)
            if not goal.holds(state):
                break
        return state, goal

    def _sample_goal_from_piles(self, num_objs, piles):
        # Sample goal pile that is different from initial
        while True:
            goal_piles = self._sample_initial_piles(num_objs)
            if goal_piles != piles:
                break
        # Create literal goal from piles
        goal_lits = []
        for pile in goal_piles:
            goal_lits.append(self.OnTable(pile[0]))
            if len(pile) == 1:
                continue
            for block1, block2 in zip(pile[1:], pile[:-1]):
                goal_lits.append(self.On(block1, block2))
        return structs.LiteralConjunction(goal_lits)

    def _sample_initial_piles(self, num_objs):
        piles = []
        for block_num in range(num_objs):
            block = self.block_type(f"block{block_num}")
            # If coin flip, start new pile
            if block_num == 0 or self._rng.uniform() < 0.2:
                piles.append([])
            # Add block to pile
            piles[-1].append(block)
        return piles

    def _sample_state_from_piles(self, piles):
        state = {}
        # Create world state
        world_state = {}
        world_state["cur_grip"] = [0.9, 0.3, 0.3]
        world_state["cur_holding"] = None
        world_state["piles"] = piles
        block_to_pile_idx = \
            self._create_block_to_pile_idx(piles)
        world_state["block_to_pile_idx"] = block_to_pile_idx
        state[WORLD] = world_state
        # Sample pile (x, y)s
        pile_to_xy = {}
        for i in range(len(piles)):
            (x, y) = self._sample_initial_pile_xy(
                self._rng, pile_to_xy.values())
            pile_to_xy[i] = (x, y)
        # Create block states
        for block, pile_idx in block_to_pile_idx.items():
            x, y = pile_to_xy[pile_idx[0]]
            z = self.table_height + self.block_size * (0.5 + pile_idx[1])
            block_state = {"pose" : [x, y, z], "held" : False}
            state[block] = block_state
        # Update world flat features
        self._update_world_flat(state)
        return state

    @staticmethod
    def _update_world_flat(state):
        """Set flat world features
        """
        holding_something = state[WORLD]["cur_holding"] is not None
        flat_feats = np.array([int(holding_something)])
        state[WORLD]["flat"] = flat_feats
        state[WORLD]["flat_names"] = np.array(["flat:holding_something"])

    @classmethod
    def _sample_initial_pile_xy(cls, rng, existing_xys, max_attempts=1000):
        for _ in range(max_attempts):
            x, y = cls._sample_xy(rng)
            if cls._table_xy_is_clear(x, y, existing_xys):
                return (x, y)
        raise Exception("Exhausted max attempts in sampling new x,y")

    @staticmethod
    def _sample_xy(rng):
        x = rng.uniform(Blocks.x_lb, Blocks.x_ub)
        y = rng.uniform(Blocks.y_lb, Blocks.y_ub)
        return (x, y)

    @classmethod
    def _table_xy_is_clear(cls, x, y, existing_xys):
        if all(abs(x-other_x) > 2*cls.block_size
               for other_x, _ in existing_xys):
            return True
        if all(abs(y-other_y) > 2*cls.block_size
               for _, other_y in existing_xys):
            return True
        return False

    @staticmethod
    def _create_block_to_pile_idx(piles):
        block_to_pile_idx = {}
        for i, pile in enumerate(piles):
            for j, block in enumerate(pile):
                assert block not in block_to_pile_idx
                block_to_pile_idx[block] = (i, j)
        return block_to_pile_idx

    def get_next_state(self, state, action):
        if action.predicate.var_types != [var.var_type for var
                                          in action.variables]:
            next_state = self._copy_state(state)
            return next_state
        # Pick
        if action.predicate == self.Pick:
            return self._get_next_state_pick(state, action)
        # PutOnTable
        if action.predicate == self.PutOnTable:
            return self._get_next_state_put_on_table(state, action)
        # Stack
        if action.predicate == self.Stack:
            return self._get_next_state_stack(state, action)
        raise Exception(f"Unexpected action: {action}")

    def _get_next_state_pick(self, state, action):
        assert action.predicate == self.Pick
        next_state = self._copy_state(state)
        # Can only pick if hand is empty
        if state[WORLD]["cur_holding"] is not None:
            return next_state
        # Can only pick if object is at top of pile
        obj = action.variables[0]
        pile_idx, in_pile_idx = state[WORLD]["block_to_pile_idx"][obj]
        if in_pile_idx != len(state[WORLD]["piles"][pile_idx])-1:
            return next_state
        # Execute pick
        next_state[WORLD]["cur_holding"] = obj
        next_state[WORLD]["block_to_pile_idx"][obj] = None
        next_state[WORLD]["piles"][pile_idx].pop(-1)
        next_state[WORLD]["cur_grip"] = state[obj]["pose"]
        next_state[obj]["held"] = True
        # Update world flat features
        self._update_world_flat(next_state)
        return next_state

    def _get_next_state_put_on_table(self, state, action):
        assert action.predicate == self.PutOnTable
        next_state = self._copy_state(state)
        # Can only put on table if holding
        holding_obj = next_state[WORLD]["cur_holding"]
        if holding_obj is None:
            return next_state
        # Can only put on table if pose is clear
        x, y, _ = action.variables[0].value
        obj_poses = [state[o]['pose'] for o in state if o != WORLD]
        existing_xys = [(p[0], p[1]) for p in obj_poses]
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        # Execute put on table
        pile_idx = len(state[WORLD]["piles"])
        new_pose = [x, y, self.table_height+0.5*self.block_size]
        next_state[WORLD]["cur_holding"] = None
        next_state[WORLD]["block_to_pile_idx"][holding_obj] = \
            (pile_idx, 0)
        next_state[WORLD]["piles"].append([holding_obj])
        next_state[WORLD]["cur_grip"] = new_pose
        next_state[holding_obj]["pose"] = new_pose
        next_state[holding_obj]["held"] = False
        # Update world flat features
        self._update_world_flat(next_state)
        return next_state

    def _get_next_state_stack(self, state, action):
        assert action.predicate == self.Stack
        next_state = self._copy_state(state)
        # Can only stack if holding
        holding_obj = next_state[WORLD]["cur_holding"]
        if holding_obj is None:
            return next_state
        # Can't stack on the object that we're holding
        obj = action.variables[0]
        if holding_obj == obj:
            return next_state
        # Can only stack if target is clear
        pile_idx, in_pile_idx = state[WORLD]["block_to_pile_idx"][obj]
        if in_pile_idx != len(state[WORLD]["piles"][pile_idx])-1:
            return next_state
        # Execute stack
        x, y, z = state[obj]["pose"]
        new_pose = [x, y, z + self.block_size]
        next_state[WORLD]["cur_holding"] = None
        next_state[WORLD]["block_to_pile_idx"][holding_obj] = \
            (pile_idx, in_pile_idx+1)
        next_state[WORLD]["piles"][pile_idx].append(holding_obj)
        next_state[WORLD]["cur_grip"] = new_pose
        next_state[holding_obj]["pose"] = new_pose
        next_state[holding_obj]["held"] = False
        # Update world flat features
        self._update_world_flat(next_state)
        return next_state

    @staticmethod
    def _IsBlock_holds(state, obj):
        return obj in state and obj.var_type == "block"

    @staticmethod
    def _On_holds(state, obj1, obj2):
        pile_idx1 = state[WORLD]['block_to_pile_idx'][obj1]
        pile_idx2 = state[WORLD]['block_to_pile_idx'][obj2]
        # One of the blocks is held
        if pile_idx1 is None or pile_idx2 is None:
            return False
        return (pile_idx1[0] == pile_idx2[0] and
                pile_idx1[1] == pile_idx2[1] + 1)

    @staticmethod
    def _Holding_holds(state, obj):
        holding = (state[WORLD]["cur_holding"] is not None and
                   state[WORLD]["cur_holding"] == obj)
        assert holding == (state[WORLD]['block_to_pile_idx'][obj] \
                           is None)
        assert holding == state[obj]["held"]
        return holding

    @staticmethod
    def _OnTable_holds(state, obj):
        pile_idx = state[WORLD]['block_to_pile_idx'][obj]
        return pile_idx is not None and pile_idx[1] == 0

    @staticmethod
    def _Clear_holds(state, obj):
        pile_idx = state[WORLD]['block_to_pile_idx'][obj]
        if pile_idx is None:
            return False
        pile_size = len(state[WORLD]['piles'][pile_idx[0]])
        return pile_idx[1] == pile_size-1

    @staticmethod
    def _HandEmpty_holds(state):
        return state[WORLD]["cur_holding"] is None

    @staticmethod
    def _pick_sampler(rng, state, *args):
        del rng  # unused
        del state  # unused
        del args  # unused
        return tuple()

    @classmethod
    def _put_on_table_sampler(cls, rng, state, *args):
        del state  # unused
        del args  # unused
        (x, y) = Blocks._sample_xy(rng)
        pose = [x, y, Blocks.table_height + \
                0.5*Blocks.block_size]
        return (pose,)

    @staticmethod
    def _stack_sampler(rng, state, *args):
        del rng  # unused
        del state  # unused
        del args  # unused
        return tuple()

    def _state_has_forbidden_collisions(self, state, held_obj_constraint=None,
                                        interacting_objs=None):
        # Set state
        self._reset_pybullet_from_state(
            state, held_obj_constraint=held_obj_constraint)
        ## Get aabbs
        # Get object aabbs
        obj_to_aabb = {}
        for obj, py_obj in self._obj_to_pybullet_obj.items():
            aabb = p.getAABB(
                py_obj, physicsClientId=self._physics_client_id)
            obj_to_aabb[obj] = aabb
        # Get table aabb
        table_aabb = p.getAABB(
            self._table_id, physicsClientId=self._physics_client_id)
        # Get robot aabbs
        robot_aabbs = []
        for idx in range(p.getNumJoints(
                self._fetch_id,
                physicsClientId=self._physics_client_id)):
            aabb = p.getAABB(self._fetch_id, idx,
                             physicsClientId=self._physics_client_id)
            robot_aabbs.append(aabb)
        # Check collisions between robot and non-held object
        for obj, aabb in obj_to_aabb.items():
            if obj in interacting_objs:
                continue
            for robot_aabb in robot_aabbs:
                if aabb_overlap(aabb, robot_aabb):
                    return "collision"
        # Check collisions between robot and table
        for robot_aabb in robot_aabbs:
            if aabb_overlap(robot_aabb, table_aabb):
                return "collision"
        return None

    def _apply_pick_constraint(self, held_obj_tf):
        base_link = np.r_[p.getLinkState(
            self._fetch_id, self._ee_id,
            physicsClientId=self._physics_client_id)[:2]]
        _, transf = held_obj_tf
        obj_loc, orn = p.multiplyTransforms(
            base_link[:3], base_link[3:], transf[0], transf[1])
        return obj_loc, orn

    @staticmethod
    def _get_camera_params():
        camera_distance = 2
        yaw = 90
        pitch = -5
        camera_target = [1.05, 0.5, 0.42]
        return camera_distance, yaw, pitch, camera_target

    def _reset_camera(self):
        camera_distance, yaw, pitch, camera_target = self._get_camera_params()
        p.resetDebugVisualizerCamera(
            camera_distance, yaw, pitch,
            camera_target, physicsClientId=self._physics_client_id)

    def _initial_pybullet_setup(self, with_gui=False):
        """One-time pybullet setup stuff.
        """
        # Cached dynamic pybullet entities
        self._obj_to_pybullet_obj = {}
        # Load things into environment.
        if with_gui:
            self._physics_client_id = p.connect(p.GUI)
        else:
            self._physics_client_id = p.connect(p.DIRECT)
        self._reset_camera()
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0,
                                   physicsClientId=self._physics_client_id)
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setAdditionalSearchPath("envs/assets/")
        p.loadURDF(get_asset_path("urdf/plane.urdf"), [0, 0, -1],
                   useFixedBase=True, physicsClientId=self._physics_client_id)
        self._fetch_id = p.loadURDF(
            get_asset_path("urdf/robots/fetch.urdf"), useFixedBase=True,
            physicsClientId=self._physics_client_id)
        base_position = [0.8, 0.7441, 0]
        base_orientation = [0., 0., 0., 1.]
        p.resetBasePositionAndOrientation(
            self._fetch_id, base_position, base_orientation,
            physicsClientId=self._physics_client_id)
        # Get joints info.
        joint_names = [p.getJointInfo(
            self._fetch_id, i,
            physicsClientId=self._physics_client_id)[1].decode("utf-8")
                       for i in range(p.getNumJoints(
                           self._fetch_id,
                           physicsClientId=self._physics_client_id))]
        self._ee_id = joint_names.index("gripper_axis")
        self._ee_orn_down = p.getQuaternionFromEuler((0, np.pi/2, -np.pi))
        self._ee_orn_side = p.getQuaternionFromEuler((0, 0, 0))
        self._arm_joints = get_kinematic_chain(
            self._fetch_id, self._ee_id,
            physics_client_id=self._physics_client_id)
        self._left_finger_id = joint_names.index("l_gripper_finger_joint")
        self._right_finger_id = joint_names.index("r_gripper_finger_joint")
        self._arm_joints.append(self._left_finger_id)
        self._arm_joints.append(self._right_finger_id)
        self._init_joint_values = inverse_kinematics(
            self._fetch_id, self._ee_id, [1., 0, 0.75], self._ee_orn_down,
            self._arm_joints, physics_client_id=self._physics_client_id)
        # Add table.
        table_urdf = get_asset_path("urdf/table.urdf")
        self._table_id = p.loadURDF(table_urdf, useFixedBase=True,
                                    physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id, (1.65, 0.5, 0.0), [0., 0., 0., 1.],
            physicsClientId=self._physics_client_id)

    def _reset_pybullet_from_state(self, state, held_obj_constraint=None):
        """Reset pybullet state.
        """
        target_position = np.add(state[WORLD]["cur_grip"], [0.0, 0.0, 0.075])
        ee_orien_to_use = self._ee_orn_down
        hint_joint_values = [
            0.47979457172467466, -1.576409316226008,
            1.8756301813146756, 0.8320363798078769,
            1.3659745447630645, -0.22762065844250637,
            -0.32964011684942474, 0.034577873746798826,
            0.03507221623551996]
        # Trick to make IK work: reset to either
        # side grasp or top grasp general position
        for joint_idx, joint_val in zip(self._arm_joints,
                                        hint_joint_values):
            p.resetJointState(self._fetch_id, joint_idx, joint_val,
                              physicsClientId=self._physics_client_id)

        # Target gripper
        joint_values = inverse_kinematics(
            self._fetch_id, self._ee_id, target_position,
            ee_orien_to_use, self._arm_joints,
            physics_client_id=self._physics_client_id)
        for joint_idx, joint_val in zip(self._arm_joints, joint_values):
            p.resetJointState(self._fetch_id, joint_idx, joint_val,
                              physicsClientId=self._physics_client_id)

        # Close fingers if holding
        if held_obj_constraint is None:
            finger_val = 0.035
        else:
            finger_val = 0.025
        for finger_idx in [self._left_finger_id, self._right_finger_id]:
            p.resetJointState(self._fetch_id, finger_idx, finger_val,
                              physicsClientId=self._physics_client_id)

        # Reset objects
        if set(self._obj_to_pybullet_obj) | {WORLD} != set(state):
            # Need to recreate objects because they've changed
            self._rebuild_pybullet_objects(state)
        for obj, py_obj in self._obj_to_pybullet_obj.items():
            if held_obj_constraint is not None and \
                obj == held_obj_constraint[0]:
                pose, orn = self._apply_pick_constraint(held_obj_constraint)
            else:
                pose = state[obj]["pose"]
                orn = [0, 0, 0, 1]
            p.resetBasePositionAndOrientation(
                py_obj, pose, orn,
                physicsClientId=self._physics_client_id)

    def _rebuild_pybullet_objects(self, state):
        # Remove any existing objects.
        for obj_id in self._obj_to_pybullet_obj.values():
            p.removeBody(obj_id, physicsClientId=self._physics_client_id)
        self._obj_to_pybullet_obj = {}
        # Add new blocks.
        colors = [
            (0.95, 0.05, 0.1, 1.),
            (0.05, 0.95, 0.1, 1.),
            (0.1, 0.05, 0.95, 1.),
            (0.4, 0.05, 0.6, 1.),
            (0.6, 0.4, 0.05, 1.),
            (0.05, 0.04, 0.6, 1.),
            (0.95, 0.95, 0.1, 1.),
            (0.95, 0.05, 0.95, 1.),
            (0.05, 0.95, 0.95, 1.),
        ]
        blocks = sorted([o for o in state if o != WORLD])
        for i, block in enumerate(blocks):
            color = colors[i%len(colors)]
            width, length, height = self.block_size, self.block_size, \
                self.block_size
            mass, friction = 0.04, 1.2
            orn_x, orn_y, orn_z, orn_w = 0, 0, 0, 1
            half_extents = [width/2, length/2, height/2]
            collision_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents,
                physicsClientId=self._physics_client_id)
            visual_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color,
                physicsClientId=self._physics_client_id)
            block_id = p.createMultiBody(
                baseMass=mass, baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id, basePosition=[0, 0, 0],
                baseOrientation=[orn_x, orn_y, orn_z, orn_w],
                physicsClientId=self._physics_client_id)
            p.changeDynamics(block_id, -1, lateralFriction=friction,
                             physicsClientId=self._physics_client_id)
            self._obj_to_pybullet_obj[block] = block_id
