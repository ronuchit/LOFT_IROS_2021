"""Painting environment.
"""

import numpy as np
import matplotlib.cm as mcm
import pybullet as p
import structs
from structs import WORLD
from envs import BaseEnv
from envs.pybullet_utils import get_kinematic_chain, inverse_kinematics, \
    get_asset_path, aabb_overlap


class Painting(BaseEnv):
    """Painting environment.
    """
    table_lb = -2.1
    table_ub = -0.2
    table_height = 0.2
    shelf_slabs = 2  # number of slabs
    shelf_h = 0.18 # shelf height
    shelf_l = 2.0 # shelf length
    shelf_d = 0.15 # shelf depth
    shelf_lb = 1.
    shelf_ub = shelf_lb + shelf_l - 0.05
    shelf_box_t = 1.  # transparency (0-1)
    box_s = 0.08  # side length
    box_y = 0.5  # y coordinate
    box_lb = box_y - box_s/10.
    box_ub = box_y + box_s/10.
    obj_height = 0.13
    obj_radius = 0.03
    obj_x = 1.65
    robot_x = 0.8
    robot_z = -0.25

    def __init__(self, config):
        super().__init__(config)
        # Types
        self.obj_type = structs.Type("obj")
        self.base_type = structs.ContinuousType("base")
        self.grip_type = structs.ContinuousType("grip")
        self.color_type = structs.ContinuousType("color")
        self.water_type = structs.ContinuousType("water")
        self.heat_type = structs.ContinuousType("heat")
        self.base_type.set_sampler(lambda rng: rng.uniform(-10, 10, size=3))
        self.grip_type.set_sampler(lambda rng: rng.uniform(-10, 10, size=3))
        self.color_type.set_sampler(lambda rng: rng.uniform(0, 1))
        self.water_type.set_sampler(lambda rng: rng.uniform(0, 1))
        self.heat_type.set_sampler(lambda rng: rng.uniform(0, 1))
        # Predicates
        self.OnTable = structs.Predicate(
            "OnTable", 1, is_action_pred=False,
            holds=self._OnTable_holds, var_types=[self.obj_type])
        self.Holding = structs.Predicate(
            "Holding", 1, is_action_pred=False,
            get_satisfying_args=self._Holding_get_satisfying_args,
            holds=self._Holding_holds, var_types=[self.obj_type])
        self.HoldingSide = structs.Predicate(
            "HoldingSide", 1, is_action_pred=False,
            get_satisfying_args=self._HoldingSide_get_satisfying_args,
            holds=self._HoldingSide_holds, var_types=[self.obj_type])
        self.HoldingTop = structs.Predicate(
            "HoldingTop", 1, is_action_pred=False,
            get_satisfying_args=self._HoldingTop_get_satisfying_args,
            holds=self._HoldingTop_holds, var_types=[self.obj_type])
        self.InShelf = structs.Predicate(
            "InShelf", 1, is_action_pred=False,
            holds=self._InShelf_holds, var_types=[self.obj_type])
        self.InBox = structs.Predicate(
            "InBox", 1, is_action_pred=False,
            holds=self._InBox_holds, var_types=[self.obj_type])
        self.HandEmpty = structs.Predicate(
            "HandEmpty", 0, is_action_pred=False,
            holds=self._HandEmpty_holds, var_types=[])
        self.IsClean = structs.Predicate(
            "IsClean", 1, is_action_pred=False,
            holds=self._IsClean_holds,
            var_types=[self.obj_type])
        self.IsDirty = structs.Predicate(
            "IsDirty", 1, is_action_pred=False,
            holds=self._IsDirty_holds,
            var_types=[self.obj_type])
        self.IsDry = structs.Predicate(
            "IsDry", 1, is_action_pred=False,
            holds=self._IsDry_holds,
            var_types=[self.obj_type])
        self.IsWet = structs.Predicate(
            "IsWet", 1, is_action_pred=False,
            holds=self._IsWet_holds,
            var_types=[self.obj_type])
        self.IsBlank = structs.Predicate(
            "IsBlank", 1, is_action_pred=False,
            holds=self._IsBlank_holds,
            var_types=[self.obj_type])
        self.IsShelfColor = structs.Predicate(
            "IsShelfColor", 1, is_action_pred=False,
            holds=self._IsShelfColor_holds,
            var_types=[self.obj_type])
        self.IsBoxColor = structs.Predicate(
            "IsBoxColor", 1, is_action_pred=False,
            holds=self._IsBoxColor_holds,
            var_types=[self.obj_type])
        # Action predicates
        self.Pick = structs.Predicate(
            "Pick", 3, is_action_pred=True,
            sampler=self._pick_sampler,
            var_types=[self.obj_type, self.base_type, self.grip_type])
        self.Place = structs.Predicate(
            "Place", 2, is_action_pred=True,
            sampler=self._place_sampler,
            var_types=[self.base_type, self.grip_type])
        self.Wash = structs.Predicate(
            "Wash", 2, is_action_pred=True,
            sampler=self._wash_sampler,
            var_types=[self.obj_type, self.water_type])
        self.Dry = structs.Predicate(
            "Dry", 2, is_action_pred=True,
            sampler=self._dry_sampler,
            var_types=[self.obj_type, self.heat_type])
        self.Paint = structs.Predicate(
            "Paint", 1, is_action_pred=True,
            sampler=self._paint_sampler,
            var_types=[self.color_type])
        self._obj_to_pybullet_obj = None
        self._initial_pybullet_setup(with_gui=False)

    def get_state_predicates(self):
        return {self.OnTable, self.Holding, self.HoldingSide, self.HoldingTop,
                self.InShelf, self.InBox, self.HandEmpty,
                self.IsClean, self.IsDirty, self.IsDry, self.IsWet,
                self.IsBlank, self.IsBoxColor, self.IsShelfColor}

    def get_action_predicates(self):
        return {self.Pick, self.Place, self.Wash, self.Dry, self.Paint}

    def _get_problems(self, num_problems, all_num_objs):
        problems = []
        for i in range(num_problems):
            j = i % len(all_num_objs)
            num_objs = all_num_objs[j]
            problems.append(self._create_problem(num_objs))
        return problems

    def _get_demo_problems(self, num):
        return self._get_problems(
            num, self._cf.painting_demo_num_objs)

    def get_test_problems(self):
        return self._get_problems(
            self._cf.painting_num_test_problems,
            self._cf.painting_test_num_objs)

    def get_random_action(self, state):
        objs = list(sorted(state.keys()))
        objs.remove(WORLD)
        act_pred = self._rng.randint(5)
        if act_pred == 0:  # Pick
            obj = objs[self._rng.choice(len(objs))]
            return self._sample_ground_act(state, self.Pick, [obj])
        if act_pred == 1:  # Place
            return self._sample_ground_act(state, self.Place, [])
        if act_pred == 2:  # Wash
            obj = objs[self._rng.choice(len(objs))]
            return self._sample_ground_act(state, self.Wash, [obj])
        if act_pred == 3:  # Dry
            obj = objs[self._rng.choice(len(objs))]
            return self._sample_ground_act(state, self.Dry, [obj])
        if act_pred == 4:  # Paint
            return self._sample_ground_act(state, self.Paint, [])
        raise Exception("Can never reach here")

    def _create_problem(self, num_objs):
        state = {}
        obj_poses = []
        goals = []
        world_state = {}
        world_state["cur_holding"] = None
        world_state["cur_holding_tf"] = None
        world_state["cur_base"] = [-0.0036, 0.0, 0.0014]
        world_state["cur_grip"] = ([1., 0, 0.75], "top")
        # Sample distinct colors for shelf and box
        color1 = self._rng.uniform(0.2, 0.4)
        color2 = self._rng.uniform(0.6, 1.0)
        if self._rng.choice(2):
            box_color, shelf_color = color1, color2
        else:
            shelf_color, box_color = color1, color2
        world_state["box_color"] = box_color
        world_state["shelf_color"] = shelf_color
        # Create objects
        for i in range(num_objs):
            obj = self.obj_type(f"obj{i}")
            pose = self._sample_initial_object_pose(obj_poses)
            obj_poses.append(pose)
            # Start out wet and clean, dry and dirty, or dry and clean
            choice = self._rng.choice(3)
            if choice == 0:
                wetness = 0.
                dirtiness = self._rng.uniform(0.5, 1.)
            elif choice == 1:
                wetness = self._rng.uniform(0.5, 1.)
                dirtiness = 0.
            else:
                wetness = 0.
                dirtiness = 0.
            obj_state = {"pose": pose, "grasp": [0., 0., 0.],
                         "color": 0., "wetness" : wetness,
                         "dirtiness" : dirtiness}
            if i == num_objs-1:
                goals.append(self.InBox(obj))
                goals.append(self.IsBoxColor(obj))
            else:
                goals.append(self.InShelf(obj))
                goals.append(self.IsShelfColor(obj))
            state[obj] = obj_state
        world_state["flat"] = np.hstack([
            int(world_state["cur_holding"] is not None),
            world_state["box_color"],
            world_state["shelf_color"],
        ])
        world_state["flat_names"] = np.hstack([
            "flat:holding_something",
            "flat:box_color",
            "flat:shelf_color",
        ])
        state[WORLD] = world_state
        return state, structs.LiteralConjunction(goals)

    def _sample_initial_object_pose(self, existing_poses):
        """Get an initial object pose that is not in collision
        with existing poses
        """
        existing_ys = [p[1] for p in existing_poses]
        while True:
            this_y = self._rng.uniform(self.table_lb, self.table_ub)
            if all(abs(this_y-other_y) > 3.5*self.obj_radius
                   for other_y in existing_ys):
                return [self.obj_x, this_y,
                        self.table_height+self.obj_height/2]

    def get_next_state(self, state, action):
        if action.predicate.var_types != [var.var_type for var
                                          in action.variables]:
            next_state = self._copy_state(state)
            return next_state
        # Pick
        if action.predicate == self.Pick:
            return self._get_next_state_pick(state, action)
        # Place
        if action.predicate == self.Place:
            return self._get_next_state_place(state, action)
        # Wash
        if action.predicate == self.Wash:
            return self._get_next_state_wash(state, action)
        # Dry
        if action.predicate == self.Dry:
            return self._get_next_state_dry(state, action)
        # Paint
        if action.predicate == self.Paint:
            return self._get_next_state_paint(state, action)
        raise Exception(f"Unexpected action: {action}")

    def _get_next_state_pick(self, state, action):
        assert action.predicate == self.Pick
        next_state = self._copy_state(state)
        obj, base, grip = action.variables
        # Cannot pick if already holding something
        if state[WORLD]["cur_holding"] is not None:
            return next_state
        # Cannot pick if not on table
        if not self.table_lb < base.value[1] < self.table_ub:
            return next_state
        # Can only pick if grip is top or side grasp
        if self._is_side_grasp(grip):
            top_or_side = "side"
        elif self._is_top_grasp(grip):
            top_or_side = "top"
        else:
            return next_state
        # Execute pick
        next_state[obj]["grasp"] = grip.value
        next_state[WORLD]["cur_holding"] = (obj, top_or_side)
        next_state[WORLD]["cur_holding_tf"] = None
        next_state[WORLD]["cur_grip"] = (grip.value, top_or_side)
        next_state[WORLD]["cur_base"] = base.value
        next_state[WORLD]["flat"][0] = 1  # update holding_something
        return next_state

    def _is_side_grasp(self, grasp):
        """Check if this grasp is a valid side grasp
        """
        return grasp.value[2] < 0.2+self.obj_height

    def _is_top_grasp(self, grasp):
        """Check if this grasp is a valid top grasp
        """
        return grasp.value[2] > 0.2+self.obj_height

    def _get_next_state_place(self, state, action):
        assert action.predicate == self.Place
        next_state = self._copy_state(state)
        base, grip = action.variables
        # Cannot place if not holding something
        if state[WORLD]["cur_holding"] is None:
            return next_state
        obj, top_or_side = state[WORLD]["cur_holding"]
        # Can only place in shelf if side grasping, box if top grasping
        if self.shelf_lb < grip.value[1] < self.shelf_ub:
            shelf_or_box = "shelf"
        elif self.box_lb < grip.value[1] < self.box_ub:
            shelf_or_box = "box"
        else:
            return next_state
        if (shelf_or_box, top_or_side) not in [("shelf", "side"),
                                               ("box", "top")]:
            return next_state
        # Execute place
        next_state[WORLD]["cur_holding"] = None
        next_state[WORLD]["cur_grip"] = (grip.value, top_or_side)
        next_state[WORLD]["cur_base"] = base.value
        next_state[WORLD]["flat"][0] = 0  # update holding_something
        next_state[obj]["pose"] = grip.value
        next_state[obj]["grasp"] = [0., 0., 0.]
        # Check for collisions
        if self._state_has_forbidden_collisions(next_state):
            return self._copy_state(state)
        return next_state

    def _get_next_state_wash(self, state, action):
        assert action.predicate == self.Wash
        next_state = self._copy_state(state)
        obj, water = action.variables
        held_obj = (None if state[WORLD]["cur_holding"] is None
                    else state[WORLD]["cur_holding"][0])
        # Can only wash if holding obj
        if held_obj != obj:
            return next_state
        # Only wash if dirty
        if state[obj]["dirtiness"] <= 1e-2:
            return next_state
        # Execute wash
        if abs(state[obj]["dirtiness"] - water.value) < 1e-6:
            next_state[obj]["dirtiness"] = 0.
            next_state[obj]["wetness"] = water.value
        return next_state

    def _get_next_state_dry(self, state, action):
        assert action.predicate == self.Dry
        next_state = self._copy_state(state)
        obj, heat = action.variables
        held_obj = (None if state[WORLD]["cur_holding"] is None
                    else state[WORLD]["cur_holding"][0])
        # Can only dry if holding obj
        if held_obj != obj:
            return next_state
        # Execute dry
        if abs(state[obj]["wetness"] - heat.value) < 1e-6:
            next_state[obj]["wetness"] = 0.
        return next_state

    def _get_next_state_paint(self, state, action):
        assert action.predicate == self.Paint
        next_state = self._copy_state(state)
        color, = action.variables
        if color.value not in [state[WORLD]["shelf_color"],
                               state[WORLD]["box_color"]]:
            return next_state
        # Can only paint if holding something
        if state[WORLD]["cur_holding"] is None:
            return next_state
        obj, _ = state[WORLD]["cur_holding"]
        # Can only paint if dry and clean
        if not (state[obj]["dirtiness"] < 1e-2 and \
                state[obj]["wetness"] < 1e-2):
            return next_state
        # Don't repaint
        if state[obj]["color"] > 0.:
            return next_state
        # Execute paint
        next_state[obj]["color"] = color.value
        return next_state

    # Shared predicate functions

    @staticmethod
    def _Holding_get_satisfying_args(state):
        if state[WORLD]["cur_holding"] is not None:
            return {(state[WORLD]["cur_holding"][0],)}
        return {}

    @staticmethod
    def _Holding_holds(state, obj):
        return (state[WORLD]["cur_holding"] is not None and
                state[WORLD]["cur_holding"][0] == obj)

    @staticmethod
    def _HoldingSide_get_satisfying_args(state):
        if state[WORLD]["cur_holding"] is not None and \
           state[WORLD]["cur_holding"][1] == "side":
            return {(state[WORLD]["cur_holding"][0],)}
        return {}

    @staticmethod
    def _HoldingSide_holds(state, obj):
        return (state[WORLD]["cur_holding"] is not None and
                state[WORLD]["cur_holding"][0] == obj and
                state[WORLD]["cur_holding"][1] == "side")

    @staticmethod
    def _HoldingTop_get_satisfying_args(state):
        if state[WORLD]["cur_holding"] is not None and \
           state[WORLD]["cur_holding"][1] == "top":
            return {(state[WORLD]["cur_holding"][0],)}
        return {}

    @staticmethod
    def _HoldingTop_holds(state, obj):
        return (state[WORLD]["cur_holding"] is not None and
                state[WORLD]["cur_holding"][0] == obj and
                state[WORLD]["cur_holding"][1] == "top")

    @staticmethod
    def _HandEmpty_holds(state):
        return state[WORLD]["cur_holding"] is None

    @staticmethod
    def _IsClean_holds(state, obj):
        return state[obj]["dirtiness"] < 1e-2

    @staticmethod
    def _IsDirty_holds(state, obj):
        return state[obj]["dirtiness"] >= 1e-2

    @staticmethod
    def _IsDry_holds(state, obj):
        return state[obj]["wetness"] < 1e-2

    @staticmethod
    def _IsWet_holds(state, obj):
        return state[obj]["wetness"] >= 1e-2

    @staticmethod
    def _IsShelfColor_holds(state, obj):
        return abs(state[obj]["color"] - \
                   state[WORLD]["shelf_color"]) < 1e-4

    @staticmethod
    def _IsBoxColor_holds(state, obj):
        return abs(state[obj]["color"] - \
                   state[WORLD]["box_color"]) < 1e-4

    @staticmethod
    def _IsBlank_holds(state, obj):
        return state[obj]["color"] == 0.

    @staticmethod
    def _OnTable_holds(state, obj):
        if Painting._Holding_holds(state, obj):
            return False
        obj_pose = state[obj]["pose"]
        cls = Painting
        return cls.table_lb < obj_pose[1] < cls.table_ub

    @staticmethod
    def _InShelf_holds(state, obj):
        if Painting._Holding_holds(state, obj):
            return False
        obj_pose = state[obj]["pose"]
        cls = Painting
        return cls.shelf_lb < obj_pose[1] < cls.shelf_ub

    @staticmethod
    def _InBox_holds(state, obj):
        if Painting._Holding_holds(state, obj):
            return False
        obj_pose = state[obj]["pose"]
        cls = Painting
        return cls.box_lb < obj_pose[1] < cls.box_ub

    # Shared samplers

    @staticmethod
    def _wash_sampler(rng, state, *args):
        _ = rng  # unused
        assert len(args) == 1
        obj = args[0]
        if obj.var_type != "obj":
            return (0.0,)
        dirtiness = state[obj]["dirtiness"]
        return (dirtiness,)

    @staticmethod
    def _dry_sampler(rng, state, *args):
        _ = rng  # unused
        assert len(args) == 1
        obj = args[0]
        if obj.var_type != "obj":
            return (0.0,)
        wetness = state[obj]["wetness"]
        return (wetness,)

    @staticmethod
    def _paint_sampler(rng, state, *args):
        # Randomly sample either shelf color or box color
        assert len(args) == 0
        shelf_color = state[WORLD]["shelf_color"]
        box_color = state[WORLD]["box_color"]
        if rng.randint(2):
            return (shelf_color,)
        return (box_color,)

    @staticmethod
    def _pick_sampler(rng, state, *args):
        assert len(args) == 1
        obj = args[0]
        if obj.var_type != "obj":
            return ([0, 0, 0], [0, 0, 0])
        obj_position = state[obj]["pose"]
        if rng.randint(2) == 0:
            # Top grasp
            base = [Painting.robot_x+0.2, obj_position[1],
                    Painting.robot_z]
            grip = np.add(obj_position, [0, 0, 0.15])
        else:
            # Side grasp
            obj_position = state[obj]["pose"]
            base = [Painting.robot_x, obj_position[1],
                    Painting.robot_z]
            grip = np.add(obj_position, [0, 0, 0.05])
        return (base, grip)

    @staticmethod
    def _place_sampler(rng, state, *args):
        _ = state  # unused
        # Randomly sample either shelf placement or box placement
        assert len(args) == 0
        if rng.randint(2) == 0:
            lb = Painting.shelf_lb
            ub = Painting.shelf_ub
            target_in_box = False
        else:
            lb = Painting.box_lb
            ub = Painting.box_ub
            target_in_box = True
        grip_y = rng.uniform(lb, ub)
        if target_in_box:  # target is in the box
            target_position = [Painting.obj_x, grip_y, \
                               Painting.table_height + \
                               Painting.obj_height/2.]
            base = [Painting.robot_x+0.2, target_position[1],
                    Painting.robot_z]
            grip = target_position
        else:  # target is in the shelf
            target_position = [1.65+Painting.shelf_d/2-0.02, grip_y,
                               0.15+Painting.shelf_h]
            base = [Painting.robot_x-0.2, target_position[1],
                    Painting.robot_z+0.02]
            # grip = np.add(target_position, [-0.3, 0, 0.03])
            grip = target_position
        return (base, grip)

    def _state_has_forbidden_collisions(self, state, held_obj_constraint=None):
        # Set state
        self._reset_pybullet_from_state(
            state, held_obj_constraint=held_obj_constraint)
        ## Get aabbs
        # Get object aabbs
        all_object_aabbs = []
        nonheld_object_aabbs = []
        held_obj_aabb = None
        for obj, py_obj in self._obj_to_pybullet_obj.items():
            aabb = p.getAABB(
                py_obj, physicsClientId=self._physics_client_id)
            all_object_aabbs.append(aabb)
            if not held_obj_constraint or \
                held_obj_constraint[0] != obj:
                nonheld_object_aabbs.append(aabb)
            else:
                held_obj_aabb = aabb
        # Get shelf aabbs
        shelf_aabbs = []
        for shelf_side_id in self._shelf_collision_ids:
            aabb = p.getAABB(self._shelf_id, shelf_side_id,
                             physicsClientId=self._physics_client_id)
            shelf_aabbs.append(aabb)
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
        ## Check collisions
        # Check collisions between objects
        for i, aabb_i in enumerate(all_object_aabbs[:-1]):
            for aabb_j in all_object_aabbs[i+1:]:
                if aabb_overlap(aabb_i, aabb_j):
                    return "collision"
        # Check collisions between held object and shelf sides
        if held_obj_aabb:
            for shelf_aabb in shelf_aabbs:
                if aabb_overlap(held_obj_aabb, shelf_aabb):
                    return "collision"
        # Check collisions between robot and shelf and table
        for robot_aabb in robot_aabbs:
            for collider_aabb in shelf_aabbs+[table_aabb]:
                if aabb_overlap(robot_aabb, collider_aabb):
                    return "collision"
        return None

    @staticmethod
    def _get_camera_params():
        camera_distance = 6
        yaw = 90
        pitch = -5
        camera_target = [1.05, 0.5, 0.42]
        return camera_distance, yaw, pitch, camera_target

    def _reset_camera(self):
        camera_distance, yaw, pitch, camera_target = self._get_camera_params()
        p.resetDebugVisualizerCamera(
            camera_distance, yaw, pitch,
            camera_target, physicsClientId=self._physics_client_id)

    def _apply_pick_constraint(self, held_obj_tf):
        base_link = np.r_[p.getLinkState(
            self._fetch_id, self._ee_id,
            physicsClientId=self._physics_client_id)[:2]]
        _, transf = held_obj_tf
        obj_loc, _ = p.multiplyTransforms(
            base_link[:3], base_link[3:], transf[0], transf[1])
        return obj_loc

    @staticmethod
    def _color_to_rgba(color):
        if isinstance(color, tuple):
            return color
        return mcm.gist_rainbow(color)  # pylint:disable=no-member

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
        table_urdf = get_asset_path("urdf/table2.urdf")
        self._table_id = p.loadURDF(table_urdf, useFixedBase=True,
                                    physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id, (1.65, 0.5, 0.0), [0., 0., 0., 1.],
            physicsClientId=self._physics_client_id)
        # Add shelf.
        link_vis = []
        link_cols = []
        link_pos = []
        cur_height = self.shelf_d+self.shelf_h/2
        origin_y = self.shelf_lb + self.shelf_l/2
        for i in range(self.shelf_slabs):
            # Left wall.
            link_cols.append(p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[self.shelf_d/2, 0.01/2,
                                         self.shelf_h/2],
                physicsClientId=self._physics_client_id))
            link_vis.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[self.shelf_d/2, 0.01/2,
                                         self.shelf_h/2],
                rgbaColor=(0.6, 0.3, 0.0, self.shelf_box_t),
                physicsClientId=self._physics_client_id))
            link_pos.append([1.65, origin_y-self.shelf_l/2, cur_height])
            # Right wall.
            link_cols.append(p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[self.shelf_d/2, 0.01/2,
                                         self.shelf_h/2],
                physicsClientId=self._physics_client_id))
            link_vis.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[self.shelf_d/2, 0.01/2,
                                         self.shelf_h/2],
                rgbaColor=(0.6, 0.3, 0.0, self.shelf_box_t),
                physicsClientId=self._physics_client_id))
            link_pos.append([1.65, origin_y+self.shelf_l/2, cur_height])
            # Back wall.
            link_cols.append(p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.01/2, self.shelf_l/2,
                                         self.shelf_h/2],
                physicsClientId=self._physics_client_id))
            link_vis.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.01/2, self.shelf_l/2,
                                         self.shelf_h/2],
                rgbaColor=(0.7, 0.4, 0.1, 0.1),
                physicsClientId=self._physics_client_id))
            link_pos.append([1.65+self.shelf_d/2, origin_y, cur_height])
            # Bottom wall.
            link_cols.append(p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[self.shelf_d/2, self.shelf_l/2,
                                         0.01/2],
                physicsClientId=self._physics_client_id))
            link_vis.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[self.shelf_d/2, self.shelf_l/2,
                                         0.01/2],
                rgbaColor=(0.7, 0.4, 0.1, self.shelf_box_t),
                physicsClientId=self._physics_client_id))
            link_pos.append([1.65, origin_y, cur_height])
            if i == self.shelf_slabs-1:
                continue
            cur_height += self.shelf_h
        self._shelf_id = p.createMultiBody(
            linkMasses=[10 for _ in link_pos],
            linkCollisionShapeIndices=link_cols,
            linkVisualShapeIndices=link_vis,
            linkPositions=link_pos,
            linkOrientations=[[0, 0, 0, 1] for _ in link_pos],
            linkInertialFramePositions=[[0, 0, 0] for _ in link_pos],
            linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in link_pos],
            linkParentIndices=[0 for _ in link_pos],
            linkJointTypes=[p.JOINT_FIXED for _ in link_pos],
            linkJointAxis=[[0, 0, 0] for _ in link_pos],
            physicsClientId=self._physics_client_id)
        # Exclude bottom and back walls
        self._shelf_collision_ids = [0, 1, 4, 5, 7]
        # Add box.
        link_vis = []
        link_cols = []
        link_pos = []
        # Left wall.
        link_cols.append(p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[self.box_s/2, 0.01/2, self.box_s/2],
            physicsClientId=self._physics_client_id))
        link_vis.append(p.createVisualShape(
            p.GEOM_BOX, halfExtents=[self.box_s/2, 0.01/2, self.box_s/2],
            rgbaColor=(0.6, 0.3, 0.0, self.shelf_box_t),
            physicsClientId=self._physics_client_id))
        link_pos.append([1.65, self.box_y-self.box_s/2, 0.2])
        # Right wall.
        link_cols.append(p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[self.box_s/2, 0.01/2, self.box_s/2],
            physicsClientId=self._physics_client_id))
        link_vis.append(p.createVisualShape(
            p.GEOM_BOX, halfExtents=[self.box_s/2, 0.01/2, self.box_s/2],
            rgbaColor=(0.6, 0.3, 0.0, self.shelf_box_t),
            physicsClientId=self._physics_client_id))
        link_pos.append([1.65, self.box_y+self.box_s/2, 0.2])
        # Front wall.
        link_cols.append(p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.01/2, self.box_s/2, self.box_s/2],
            physicsClientId=self._physics_client_id))
        link_vis.append(p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.01/2, self.box_s/2, self.box_s/2],
            rgbaColor=(0.6, 0.3, 0.0, self.shelf_box_t),
            physicsClientId=self._physics_client_id))
        link_pos.append([1.65-self.box_s/2, self.box_y, 0.2])
        # Right wall.
        link_cols.append(p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.01/2, self.box_s/2, self.box_s/2],
            physicsClientId=self._physics_client_id))
        link_vis.append(p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.01/2, self.box_s/2, self.box_s/2],
            rgbaColor=(0.6, 0.3, 0.0, self.shelf_box_t),
            physicsClientId=self._physics_client_id))
        link_pos.append([1.65+self.box_s/2, self.box_y, 0.2])
        self._box_id = p.createMultiBody(
            linkMasses=[10 for _ in link_pos],
            linkCollisionShapeIndices=link_cols,
            linkVisualShapeIndices=link_vis,
            linkPositions=link_pos,
            linkOrientations=[[0, 0, 0, 1] for _ in link_pos],
            linkInertialFramePositions=[[0, 0, 0] for _ in link_pos],
            linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in link_pos],
            linkParentIndices=[0 for _ in link_pos],
            linkJointTypes=[p.JOINT_FIXED for _ in link_pos],
            linkJointAxis=[[0, 0, 0] for _ in link_pos],
            physicsClientId=self._physics_client_id)

    def _reset_pybullet_from_state(self, state, held_obj_constraint=None):
        """Reset pybullet state.
        """
        # Reset robot base
        if state[WORLD]["cur_base"] is not None:
            p.resetBasePositionAndOrientation(
                self._fetch_id, state[WORLD]["cur_base"],
                [0, 0, 0, 1], physicsClientId=self._physics_client_id)

        # Reset robot arm joints using IK
        if state[WORLD]["cur_grip"] is None:
            # Not yet holding anything
            joint_values = self._init_joint_values
            for joint_idx, joint_val in zip(self._arm_joints, joint_values):
                p.resetJointState(self._fetch_id, joint_idx, joint_val,
                                  physicsClientId=self._physics_client_id)
        else:
            target_position, top_or_side = state[WORLD]["cur_grip"]
            # Side or top grasping?
            if top_or_side == "side":
                ee_orien_to_use = self._ee_orn_side
                hint_joint_values = [
                    -0.24741349185819886, -1.1140099619162491,
                    -0.10474956733626553, 1.3250953017871197,
                    0.2971917287636747, -0.1769482446641858,
                    -0.1520215970979298, 0.03481383097068269,
                    0.03480223078302643]
            else:
                assert top_or_side == "top"
                target_position = target_position.copy()
                if state[WORLD]["cur_holding"] is None:
                    target_position[2] += 0.05
                else:
                    target_position[2] -= 0.1
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

        # Don't close fingers
        for finger_idx in [self._left_finger_id, self._right_finger_id]:
            p.resetJointState(self._fetch_id, finger_idx, 0.04,
                              physicsClientId=self._physics_client_id)

        # Reset objects
        if set(self._obj_to_pybullet_obj) | {WORLD} != set(state):
            # Need to recreate objects because they've changed
            self._rebuild_pybullet_objects(state)
        for obj, py_obj in self._obj_to_pybullet_obj.items():
            if held_obj_constraint is not None and \
                obj == held_obj_constraint[0]:
                pose = self._apply_pick_constraint(held_obj_constraint)
            else:
                pose = state[obj]["pose"]
            p.resetBasePositionAndOrientation(
                py_obj, pose, [0, 0, 0, 1],
                physicsClientId=self._physics_client_id)

    def _rebuild_pybullet_objects(self, state):
        # Remove any existing objects.
        for obj_id in self._obj_to_pybullet_obj.values():
            p.removeBody(obj_id, physicsClientId=self._physics_client_id)
        self._obj_to_pybullet_obj = {}
        # Add new objects.
        for obj in state:
            if obj == WORLD:
                continue
            color = self._color_to_rgba(state[obj]["color"])
            mass, friction = 0.04, 1.2
            collision_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=self.obj_radius, height=self.obj_height,
                physicsClientId=self._physics_client_id)
            visual_id = p.createVisualShape(
                p.GEOM_CYLINDER, radius=self.obj_radius, length=self.obj_height,
                rgbaColor=color, physicsClientId=self._physics_client_id)
            obj_id = p.createMultiBody(
                baseMass=mass, baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id, basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self._physics_client_id)
            p.changeDynamics(obj_id, -1, lateralFriction=friction,
                             physicsClientId=self._physics_client_id)
            self._obj_to_pybullet_obj[obj] = obj_id
