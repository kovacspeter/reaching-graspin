from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.reacher3d import Reacher3DEnv
from gym.envs.mujoco.grasper3d import Grasper3DEnv
from gym.envs.mujoco.grasper3d_3obj import Grasper3DEnv_3obj
from gym.envs.mujoco.grasper3d_wall import Grasper3DEnvWall
from gym.envs.mujoco.grasper3d_move import MovingGrasper3DEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.reacher3d_wall import Reacher3DEnvWall
