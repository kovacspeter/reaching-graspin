import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Reacher3DEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher3d.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("palm") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        # print a, vec
        # print reward_ctrl, reward_dist
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid=0
        self.viewer.cam.distance = self.model.stat.extent * .7

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.10, high=.10, size=3)
            if np.linalg.norm(self.goal) < 3: break

        qpos[-3:] = self.goal
        qpos[-1] = 0
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-3:] = 0
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:4]

        return np.concatenate([
            theta,
            self.model.data.qpos.flat[4:],
            self.model.data.qvel.flat[:4],
            self.get_body_com("palm") - self.get_body_com("target")
        ])
