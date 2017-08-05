import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Reacher3DEnvWall(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self.target_int = np.random.randint(2)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher3d_wall.xml', 2)

    def _step(self, a):
        # DISTANCE REWARDS
        vec = self.get_body_com("palm") - self.get_body_com("target")
        vec4 = self.get_body_com("wall") - self.get_body_com("palm")
        GC = GrasperContact(self)
        done = False

        wall_contact = - GC.is_contact(17, 14) - GC.is_contact(17, 13) - GC.is_contact(17, 10) - GC.is_contact(17, 9) \
                       - GC.is_contact(17, 15) - GC.is_contact(17, 16) - GC.is_contact(17, 11) - GC.is_contact(17, 12) \
                       - GC.is_contact(17, 6) - GC.is_contact(17, 7)

        wrong_wall_side_penalty = -1 if np.linalg.norm(vec) > np.linalg.norm(vec4) else 0

        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl + float(wall_contact) + wrong_wall_side_penalty\
             # + float(reward_cont) + heigth_rew
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()


        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 17# 0
        self.viewer.cam.distance = self.model.stat.extent * .7

    def reset_model(self):
        self.target_int = np.random.randint(2)
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        while True:
            self.goal = self.np_random.uniform(low=-0.15, high=0.15, size=3)
            if np.linalg.norm(self.goal) < 3: break

        qpos[-3:] = self.goal
        if self.target_int == 0:
            # left(start camera view) of wall
            qpos[-3] = 0
        else:
            # right(start camera view) of wall
            qpos[-3] = .2
            qpos[0] = -179
        qpos[-1] = 0
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-3:] = 0
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:6]

        obs = np.concatenate([
            theta,
            self.model.data.qpos.flat[7:],
            self.model.data.qvel.flat[:6],
            # TODO popisat v diplomke - measure different feature representations
            self.get_body_com("palm") - self.get_body_com("target"),
            [1,0] if self.target_int == 0 else [0,1]
        ])

        return obs

class GrasperContact():

    def __init__(self, env):
        self.env = env

    def get_contacts(self):

        data = self.env.unwrapped.data
        cons = []
        for i in range(data.ncon):
            cons.append(data.obj.contact[i])

        return cons

    def is_contact(self, id1, id2):
        for contact in self.get_contacts():
            if self._is_contact(contact, id1, id2):
                return True
        return False

    def target_contact(self, c):
        if c.geom1 == 17 or c.geom2 == 17:
            return True
        return False

    def grasper_contact(self, c):
        if (c.geom1 > 8 and c.geom1 < 17) or (c.geom2 > 8 and c.geom2 < 17):
            return True
        return False

    def ground_contact(self, c):
        if c.geom1 == 0 or c.geom2 == 0:
            return True
        return False

    def _is_contact(self, c, id1, id2):
        if (c.geom1 == id1 and c.geom2 == id2) or (c.geom1 == id2 and c.geom2 == id1):
            return True
        return False
