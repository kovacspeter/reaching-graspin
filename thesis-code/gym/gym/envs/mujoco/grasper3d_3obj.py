import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class Grasper3DEnv_3obj(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target = ['box', 'ball', 'box2']
        self.target_int = np.random.randint(3)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'grasper3d_3obj.xml', 2)

    def _step(self, a):
        # # DISTANCE REWARDS
        # vec = self.get_body_com("palm") - self.get_body_com(self.target[self.target_int])
        # vec2 = self.get_body_com("rightclaw") - self.get_body_com(self.target[self.target_int])
        # vec3 = self.get_body_com("leftclaw") - self.get_body_com(self.target[self.target_int])
        #
        # # CONTACT REWARDS
        # GC = GrasperContact(self)
        # c1 = GC.is_contact(17, 14) + GC.is_contact(17, 16)
        # c2 = GC.is_contact(17, 13) + GC.is_contact(17, 15)
        # c3 = GC.is_contact(17, 10) + GC.is_contact(17, 11)
        # c4 = GC.is_contact(17, 9) + GC.is_contact(17, 12)
        #
        # target_gripper_contact = float(c1 + c2 + c3 + c4)
        #
        # # ground_contact = - GC.is_contact(0, 16) - GC.is_contact(0, 15) - GC.is_contact(0, 12) - GC.is_contact(0, 11)
        # # GRIPPER-fingers COLLISION
        # # c9 = - GC.is_contact(16, 12) - GC.is_contact(15, 11) if ground_contact < 3 else 0
        #
        # done = False
        #
        # heigth = (self.get_body_com(self.target[self.target_int])[2] + .19)
        #
        # if heigth < 0.1:
        #     heigth_rew = 0
        # else:
        #     multiplier = 0
        #     if target_gripper_contact > 3:
        #         multiplier = 1.
        #         done = True
        #     heigth_rew = (1000 + heigth * 1000) * multiplier
        #     if target_gripper_contact > 3:
        #         print (heigth_rew)
        #
        # reward_cont = target_gripper_contact  # + ground_contact + c9
        # reward_dist = - np.linalg.norm(vec) - np.linalg.norm(vec2) - np.linalg.norm(vec3)
        # reward_ctrl = - np.square(a).sum()
        # reward = reward_dist + reward_ctrl + float(reward_cont) + heigth_rew
        # self.do_simulation(a, self.frame_skip)
        # ob = self._get_obs()
        # DISTANCE REWARDS
        vec = self.get_body_com("palm")-self.get_body_com(self.target[self.target_int])
        vec2 = self.get_body_com("rightclaw") - self.get_body_com(self.target[self.target_int])
        vec3 = self.get_body_com("leftclaw") - self.get_body_com(self.target[self.target_int])
        vec4 = self.get_body_com("leftclaw") - self.get_body_com("rightclaw")

        # CONTACT REWARDS
        GC = GrasperContact(self)
        c1 = GC.is_contact(17, 14) + GC.is_contact(17, 16)
        c2 = GC.is_contact(17, 13) + GC.is_contact(17, 15)
        c3 = GC.is_contact(17, 10) + GC.is_contact(17, 11)
        c4 = GC.is_contact(17, 9) + GC.is_contact(17, 12)

        target_gripper_contact = float(c1 + c2 + c3 + c4)

        ground_contact = - GC.is_contact(0, 16) - GC.is_contact(0, 15) - GC.is_contact(0, 12) - GC.is_contact(0, 11)
        # # GRIPPER-fingers COLLISION
        c9 = - GC.is_contact(16, 12) - GC.is_contact(15, 11) if target_gripper_contact <= 4 else 0

        done = False

        heigth = (self.get_body_com(self.target[self.target_int])[2] + .2)

        if heigth < 0.05:
            heigth_rew = 0
        else:
            multiplier = 0
            if target_gripper_contact > 3:
                multiplier = 1.
                done = True
            heigth_rew = (1000. + heigth*1000) * multiplier
            # if target_gripper_contact > 3:
            #     print (heigth_rew)


        reward_cont = target_gripper_contact + ground_contact + c9
        reward_dist = - np.linalg.norm(vec) - np.linalg.norm(vec2) - np.linalg.norm(vec3) + np.linalg.norm(vec4)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl + float(reward_cont) + heigth_rew
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * .7

    def reset_model(self):
        self.target_int = np.random.randint(3)

        qpos = self.np_random.uniform(low=-.1, high=.1, size=self.model.nq) + self.init_qpos

        shift = self.target_int * 3
        f = 7 + shift
        t = 10 + shift

        # qpos[6:15] = [.51]*9

        while True:
            self.goal = self.np_random.uniform(low=-0.1, high=0.1, size=2)
            self.goal += [-.51, -.51]
            if np.linalg.norm(self.goal) < 3: break

        qpos[f:t - 1] = self.goal
        qpos[t-1] = 0

        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-9:] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:6]

        shift = self.target_int * 3
        f = 7 + shift
        t = 10 + shift

        obs = np.concatenate([
            theta,
            # self.get_body_com(self.target[self.target_int]),
            self.model.data.qpos.flat[f:t],
            self.model.data.qvel.flat[:6],
            self.get_body_com("palm") - self.get_body_com(self.target[self.target_int]), # TODO testovat robustnost na nepresnost
        ])
        print obs
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
