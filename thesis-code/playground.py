import gym

def str_mj_arr(arr):
    return ' '.join(['%0.3f' % arr[i] for i in range(arr._length_)])

def print_contact_info(env):
    d = env.unwrapped.data
    for coni in range(d.ncon):
        print('  Contact %d:' % (coni,))
        con = d.obj.contact[coni]
        print('    dist     = %0.3f' % (con.dist,))
        print('    pos      = %s' % (str_mj_arr(con.pos),))
        print('    frame    = %s' % (str_mj_arr(con.frame),))
        print('    friction = %s' % (str_mj_arr(con.friction),))
        print('    dim      = %d' % (con.dim,))
        print('    geom1    = %d' % (con.geom1,))
        print('    geom2    = %d' % (con.geom2,))

def run_env(env, step_cb):
    env.reset()
    stepi = 0
    while True:
        print('Step %d:' % (stepi,))
        step_cb(env)
        obs, rew, done, info = env.step(env.action_space.sample())
        stepi += 1
        if done: break
        env.render()
        # x = input()

def main():
    run_env(gym.make('Grasper3d-v0'), print_contact_info)

if __name__ == '__main__': main()