import multiprocessing, errno
from multiprocessing import Process, JoinableQueue, Queue
import sys, os, json
import argparse
import numpy as np
import scipy
import scipy.signal
import tensorflow as tf
import time
import signal

sys.path.append('./gym')
import gym

signal.signal(signal.SIGINT, signal.default_int_handler)

# PROGRAM PARAMETERS
ENV_NAME_DEFAULT = 'Grasper3d-v0'
SCOPE_DEFAULT = 'grasper'
IS_LOAD_DEFAULT = False
N_PROCESSES_DEFAULT = multiprocessing.cpu_count()
if sys.platform == "darwin":  # we want physical cores not logical
    N_PROCESSES_DEFAULT /= 2
IS_MONITOR_DEFAULT = True
IS_TRAIN_DEFAULT = True
MONITOR_FREQ_DEFAULT = 50

# SAVING & STUFF
LOG_DIR_DEFAULT = './logs'
CHECKPOINT_DIR_DEFAULT = './checkpoints'
CHECKPOINT_FREQ_DEFAULT = 50

# ALGORITHM PARAMETERS
GAMMA_DEFAULT = .995
MAX_KL_DEFAULT = .01
TRAJ_LEN_DEFAULT = 100
N_TRAJS_DEFAULT = 100


class RolloutProcess(Process):
    def __init__(self, env_name, job_queue, result_queue, process_id, traj_len, max_kl):
        Process.__init__(self)
        self.process_id = process_id
        self.traj_len = traj_len
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.env_name = env_name
        self.max_kl = max_kl

    def run(self):
        self.env = gym.make(self.env_name)
        self.env.seed(np.random.randint(0, 999999))
        self.input_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.trpo = TRPO("Process-" + str(self.process_id), self.input_dim, self.action_dim, self.max_kl)
        # WAIT FOR TASK AND LOOP :)
        while True:
            task = self.job_queue.get()

            task_id = task[0]
            task_data = task[1]

            # COLLECT TRAJECTORY AND PUT IT IN QUEUE
            if task_id == "get_traj":
                path = get_trajs(self.env, self.trpo, 1, self.traj_len)[0]
                self.job_queue.task_done()
                self.result_queue.put(path)

            # STOP PROCESS
            elif task_id == "kill":
                print ("kill message")
                self.job_queue.task_done()
                break

            # SET POLICY WEIGHTS
            elif task_id == 'update_params':
                self.trpo.sff(*task_data)
                # not a nice way to ensure that other actors update its
                # weights as well but it works ...
                time.sleep(0.2)
                self.job_queue.task_done()


class ParallelRollout():
    def __init__(self, env_name, traj_len, n_processes, max_kl):
        self.n_processes = n_processes
        self.job_queue = JoinableQueue()
        self.result_queue = Queue()

        self.processes = []
        for i in range(n_processes):
            self.processes.append(RolloutProcess(env_name, self.job_queue, self.result_queue, i, traj_len, max_kl))

        for proc in self.processes:
            proc.start()

    def get_trajs(self, n_trajectories):

        for i in range(n_trajectories):
            self.job_queue.put(['get_traj', ()])

        self.job_queue.join()

        i = n_trajectories
        trajs = []
        while i:
            i -= 1
            trajs.append(self.result_queue.get())

        return trajs

    def set_trpo_policies(self, weights):

        for i in range(self.n_processes):
            self.job_queue.put(['update_params', (weights,)])

        self.job_queue.join()

    def end(self):

        for i in range(self.n_processes):
            self.job_queue.put(['kill', ()])

        self.job_queue.join()

    def __del__(self):
        self.end()


class TRPOAgent(Process):
    def __init__(self, env_name, name, max_kl, job_queue, result_queue):
        Process.__init__(self)

        self.env_name = env_name
        self.name = name
        self.task_q = job_queue
        self.max_kl = max_kl
        self.result_q = result_queue

        self.data = {
            "timestamps": [],
            "rewards": []
        }

    def run(self):
        self.env = gym.make(self.env_name)
        self.input_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.trpo = TRPO(self.name, self.input_dim, self.action_dim, self.max_kl)

        while True:
            task = self.task_q.get()

            # task is array of format (task_id, task_data)
            task_id = task[0]
            task_data = task[1]

            if task_id == "save":
                self.save(*task_data)
                self.task_q.task_done()

            elif task_id == 'load':
                self.load(*task_data)
                self.task_q.task_done()

            elif task_id == 'learn':
                self.learn(*task_data)
                self.task_q.task_done()

            elif task_id == 'kill':
                self.job_queue.task_done()
                break

            elif task_id == 'act':
                action = self.act(*task_data)
                self.task_q.task_done()
                self.result_q.put(action)

            elif task_id == 'log':
                self.log(*task_data)
                self.task_q.task_done()

    def log(self, path, file_name):
        file_path = os.path.join(path, file_name)
        print ("Logging to ", file_path)

        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                data = json.loads(f.read())
                data['rewards'].extend(self.data['rewards'])
                data['timestamps'].extend(self.data['timestamps'])
        else:
            data = self.data
            try:
                os.makedirs(path)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

        with open(file_path, "w+") as f:
            f.write(json.dumps(data))

        self.data = {
            "timestamps": [],
            "rewards": []
        }

    def save(self, path):
        print ("Saving to ", path)
        self.trpo.save(path)

    def load(self, path):
        print ("Loading from ", path)
        self.trpo.load(path)

    def act(self, obs):
        return self.trpo.act(obs)

    def _collect_trajectories(self, episodes, episode_length):
        return get_trajs(self.env, self.trpo, episodes, episode_length)

    def _update_policy(self, observations, actions, advantages, action_mean, action_logstd):
        self.trpo.update_policy(observations, actions, advantages, action_mean, action_logstd)

    def learn(self, GAMMA, episodes, episode_length):
        # COLLECT TRAJECTORIES (SAMPLES)
        trajs = self._collect_trajectories(episodes, episode_length)

        for traj in trajs:
            traj["baseline"] = self.trpo.vf.predict(traj)
            traj["returns"] = discount_cumsum(traj["rews"], GAMMA)
            traj["advantage"] = traj["returns"] - traj["baseline"]

        action_mean = np.concatenate([traj["action_means"] for traj in trajs])
        action_logstd = np.concatenate([traj["action_stds"] for traj in trajs])
        obs_n = np.concatenate([traj["obs"] for traj in trajs])
        action_n = np.concatenate([traj["acts"] for traj in trajs])

        advant_n = np.concatenate([traj["advantage"] for traj in trajs])
        advant_n -= advant_n.mean()
        advant_n /= (advant_n.std() + 1e-8)

        # UPDATE VALUE FUNCTION
        self.trpo.vf.fit(trajs)

        # UPDATE TRPO POLICY
        self._update_policy(obs_n, action_n, advant_n, action_mean, action_logstd)

        episoderewards = np.array([traj["rews"].sum() for traj in trajs])

        mean = episoderewards.mean()
        self.data['rewards'].append(mean)
        self.data['timestamps'].append(time.time())

        print ("Average sum of rewards per episode", mean)


class ParallelTRPOAgent(TRPOAgent):
    def __init__(self, env_name, name, max_kl, job_queue, result_queue, parallel_rollouts):
        self.parallel_rollouts = parallel_rollouts
        TRPOAgent.__init__(self, env_name, name, max_kl, job_queue, result_queue)

    def load(self, file_name):
        TRPOAgent.load(self, file_name)
        self.parallel_rollouts.set_trpo_policies(self.trpo.gf())

    def _collect_trajectories(self, episodes, episode_length):
        # COLLECT TRAJECTORIES ON MULTIPLE CORES
        return self.parallel_rollouts.get_trajs(episodes)

    def _update_policy(self, observations, actions, advantages, action_mean, action_logstd):
        TRPOAgent._update_policy(self, observations, actions, advantages, action_mean, action_logstd)

        self.parallel_rollouts.set_trpo_policies(self.trpo.gf())


class TRPO():
    def __init__(self, scope, input_dim, action_dim, max_kl):
        self.sess = tf.Session()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.max_kl = max_kl
        self.vf = LinearVF()
        # self.vf = NeuralVF(self.sess, 1, 30, self.input_dim)
        self.last_dir = None

        with tf.variable_scope(scope):

            self.observation = tf.placeholder(tf.float32, [None, input_dim], 'observation')
            self.action = tf.placeholder(tf.float32, [None, action_dim], 'action')
            self.advantage = tf.placeholder(tf.float32, [None], 'advantage')

            self.old_action_mean = tf.placeholder(tf.float32, [None, action_dim], 'old_action_mean')
            self.old_action_logstd = tf.placeholder(tf.float32, [None, action_dim], 'old_action_logstd')

            with tf.variable_scope("policy"):
                h1 = tf.nn.tanh(self.forward(self.observation, input_dim, 100, "h1"))
                h2 = tf.nn.tanh(self.forward(h1, 100, 100, "h2"))
                out = self.forward(h2, 100, action_dim, "h3")

                # TODO try bigger/lower "exploration"
                action_logstd = tf.Variable((.1 * np.random.randn(1, action_dim)).astype(np.float32),
                                            name="action_logstd")
            self.action_mean = out
            self.action_logstd = action_logstd

            # probability of actions with old theta and new theta
            prob = self.loglikelihood(self.action_mean, self.action_logstd, self.action)
            oldprob = self.loglikelihood(self.old_action_mean, self.old_action_logstd, self.action)

            # Sampling ration between old and new
            sampling_ratio = tf.exp(prob - oldprob)

            # loss function L = Expected value [sampling_ration * Q]
            # instead Q we are using Advantage
            self.loss = -tf.reduce_mean(sampling_ratio * self.advantage)

            # Fisher Information Matrix (FIM) is second derivative of KL divergence D_kl(theta || theta)
            # theta = tf.trainable_variables()
            theta = []
            for i in tf.trainable_variables():
                if "ValueFunction" not in i.name:
                    theta.append(i)

            self.policy_gradient = flatgrad(self.loss, theta)

            samples = tf.cast(tf.shape(self.observation)[0], tf.float32)
            kl = self.D_kl_gauss(tf.stop_gradient(self.action_mean),
                                 tf.stop_gradient(self.action_logstd),
                                 self.action_mean,
                                 self.action_logstd) / samples

            grads = tf.gradients(kl, theta)
            # what vector we're multiplying by
            self.multiply_vector = tf.placeholder(tf.float32, [None])
            shapes = map(var_shape, theta)
            start = 0
            tangents = []
            for shape in shapes:
                size = np.prod(shape)
                param = tf.reshape(self.multiply_vector[start:(start + size)], shape)
                tangents.append(param)
                start += size

            # gradient of KL w/ itself * tangent
            gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
            # 2nd gradient of KL w/ itself * tangent
            self.fvp = flatgrad(gvp, theta)

            # self.gf() will return flattned parameters
            self.gf = GetFlat(self.sess, theta)
            # self.sff(theta) will set parameters of our policy
            self.sff = SetFromFlat(self.sess, theta)

        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()

    def save(self, path):
        """
        Save weights to file in path
        :param path: path to file where we ant to store weights
        :return: None
        """
        self.saver.save(self.sess, path)

    def load(self, path):
        """
        Restore weights from file with path "file_name"
        :param path: path to file containing weights
        :return: None
        """
        self.saver.restore(self.sess, path)

    def forward(self, input_layer, input_dim, hidden_dim, scope):
        """
        Just creates and returns forward layer(part of computation graph).

        :param input_layer: Input Tensor
        :param input_dim: dimension of input
        :param hidden_dim: output layer dimension
        :param scope: name of scope
        :return: operation doing forward pass through layer Wx + b
        """
        with tf.variable_scope(scope):
            W = tf.get_variable("W", [input_dim, hidden_dim],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", [hidden_dim], initializer=tf.constant_initializer(0.0))

        return tf.matmul(input_layer, W) + b

    def loglikelihood(self, mu, logstd, x):
        """
        Computes probability to take action x, given paramaterized guassian distribution, according to :
        https://www.ii.pwr.edu.pl/~tomczak/PDF/[JMT]Fisher_inf.pdf - we need this for FIM check example 3 (17)

        :param mu: Mean of gaussian
        :param logstd: Logarithm of std of gaussian
        :param x: Action which probability we want to compute
        :return: Probability of taking action x
        """

        # e ^ (2 log(x)) = x ^ 2
        var = tf.exp(2 * logstd)
        prob = -0.5 * (tf.square(x - mu) / var + tf.log(tf.constant(2 * np.pi))) - logstd
        return tf.reduce_sum(prob, [1])

    def D_kl_gauss(self, mean1, logstd1, mean2, logstd2):
        """
        Computes KL divergence of two gaussian distributions according to this thread :
        http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians

        :param mean1: Mean of the first distribution
        :param mean2: Mean of the second distribution
        :param logstd1: Logarithm of standard deviation of the first distribution
        :param logstd2: Logarithm of standard deviation of the second distribution
        :return: KL divergence of two gaussian distributions
        """

        # e ^ (2 log(x)) = x ^ 2
        var1 = tf.exp(2 * logstd1)
        var2 = tf.exp(2 * logstd2)

        kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mean1 - mean2)) / (2 * var2) - 0.5)
        return kl

    def update_policy(self, observations, actions, advantages, action_means, action_stds):
        """
        Updates policy according to sampled experience

        :param observations: Sampled observations
        :param actions: Sampled actions
        :param advantages: Sampled advantages
        :param action_means: Sampled action means
        :param action_stds: Sampled action standard deviations
        :return: None
        """

        feed_dict = {
            self.observation: observations,
            self.action: actions,
            self.advantage: advantages,
            self.old_action_mean: action_means,
            self.old_action_logstd: action_stds
        }

        # parameters
        thprev = self.gf()

        # computes fisher vector product: F * [self.pg]
        def fisher_vector_product(p):
            feed_dict[self.multiply_vector] = p
            # TODO cg-damping as hyperparameter(argument)
            CG_DAMP = 0.1
            # damping can be interpreted as "controller" of how conservative
            # approximation is
            return self.sess.run(self.fvp, feed_dict) + p * CG_DAMP

        g = self.sess.run(self.policy_gradient, feed_dict)

        # solve Ax = g, where A is FIM and g is gradient of policy parameters
        # stepdir = A_inverse * g = x
        stepdir = conjugate_gradient(fisher_vector_product, -g, self.last_dir)

        # Having computed step direction we need to compute maximal length step beta, such that
        # theta + beta * s will statisfy D_kl constraints.
        # beta = sqrt( 2*maxKL / sAs ) where A is FIM
        # Check appendix C in TRPO article to understand following lines.
        sAs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))

        beta = np.sqrt(2 * self.max_kl / sAs)

        # full step on our parameters
        fullstep = beta * stepdir
        self.last_dir = fullstep

        def loss_f(th):
            self.sff(th)
            return self.sess.run(self.loss, feed_dict)

        negative_g_dot_steppdir = -g.dot(stepdir)
        linesearch(loss_f, thprev, fullstep, negative_g_dot_steppdir / beta)

    def act(self, observations):
        """
        Samples from gaussian distribution and returns sampled action, action mean and action logstd.

        :param observations: Observations based on which policy generates action.
        :return: Sampled action, action mean and action logstd according to learned policy.
        """
        obs = np.reshape(observations, (1, self.input_dim))
        action_mean, action_logstd = self.sess.run([self.action_mean,
                                                    self.action_logstd], feed_dict={self.observation: obs})

        act = action_mean + np.exp(action_logstd) * np.random.randn(*action_logstd.shape)
        return act.ravel(), action_mean.ravel(), action_logstd.ravel()


class LinearVF(object):
    coeffs = None

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["rews"])
        al = np.arange(l).reshape(-1, 1) / 100.0

        # coarse_code = [np.ravel([self.gaussian_coarse_coding(x, -.2, .2) for x in step[-3:]]) for step in o]
        # return np.concatenate([o, o ** 2, coarse_code, al, al ** 2, np.ones((l, 1))], axis=1)
        return np.concatenate([o, o ** 2, al, al ** 2, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(featmat.T.dot(featmat) + lamb * np.identity(n_col), featmat.T.dot(returns))[0]

    def predict(self, path):
        return np.zeros(len(path["rews"])) if self.coeffs is None else self._features(
            path).dot(self.coeffs)

    def gaussian_coarse_coding(self, x, start, end, sigma=10, n_gaussians=10):
        sep = (end - start) / (n_gaussians - 1)
        return [self.gauss_pdf(x, (start + i * sep), sigma) for i in range(n_gaussians)]

    def gauss_pdf(self, x, mean, sigma):
        return np.exp(-(x - mean) ** 2 / 2 * sigma ** 2)


class NeuralVF():
    def __init__(self, session, n_hidden_layers, hidden_size, input_shape):
        self.sess = session
        with tf.variable_scope('ValueFunction'):
            self.input_x = tf.placeholder(tf.float32, [None, input_shape], 'NeuralValueFunction_x')
            self.input_y = tf.placeholder(tf.float32, [None, 1], 'NeuralValueFunction_y')

            hidden = tf.nn.relu(self.forward2(self.input_x, input_shape, hidden_size, "hidden_%d" % 0))
            for l in range(n_hidden_layers - 1):
                hidden = tf.nn.relu(self.forward2(hidden, hidden_size, hidden_size, "hidden_%d" % (l + 1)))

            self.output = self.forward2(hidden, hidden_size, 1, 'output_layer')

            reg_losses = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss = tf.reduce_mean(tf.nn.l2_loss(self.output - self.input_y)) + reg_losses

            self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

    def _features(self, path):
        o = path["obs"].astype('float32')
        return o

    def forward2(self, input_layer, input_dim, hidden_dim, scope):
        """
        Just creates and returns forward layer(part of computation graph).

        :param input_layer: Input Tensor
        :param input_dim: dimension of input
        :param hidden_dim: output layer dimension
        :param scope: name of scope
        :return: operation doing forward pass through layer Wx + b
        """
        with tf.variable_scope(scope):
            W = tf.get_variable("W", [input_dim, hidden_dim],
                                initializer=tf.random_normal_initializer(0, 1e-4),
                                regularizer=tf.contrib.layers.l2_regularizer(0.01))
            b = tf.get_variable("b", [hidden_dim], initializer=tf.constant_initializer(0.0))

        return tf.matmul(input_layer, W) + b

    def fit(self, paths):
        observations = []
        for path in paths:
            for obs in self._features(path):
                observations.append(obs)

        returns = []
        for path in paths:
            for ret in path['returns'].astype('float32'):
                returns.append([ret])

        feed_dict = {
            self.input_x: observations,
            self.input_y: returns
        }

        for i in range(5):
            _, loss_value = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

        print ('LOSS = ', loss_value)

    def predict(self, path):
        observations = self._features(path)
        feed_dict = {self.input_x: observations}
        return self.sess.run(self.output, feed_dict=feed_dict).ravel()

    def get_batch(self, X, y, batch_size):
        X, y = np.array(X), np.array(y)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        return X[indices[:batch_size]], y[indices[:batch_size]]


def get_trajs(env, agent, number_of_trajs, traj_max_lenght):
    """
    Simulate multiple trajectories with agent's policy.

    :param env: Environment in which simulations will take place.
    :param agent: Agent whose police we will be following.
    :param number_of_trajs: Number of trajectories we want to simulate.
    :param traj_max_lenght: Maximum length of one trajectory
    :return: List of dicts with following structure :
        [{
            "rews": np.array(rews),
            "action_means": np.array(action_means),
            "action_stds": np.array(action_stds),
            "acts": np.array(acts),
            "obs": np.array(obs)
        }]
    """
    trajs = []
    for i in range(number_of_trajs):
        o = env.reset()
        obs, acts, rews, action_means, action_stds = [], [], [], [], []
        for j in range(traj_max_lenght):
            a, a_mean, a_std = agent.act(o)
            a = np.array(a).ravel()
            obs.append(o)
            o, r, done, _ = env.step(a)
            action_means.append(a_mean)
            action_stds.append(a_std)
            acts.append(a)
            rews.append(r)
            if done: break
        traj = {
            "rews": np.array(rews),
            "action_means": np.array(action_means),
            "action_stds": np.array(action_stds),
            "acts": np.array(acts),
            "obs": np.array(obs)
        }
        trajs.append(traj)

    return trajs


# https://en.wikipedia.org/wiki/Conjugate_gradient_method
def conjugate_gradient(f_Ax, b, x=None, iters=10, residual_tol=1e-10):
    r = b.copy()
    p = b.copy()
    if x is None:
        x = np.zeros_like(b)
    else:
        x = x
    rdotr = r.dot(r)
    for i in xrange(iters):
        fax_p = f_Ax(p)
        alpha = rdotr / p.dot(fax_p)
        x += alpha * p
        r -= alpha * fax_p
        new_rdotr = r.dot(r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def discount_cumsum(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat(0, [tf.reshape(grad, [np.prod(var_shape(v))]) for (v, grad) in zip(var_list, grads)])


class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat(0, [tf.reshape(v, [np.prod(var_shape(v))]) for v in var_list])

    def __call__(self):
        return self.op.eval(session=self.session)


class SetFromFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(tf.float32, [total_size])
        start = 0
        assigns = []
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    print ("---------PARSER FLAGS---------")
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))
    print ("------------------------------")


from gym import wrappers


def main(_):
    print_flags()
    initialize_folders()

    env = gym.make(FLAGS.env_name)

    if FLAGS.is_train and FLAGS.is_monitor:
        def monitor_frequency_func(iteration):
            return (iteration + FLAGS.monitor_frequency) % FLAGS.monitor_frequency == 0

        env = wrappers.Monitor(env, FLAGS.log_dir + "/" + FLAGS.scope,
                               video_callable=monitor_frequency_func,
                               resume=FLAGS.is_load)

    job_queue = JoinableQueue()
    result_queue = Queue()
    e = 0

    if FLAGS.n_processes == 1 or not FLAGS.is_train:
        reacher = TRPOAgent(FLAGS.env_name, FLAGS.scope, FLAGS.max_kl, job_queue, result_queue)
        reacher.start()
    else:
        # PARALLEL TRAINING OFFERS ALMOST LINEAR IMPROVEMENT ON 2 processors
        proll = ParallelRollout(FLAGS.env_name, FLAGS.traj_len, FLAGS.n_processes, FLAGS.max_kl)
        parallel_reacher = ParallelTRPOAgent(FLAGS.env_name, FLAGS.scope, FLAGS.max_kl, job_queue, result_queue, proll)
        parallel_reacher.start()

    if FLAGS.is_load:
        job_queue.put(('load', (FLAGS.checkpoint_dir + '/' + FLAGS.scope,)))
        job_queue.join()

    try:
        while True:

            e += 1
            if FLAGS.is_train:
                print ("EPISODE =", e)
                start = time.time()
                job_queue.put(('learn', (FLAGS.gamma, FLAGS.n_trajs, FLAGS.traj_len)))
                job_queue.join()
                end = time.time()
                print ("ROLLOUT TAKES", end - start)

            obs = env.reset()
            for i in range(FLAGS.traj_len):
                job_queue.put(('act', (obs,)))
                job_queue.join()
                obs, _, done, _ = env.step(result_queue.get())
                if not FLAGS.is_train:
                    env.render()
                if done: break

            if e % FLAGS.checkpoint_freq == 0 and FLAGS.is_train:
                job_queue.put(('save', (FLAGS.checkpoint_dir + '/' + FLAGS.scope,)))
                job_queue.join()
                job_queue.put(('log', (FLAGS.log_dir + '/' + FLAGS.scope, 'my_log.json',)))
                job_queue.join()

    except KeyboardInterrupt:
        print('You pressed Ctrl+C!')
        if FLAGS.is_train and FLAGS.is_monitor:
            env.close()
        proll.end()
        parallel_reacher.join()
        sys.exit(0)
        # TODO - Parallel processes (sometime) hangs after Ctrl+C dont know how to solve it.


if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser()

    # RUN RELATED STUFF
    parser.add_argument('--env_name', type=str, default=ENV_NAME_DEFAULT,
                        help='Name of learning environment')
    parser.add_argument('--n_processes', type=int, default=N_PROCESSES_DEFAULT,
                        help='Number of processes.')
    parser.add_argument('--is_load', type=str, default=IS_LOAD_DEFAULT,
                        help='Load from "./checkpoints/scope" where scope is argument')
    parser.add_argument('--is_train', type=str, default=IS_TRAIN_DEFAULT,
                        help='If we want to update policy parameters.')
    parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQ_DEFAULT,
                        help='Frequency with which the model state is saved.')

    # REPORTING
    parser.add_argument('--scope', type=str, default=SCOPE_DEFAULT,
                        help='Name of current settings')
    parser.add_argument('--is_monitor', type=str, default=IS_MONITOR_DEFAULT,
                        help='Environment will record video into ./logs/scope')
    parser.add_argument('--monitor_frequency', type=int, default=MONITOR_FREQ_DEFAULT,
                        help='How often will monitor record video.')

    # DIRECTORIES FOR LOGS/SAVES
    parser.add_argument('--log_dir', type=str, default=LOG_DIR_DEFAULT,
                        help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR_DEFAULT,
                        help='Checkpoint directory')

    # ALGORITHM VARIABLES
    parser.add_argument('--traj_len', type=int, default=TRAJ_LEN_DEFAULT,
                        help='Lengt of the trajectory in environment before its reset')
    parser.add_argument('--n_trajs', type=int, default=N_TRAJS_DEFAULT,
                        help='Number of trajectories to estimate Q(s,a)')
    parser.add_argument('--gamma', type=float, default=GAMMA_DEFAULT,
                        help='Discount factor')
    parser.add_argument('--max_kl', type=float, default=MAX_KL_DEFAULT,
                        help='Maximum D_kl of two policies')

    FLAGS, unparsed = parser.parse_known_args()

    FLAGS.is_load = FLAGS.is_load == "True" or FLAGS.is_load == True
    FLAGS.is_monitor = FLAGS.is_monitor == "True" or FLAGS.is_monitor == True
    FLAGS.is_train = FLAGS.is_train == "True" or FLAGS.is_train == True

    tf.app.run()
