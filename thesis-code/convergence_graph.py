import json, os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse
# for i in range(1, 11):
#     file_path = './logs/grasper_prevdir300_{}/my_log.json'.format(i)
#     with open(file_path, 'r') as f:
#         data = json.loads(f.read())
#
#     data['timestamps'] = data['timestamps'][:100]
#     data['rewards'] = data['rewards'][:100]
#
#     with open(file_path, 'w') as f:
#         f.write(json.dumps(data))
#

def create_1(path, title, run_len, file_name, runs=10):

    rewards = []
    for i in range(1, runs+1):
        with open(path + str(i) + '/my_log.json', 'r') as f:
            data = json.loads(f.read())
            rewards.append(data['rewards'])

    mean1 = np.mean(rewards, 0)
    std_1 = np.std(rewards, 0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(run_len), mean1)
    ax.fill_between(range(run_len), mean1 + std_1, mean1 - std_1, alpha=0.2, hatch='/')
    ax.set_title(title)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')

    fig.savefig(os.path.join('./logs', file_name))


def compare_4(path, task, title, labels, run_len, file_name, runs=10):

    memory = path + '/' + task + '_memory_'
    prevdir = path + '/' + task + '_prevdir_'
    mrpd = path + '/' + task + '_mrpd_'
    normal = path + '/' + task + '_'

    from matplotlib.pyplot import cm

    # variable n should be number of curves to plot (I skipped this earlier thinking that it is obvious when looking at picture - srry my bad mistake xD): n=len(array_of_curves_to_plot)
    # version 1:

    color = ['r', 'g', 'b', 'black']
    # hatch = ['.', '+', "\\", "/"]
    linestyle = ['-', '--', '-.', ':']

    means = []
    stds = []
    for path in (memory, prevdir, mrpd, normal):
        rewards = []
        for i in range(1, runs+1):
            with open(path + str(i) + '/my_log.json', 'r') as f:
                data = json.loads(f.read())
                rewards.append(data['rewards'])

        means.append(np.mean(rewards,0))
        stds.append(np.std(rewards, 0))



    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(means)):
        ax.plot(range(run_len), means[i], label=labels[i], color=color[i], linestyle=linestyle[i])
        ax.fill_between(range(run_len), means[i] + stds[i], means[i] - stds[i],
                        color=color[i], alpha=0.1, linestyle=linestyle[i])
    ax.set_title(title)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)
    fig.savefig(os.path.join('./logs', file_name))


def compare_graph(path1, path2, title, label1, label2, run_len, file_name, runs=10):

    rewards = []
    for i in range(1, runs+1):
        with open(path1 + str(i) + '/my_log.json', 'r') as f:
            data = json.loads(f.read())
            rewards.append(data['rewards'])

    mean1 = np.mean(rewards,0)
    std_1 = np.std(rewards, 0)

    rewards = []
    for i in range(1, runs+1):

        with open(path2 + str(i) + '/my_log.json', 'r') as f:
            data = json.loads(f.read())
            rewards.append(data['rewards'])

    mean2 = np.mean(rewards, 0)
    std_2 = np.std(rewards, 0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(run_len), mean1, label=label1, color='g')
    ax.fill_between(range(run_len), mean1 + std_1, mean1 - std_1, facecolor='green', alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')

    ax.plot(range(run_len), mean2, label=label2,color='b',linestyle='--')
    ax.fill_between(range(run_len), mean2 + std_2, mean2 - std_2, facecolor='blue', linestyle='--', alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)
    fig.savefig(os.path.join('./logs', file_name))


# -------------MEMORY COMPARISON-------------
# REACHER 3D
compare_graph('./logs/convergence_improvements/memory_reacher3d/reacher3d_memory_',
              './logs/convergence_improvements/memory_reacher3d/reacher3d_',
              'Memory replay comparison on Reacher3d',
              'TRPO With memory replay',
              'Standard TRPO',
              100,
              'memory_replay_reacher3d')
# GRASPER 3D
compare_graph('./logs/convergence_improvements/memory_grasper3d/grasper_memory_',
              './logs/convergence_improvements/memory_grasper3d/grasper_',
              'Memory replay comparison Grasper3d',
              'TRPO With memory replay',
              'Standard TRPO',
              100,
              'memory_replay_grasper3d')
# REACHER 2D
compare_graph('./logs/convergence_improvements/memory_reacher2d/reacher_memory_',
              './logs/convergence_improvements/memory_reacher2d/reacher_',
              'Memory replay comparison on Reacher2d',
              'TRPO With memory replay',
              'Standard TRPO',
              100,
              'memory_replay_reacher2d')

# -------------PREV. DIRECTION COMPARISON-------------
# GRASPER 3D
compare_graph('./logs/convergence_improvements/prevdir_grasper3d/grasper_prevdir_',
              './logs/convergence_improvements/prevdir_grasper3d/grasper_',
              'Previous direction in CG on Grasper3d',
              'Reusing previous direction',
              'Standard TRPO',
              100,
              'prevdir_grasper3d')
# REACHER 3D
compare_graph('./logs/convergence_improvements/prevdir_reacher3d/reacher3d_prevdir_',
              './logs/convergence_improvements/prevdir_reacher3d/reacher3d_',
              'Previous direction in CG on Reacher3d',
              'Reusing previous direction',
              'Standard TRPO',
              100,
              'prevdir_reacher3d')
# REACHER 2D
compare_graph('./logs/convergence_improvements/prevdir_reacher2d/reacher_prevdir_',
              './logs/convergence_improvements/prevdir_reacher2d/reacher_',
              'Previous direction in CG on Reacher3d',
              'Reusing previous direction',
              'Standard TRPO',
              100,
              'prevdir_reacher2d')


compare_4('./logs/convergence_improvements/combined_grasper3d',
          'grasper',
          "Combined memory replay with using previous direction.",
          ['With mem. replay (MR)', 'Reusing prev. direction (PD) in CG', 'Combination of MR + PD', 'Standard TRPO'],
          100,
          'combined_grasper3d'
          )

compare_4('./logs/convergence_improvements/combined_reacher3d',
          'reacher3d',
          "Combined memory replay with using previous direction.",
          ['With mem. replay (MR)', 'Reusing prev. direction (PD) in CG', 'Combination of MR + PD', 'Standard TRPO'],
          100,
          'combined_reacher3d'
          )

compare_4('./logs/convergence_improvements/combined_reacher2d',
          'reacher',
          "Combined memory replay with using previous direction.",
          ['With mem. replay (MR)', 'Reusing prev. direction (PD) in CG', 'Combination of MR + PD', 'Standard TRPO'],
          100,
          'combined_reacher2d'
          )

create_1('./logs/convergence_improvements/combined_reacher3d/reacher3d_',
         'Reacher3d convergence',
         100,
         'reacher3d_reward_in_episodes'
         )

# create_1('./logs/combined_reacher3d/reacher3d_',
#          'Reacher3d_WALL convergence',
#          100,
#          'reacher3d_WALL_reward_in_episodes'
#          )