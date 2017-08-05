import argparse
import os, json
import errno
import matplotlib as mpl
import numpy as np
mpl.use('Agg')
import matplotlib.pyplot as plt



def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def process_manifest(json_data):
    data = {}
    data['stat_files'] = [json_data['stats']]
    data['video_files'] = [video[0] for video in json_data['videos']]

    return data

def get_data(path):
    data = None
    data_files = []
    for filename in os.listdir(path):
        if 'manifest' in filename:
            data_files.append(filename)
    for file in data_files:
        with open(os.path.join(path, file)) as content_file:
            content = process_manifest(json.loads(content_file.read()))
            if data is None:
                data = content
            else:
                data['stat_files'].extend(content['stat_files'])
                data['video_files'].extend(content['video_files'])

    return data

def merge_stat_files(path, stat_files):
    data = None
    for file in stat_files:
        with open(os.path.join(path, file)) as content_file:
            content = json.loads(content_file.read())
            if data is None:
                data = {}
                data['time_from_start'] = [timestamp - content['initial_reset_timestamp']
                                           for timestamp in content['timestamps']]
                data['rewards'] = content['episode_rewards']
            else:
                last = data['time_from_start'][-1]
                data['time_from_start'].extend([last + timestamp - content['initial_reset_timestamp']
                                                for timestamp in content['timestamps']])
                data['rewards'].extend(content['episode_rewards'])

    return data

def create_graphs(path, data):
    x_time = data['timestamps']
    y = data['rewards']

    x_time = [x_time[i] - x_time[0] for i in range(len(x_time))]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_time, y, 'b')
    ax.set_title("Training")
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Reward')

    fig.savefig(os.path.join(path, 'merged', 'reward_in_time'))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(x_time)), y)
    ax.set_title("Training")
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    fig.savefig(os.path.join(path, 'merged', 'reward_in_episodes'))


def merge_videos(path):
    print os.system("cd {} && find *.mp4 | sed 's:\ :\\\ :g'| sed 's/^/file /' > fl.txt; ffmpeg -f concat -i fl.txt -c copy merged/merged_videos.mp4; rm fl.txt".format(os.path.join(path)))


def create_time_comparison_graph(path, stats1, stats2):
    x_time1 = stats1['timestamps']
    x_time1 = [x_time1[i] - x_time1[0] for i in range(len(x_time1))]
    y1 = stats1['rewards']

    x_time2 = stats2['timestamps']
    x_time2 = np.array([x_time2[i] - x_time2[0] for i in range(len(x_time2))])
    y2 = stats2['rewards']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(x_time1)), x_time1, label='Parallel')
    ax.set_title("Parallel vs. sequential with {}x speedup".format(round(float(x_time2[-1])/x_time1[-1], 3)))
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Seconds')

    ax.plot(range(len(x_time2)), x_time2, label='Serial')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2)
    fig.savefig(os.path.join(path, 'merged', 'time_comparison'))

def compare_convergence(path, stats1, stats2):
    x_time1 = stats1['timestamps']
    y1 = stats1['rewards']

    x_time2 = stats2['timestamps']
    y2 = stats2['rewards']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(x_time1)), y2, label='Using previous direction')
    ax.set_title("Previous direction convergence comparison")
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Seconds')

    ax.plot(range(len(x_time1)), y1, label='Using zero vector')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)
    fig.savefig(os.path.join(path, 'merged', 'convergence'))


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    PATH_DEFAULT = None

    # RUN RELATED STUFF
    parser.add_argument('--path', type=str, default=PATH_DEFAULT,
                        help='Path to log folder with video and stats')
    parser.add_argument('--path2', type=str, default=PATH_DEFAULT,
                        help='Will compare stats in time for path with stats in path2 (assumes parallel is first path)')

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.path is None:
        print('NO PATH ERROR! \n Try following command: \n\n python run_stats.py --path /logs/scope_name \n')
    else:
        # data_files = get_data(FLAGS.path)
        # merged_stats = merge_stat_files(FLAGS.path, data_files['stat_files'])
        # # CREATES FOLDER IN WHICH IT WILL SAVE MERGED STUFF
        # make_sure_path_exists(os.path.join(FLAGS.path, "merged"))

        make_sure_path_exists(os.path.join(FLAGS.path, 'merged'))

        file_path1 = os.path.join(FLAGS.path, 'my_log.json')

        if os.path.isfile(file_path1):
            with open(file_path1, "r") as f:
                data = json.loads(f.read())

        # CREATE PERFORMANCE GRAPHS
        create_graphs(FLAGS.path, data)
        # MERGES VIDEOS INTO ONE
        merge_videos(FLAGS.path)

        if FLAGS.path2:
            file_path2 = os.path.join(FLAGS.path2, 'my_log.json')
            if os.path.isfile(file_path2):
                with open(file_path2, "r") as f:
                    data2 = json.loads(f.read())

            create_time_comparison_graph(FLAGS.path, data, data2)
            compare_convergence(FLAGS.path, data, data2)

