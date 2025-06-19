from matplotlib import pyplot as plt
from matplotlib import colors

# https://www.kaggle.com/code/allegich/arc-2024-show-all-400-training-tasks


# 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
)
norm = colors.Normalize(vmin=0, vmax=9)


def plot_task(task, task_index, task_key):
    """Plots the first train and test pairs of a specified task,
    using the same color scheme as the ARC app"""

    num_train = len(task['train'])
    num_test = len(task['test'])

    w = num_train + num_test
    fig, axs = plt.subplots(2, w, figsize=(3 * w, 3 * 2))
    plt.suptitle(f'Set #{task_index}, {task_key}:', fontsize=20, fontweight='bold', y=1)

    for j in range(num_train):
        plot_one(axs[0, j], task, 'train', j, 'input')
        plot_one(axs[1, j], task, 'train', j, 'output')

    plot_one(axs[0, num_train], task, 'test', 0, 'input')
    plot_one(axs[1, num_train], task, 'test', 0, 'output')

    # add separators between samples
    # axs[1, num_train] = plt.figure(1).add_subplot(111)
    # axs[1, num_train].set_xlim([0, num_train + 1])
    # for m in range(1, num_train):
    #     axs[1, num_train].plot([m, m], [0, 1], '--', linewidth=1, color='black')
    # axs[1, num_train].plot([num_train, num_train], [0, 1], '-', linewidth=3, color='black')
    # axs[1, num_train].axis('off')

    # patch figure
    fig.patch.set_linewidth(5)
    fig.patch.set_edgecolor('black')
    fig.patch.set_facecolor('#dddddd')

    plt.tight_layout()


def plot_one(ax, task, train_or_test, i, input_or_output):
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)

    plt.setp(plt.gcf().get_axes(), xticklabels=[], yticklabels=[])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])

    ax.set_title(train_or_test + ' ' + input_or_output)


def main():
    from data.arc import prepare_arc_tasks

    train_tasks = prepare_arc_tasks('data/arc1')[0]

    task_index = 0
    task_key = list(train_tasks)[task_index]
    task = train_tasks[task_key]

    plot_task(task, task_index, task_key)
    plt.savefig(f'task_{task_key}.png', dpi=300)


if __name__ == '__main__':
    main()
