
import numpy as np
import matplotlib.pyplot as plt

def plotBars(x1, x2, x3, y1, y2, y3, title, xlabel, y1label, y2label, x1ticks, x2ticks, yticks, legend_labels, savePath):
    plt.rc('text', usetex=False)
    plt.rc('font', family='Helvetica', size=13)
    tableau_colors = [
        (31, 119, 180),
        (174, 199, 232),
        (255, 127, 14),
        (255, 187, 120),
        (44, 160, 44),
        (152, 223, 138),
        (214, 39, 40),
        (255, 152, 150),
        (148, 103, 189),
        (197, 176, 213)
    ]
    for i in range(len(tableau_colors)):
        r, g, b = tableau_colors[i]
        tableau_colors[i] = (r / 255., g / 255., b / 255.)
    
    fig, ax = plt.subplots(
        figsize=(10, 12)
    )
    ax1 = ax
    
    ax1.bar(x1-0.2, y1, color=tableau_colors[0], width=0.4, label=legend_labels[0])
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label)
    ax1.legend()
    rects = ax1.patches
    
    ax2 = ax1
    ax2.bar(x2+0.2, y2, color=tableau_colors[1], width=0.4, label=legend_labels[1])
    ax2.legend()
    rects = ax2.patches
    for rect, label in zip(rects, np.append(y1, y2, axis=0)):
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, height, f"{label:.2f}", ha='center', va='bottom', fontsize=13)
    
    # ax1.set_xticks(np.append(x1, x2, axis=0), x1ticks + x2ticks)
    ax1.set_xticks(x1, minor=False)
    ax1.set_xticklabels(x1ticks + x2ticks)
    
    if x3 is not None:
        ax3 = ax1.twinx()
        ax3.plot(x3, y3, color=tableau_colors[2])
        ax3.yaxis.set_visible(False)
        for x_val, y_val in zip(x3, y3):
            ax3.text(x_val, y_val, f'{y_val*100:.2f}\%', ha='center', va='bottom', color = "r", fontsize=13)

    plt.savefig(savePath, dpi=244, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_unharden_training(x, y1, y2, save_path):
    import matplotlib.pyplot as plt
    import matplotlib

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    # Plot the lines
    ax.plot(x, y1, label='Test Acc.', marker = 'o')
    ax.plot(x, y2, label='Adv. Acc. (White-Box)', marker = 'o', linestyle = '--')

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 15}

    matplotlib.rc('font', **font)

    # Add some text for labels, title, and legend
    ax.set_xlabel('Rounds of Extraction')
    ax.set_ylabel('Accuracy')
    ax.set_title('ARU-E Extraction From FAT')
    ax.legend()

    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_agg_inclusion(save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib

    # Data for the bars
    labels = ['N = 1', 'N = 5', 'N = 10']
    bar1_values = [0.7500, 0.8999, 0.9499]
    bar2_values = [0, 4.4037e-06, 2.1138e-05,]

    # Position of the bars on the x-axis
    x = np.arange(len(labels))

    # Width of each bar
    width = 0.35

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)

    # Plot the bars
    rects1 = ax.bar(x - width/2, bar1_values, width, label='Trimmed Mean', color = 'tab:red')
    rects2 = ax.bar(x + width/2, bar2_values, width, label='Median', color = 'tab:blue')

    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 16}

    matplotlib.rc('font', **font)


    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_ylabel('Inclusion Rate', fontsize=16)
    ax.set_title('Replacement Based Adv. Unhardening', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='center right')
    ax.set_ylim([0, 1.10])

    # Add values on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.savefig(save_path, dpi=300)
    plt.show()



if "__main__" == __name__:
    methods = [
        "FedAvg", "FAT", "Injection"
    ]
    barDataPath = [
        "/home/ubuntu/Documents/jiarui/experiments/fedavg/gt_epoch200/eval",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg_adv/gt_1leaner_adv/weights/gt200/eval",
        "/home/ubuntu/Documents/jiarui/experiments/FedAvg_adv/rep_atk/rep_scale216/eval"
    ]
    origAcc = []
    advAcc = []
    for i in barDataPath:
        orig = np.load(f"{i}/all_acc.npy")
        origAcc.append(np.sum(orig) / (orig.shape[0] * orig.shape[1]))
        adv = np.load(f"{i}/all_adv_acc.npy")
        advAcc.append(np.sum(adv) / (adv.shape[0] * adv.shape[1]))
    savePath = "/home/ubuntu/Documents/jiarui/code/pFedDef/Evaluation/plots"

    plotBars(
        np.arange(len(barDataPath)),
        np.arange(len(barDataPath)),
        None,
        np.array(origAcc),
        np.array(advAcc),
        None,
        "Accuracy of Injection Attack Compared to Baseline and Injected Model",
        "Method",
        "Acc",
        "y2",
        methods,
        [],
        None,
        ["Test", "Adv"],
        f"{savePath}/barPlot"
    )


