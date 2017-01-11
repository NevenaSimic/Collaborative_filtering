import matplotlib.pyplot as plt
import seaborn as sns


def save_heatmap_image(data, file_name):
    sns.heatmap(data, yticklabels=False, xticklabels=False)
    plt.savefig('./images/' + file_name + '.png')
    plt.clf()