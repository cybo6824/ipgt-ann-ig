import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

DAY_SECONDS = 86400


class PairAttrPlot:
    def __init__(self, group_mat, attr_1d, clusters, absolute=True):
        self.fig = plt.figure(figsize=(12, 7))
        self.plt_x = group_mat[:, 1] / DAY_SECONDS
        self.plt_y = group_mat[:, 0] / 1000
        self.cm = mpl.cm.get_cmap("viridis")
        self.clustering = clusters
        self.prev_index = 0

        attr_1d = np.mean(attr_1d, axis=-1)
        if absolute:
            self.attr_1d = np.abs(attr_1d)
        else:
            self.attr_1d = attr_1d

        self.plot_pair(0, 10, init=True)

    def plot_pair(self, index_1, index_2, init=False):
        attr_1d_i = np.copy(self.attr_1d[index_1, index_2])
        attr_1d_norm = np.delete(attr_1d_i, [index_1, index_2])
        attr_1d_mean = np.mean(attr_1d_norm)
        attr_1d_i[index_1] = attr_1d_mean
        attr_1d_i[index_2] = attr_1d_mean
        attr_max = np.amax(attr_1d_norm)
        attr_min = np.amin(attr_1d_norm)

        plt.clf()
        cluster_1 = self.clustering == self.clustering[index_1]
        cluster_2 = self.clustering == self.clustering[index_2]

        plt.scatter(
            self.plt_x,
            self.plt_y,
            c=attr_1d_i,
            s=15,
            picker=True,
            cmap=self.cm,
            vmin=attr_min,
            vmax=attr_max,
        )
        plt.colorbar(label="Positional Importance", pad=0.01)
        plt.scatter(
            self.plt_x[cluster_1],
            self.plt_y[cluster_1],
            c=attr_1d_i[cluster_1],
            s=30,
            marker="D",
            picker=False,
            cmap=self.cm,
            vmin=attr_min,
            vmax=attr_max,
        )
        plt.scatter(
            self.plt_x[cluster_2],
            self.plt_y[cluster_2],
            c=attr_1d_i[cluster_2],
            s=30,
            marker="s",
            picker=False,
            cmap=self.cm,
            vmin=attr_min,
            vmax=attr_max,
        )
        line = plt.plot(
            [self.plt_x[index_1], self.plt_x[index_2]],
            [self.plt_y[index_1], self.plt_y[index_2]],
            c="black",
            linestyle="None",
            picker=False,
        )
        plt.scatter(
            [self.plt_x[index_1]],
            [self.plt_y[index_1]],
            c="red",
            s=35,
            marker="D",
            picker=False,
        )
        plt.scatter(
            [self.plt_x[index_2]],
            [self.plt_y[index_2]],
            c="red",
            s=35,
            marker="s",
            picker=False,
        )
        if self.clustering[index_1] == self.clustering[index_2]:
            plt.scatter(
                [self.plt_x[index_1]],
                [self.plt_y[index_1]],
                c="red",
                s=35,
                marker="s",
                picker=False,
            )
            plt.scatter(
                [self.plt_x[index_2]],
                [self.plt_y[index_2]],
                c="red",
                s=35,
                marker="D",
                picker=False,
            )
        line[0].axes.annotate(
            "",
            xytext=(self.plt_x[index_1], self.plt_y[index_1]),
            xy=(self.plt_x[index_2], self.plt_y[index_2]),
            arrowprops=dict(arrowstyle="-|>", color="black"),
            size=14,
        )

        plt.xlabel("Time (days)")
        plt.ylabel("Weight (Kg)")
        ax = plt.gca()
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if init:
            self.fig.canvas.mpl_connect("pick_event", self.pick_event)
            plt.tight_layout()
            plt.show()
        plt.draw()

    def pick_event(self, event):
        new_index = event.ind[0]
        prev_index = self.prev_index
        self.prev_index = new_index
        self.plot_pair(prev_index, new_index)


def main():
    with open("attr_ig.dat", "rb") as fh:
        attr_data = pickle.load(fh)
    with open("attr_cluster.dat", "rb") as fh:
        clusters = pickle.load(fh)["clusters"]

    PairAttrPlot(attr_data["group_mat"], attr_data["attr_1d"], clusters)


if __name__ == "__main__":
    main()
