import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from mst_clustering import HierarchicalClustering, get_graph_segments
import os
import sys
import warnings
import multiprocess as mp

warnings.simplefilter(action="ignore", category=FutureWarning)
from tqdm import tqdm


def get_clusters(filename, edge_cutoff, savefolder):
    """Extracts clusters from storm data if insulin aggregates inspired by the example
    by Jake VanderPlas on
    https://www.astroml.org/book_figures/chapter6/fig_great_wall_MST.html

    Parameters
    ----------
    filename : str
        path to the data file
    edge_cutoff : float
        Percentile cut in the distribution of lengths in the Euclidean minimum spanning tree
        to produce clusters
    savefolder : str
        path to folder in which to save clusters

    """
    data = np.genfromtxt(filename, skip_header=1, delimiter=",")
    X = data[:, 1:3]
    xmin, xmax = (0, 83000)
    ymin, ymax = (0, 83000)

    # ------------------------------------------------------------
    # Compute the MST clustering model
    print("\tFitting clustering model")
    n_neighbors = 10

    cluster_cutoff = 400
    model = HierarchicalClustering(
        n_neighbors=n_neighbors,
        edge_cutoff=edge_cutoff,
        min_cluster_size=cluster_cutoff,
    )
    model.fit(X)

    n_components = model.n_components_
    labels = model.labels_

    # ------------------------------------------------------------
    # Get the x, y coordinates of the beginning and end of each line segment
    T_x, T_y = get_graph_segments(model.X_train_, model.full_tree_)
    T_trunc_x, T_trunc_y = get_graph_segments(model.X_train_, model.cluster_graph_)

    # ------------------------------------------------------------
    # Fit a GMM to each individual cluster
    Nx = 500
    Ny = 500
    Xgrid = np.vstack(
        map(
            np.ravel,
            np.meshgrid(np.linspace(xmin, xmax, Nx), np.linspace(ymin, ymax, Ny)),
        )
    ).T
    density = np.zeros(Xgrid.shape[0])
    print("\tsaving clusters")
    for i in range(n_components):
        ind = labels == i
        Npts = ind.sum()
        Nclusters = min(12, Npts / 5)

        gmm = GaussianMixture(n_components=int(Nclusters)).fit(X[ind])
        np.savetxt(
            os.path.join(savefolder, f"Group {i}.csv"), data[ind, :], delimiter=","
        )
        dens = np.exp(gmm.score_samples(Xgrid))
        density += dens / dens.max()

    density = density.reshape((Ny, Nx))

    # ----------------------------------------------------------------------
    # Plot the results
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
    print("\tplotting points")
    norm = matplotlib.colors.Normalize(vmin=np.min(labels), vmax=np.max(labels))
    cm = matplotlib.cm
    cm.norm = norm
    cm = cm.get_cmap("tab20")
    axs[0].scatter(X[:, 1], X[:, 0], s=1, lw=0, c=labels, cmap=cm)
    axs[0].set_xlim(ymin, ymax)
    axs[0].set_ylim(xmin, xmax)
    axs[0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0].set_xlabel("(nm)")
    axs[0].set_ylabel("(nm)")

    print("\tplotting Tree1")
    axs[1].plot(T_y, T_x, c="k", lw=0.5)
    axs[1].set_xlim(ymin, ymax)
    axs[1].set_ylim(xmin, xmax)
    axs[1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1].set_xlabel("(nm)")
    print("\tplotting Tree2")

    axs[2].plot(T_trunc_y, T_trunc_x, c="k", lw=0.5)
    axs[2].imshow(
        density.T, origin="lower", cmap=plt.cm.hot_r, extent=[ymin, ymax, xmin, xmax]
    )

    axs[2].set_xlim(ymin, ymax)
    axs[2].set_ylim(xmin, xmax)
    axs[2].set_xlabel("(nm)")
    plt.savefig(os.path.join(savefolder, "plot overview"), dpi=500)


def Group_analysis(main_datafolder):
    """Computes the area as a function of time by density estimation
    in each frame and filtering by clustering

    Parameters
    ----------
    main_datafolder : str
        path to folder in which to find clusters to analyze

    """
    subfolders_paths = [
        os.path.join(main_datafolder, i)
        for i in os.listdir(main_datafolder)
        if i[0] != "." and i[:5] == "Group"
    ]

    def run_on_cluster(gn):
        filename = gn.split("/")[-1].split(".")[0]
        print(f"\tMeasuring aggregate time-seires for {filename}")

        data = np.genfromtxt(gn, skip_header=1, delimiter=",")
        frames = []
        xmin, xmax = (np.min(data[:, 1]), np.max(data[:, 1]))
        ymin, ymax = (np.min(data[:, 2]), np.max(data[:, 2]))
        for t in tqdm(range(0, int(np.max(data[:, 0])))):
            # if t % 50 == 0:
            # print(t)
            X = data[:, 1:3][data[:, 0] < t]
            if len(X) > 2:
                # ------------------------------------------------------------
                # Compute the MST clustering model
                edge_cutoff = 0.99
                cluster_cutoff = 10
                distcut = 400
                model = HierarchicalClustering(
                    n_neighbors=np.min([10, len(X) - 1]),
                    edge_cutoff=edge_cutoff,
                    min_cluster_size=cluster_cutoff,
                )
                # try:
                model.fit(X)
                n_components, labels, clustergraph = model.compute_clusters(
                    min_cluster_size=cluster_cutoff, distcutoff=distcut
                )

                # ------------------------------------------------------------
                # Get the x, y coordinates of the beginning and end of each line segment
                T_x, T_y = get_graph_segments(model.X_train_, model.full_tree_)
                T_trunc_x, T_trunc_y = get_graph_segments(model.X_train_, clustergraph)
                if np.all(np.bincount(labels[labels > -1]) == 0):
                    frames.append((None, None, None, None, 0, t))
                else:
                    mostocc = np.argmax(np.bincount(labels[labels > -1]))
                    # Fit a GMM to biggest cluster
                    Nx = 500
                    Ny = 500
                    Xgrid = np.vstack(
                        map(
                            np.ravel,
                            np.meshgrid(
                                np.linspace(xmin, xmax, Nx),
                                np.linspace(ymin, ymax, Ny),
                            ),
                        )
                    ).T
                    density = np.zeros(Xgrid.shape[0])
                    ind = labels == mostocc
                    Npts = ind.sum()
                    Nclusters = min(25, Npts / 5)

                    gmm = GaussianMixture(n_components=int(Nclusters)).fit(X[ind])
                    dens = np.exp(gmm.score_samples(Xgrid))
                    density += dens / dens.max()
                    density = density.reshape((Ny, Nx))

                    gridarea = ((xmax - xmin) / Nx) * ((ymax - ymin) / Ny)
                    size = np.sum(density > np.mean(density)) * gridarea
                    frames.append(
                        (
                            X[labels == mostocc],
                            T_trunc_x,
                            T_trunc_y,
                            density,
                            size,
                            t,
                        )
                    )
                # except:
                # frames.append((None, None, None, None, -1, t))
        import matplotlib.animation as animation

        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(231, aspect="equal")
        ax2 = fig.add_subplot(232, aspect="equal")
        ax3 = fig.add_subplot(212, aspect="auto")
        ax4 = fig.add_subplot(233, aspect="equal")
        ts = np.array([frames[i][5] for i in range(len(frames))])
        sizes = np.array([frames[i][4] for i in range(len(frames))])

        def animator(i):
            if i % 50 == 0:
                print(f"\tGroup {filename} reached frame {i} out of {len(frames)}")
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
            if frames[i][0] is None:
                ax3.set_xlim((0, frames[-1][5]))
                ax3.set_ylim(0, frames[-1][4])
                alin = ax3.plot([ts[:i]], [sizes[:i]], "ko")
                ax3.set(xlabel="Frame", ylabel=f"Size in $nm^2$")
            else:
                ax1.scatter(frames[i][0][:, 0], frames[i][0][:, 1], c="k", s=0.5)
                ax2.plot(frames[i][1], frames[i][2], c="k", lw=0.5)
                ax1.set_xlim((xmin, xmax))
                ax1.set_ylim((ymin, ymax))

                ax1.imshow(
                    frames[i][3],
                    origin="lower",
                    cmap=plt.cm.hot_r,
                    extent=[xmin, xmax, ymin, ymax],
                )
                ax2.set_xlim((xmin, xmax))
                ax2.set_ylim((ymin, ymax))

                ax4.imshow(
                    frames[i][3] > np.mean(frames[i][3]),
                    origin="lower",
                    extent=[xmin, xmax, ymin, ymax],
                )
                ax4.set_xlim((xmin, xmax))
                ax4.set_ylim((ymin, ymax))

                # fit nonzero growth
                def Logistic(x, vmax, k, kD, v0):
                    return vmax / (1 + np.exp(-k * (x - kD))) + v0

                def line(x, x0, Rate, Offset):
                    return (x - x0) * Rate + Offset

                def twoline(x, x01, r1, Offset, r2, switch):
                    x02 = switch - (switch - x01) * r1 / r2
                    if x < switch:
                        return (x - x01) * r1 + Offset
                    else:
                        return (x - x02) * r2 + Offset

                mask = sizes > 0
                x, y = ts[mask][:i], sizes[mask][:i]
                if i == len(frames) - 1:
                    np.savetxt(
                        os.path.join(main_datafolder, f"{filename} Growth curve"),
                        np.array([ts, sizes]).T,
                    )
                    ax3.set_xlim((0, frames[-1][5]))
                    ax3.set_ylim(0, frames[-1][4])
                    alin = ax3.plot([ts[:i]], [sizes[:i]], "ko")
                    ax3.set(xlabel="Frame", ylabel="Size in pixels")
                else:
                    ax3.set_xlim((0, frames[-1][5]))
                    ax3.set_ylim(0, frames[-1][4])
                    alin = ax3.plot([ts[:i]], [sizes[:i]], "ko")
                    ax3.set(xlabel="Frame", ylabel="Size in pixels")

        print(f"\tMaking movie for {filename}")
        ani = animation.FuncAnimation(
            fig, animator, interval=400, blit=False, frames=len(frames)
        )

        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=10, metadata=dict(artist="Me"), bitrate=1800)
        print(os.path.join(main_datafolder, f"Clustered {filename}.mp4"))
        ani.save(
            os.path.join(main_datafolder, f"Clustered {filename}.mp4"), writer=writer
        )
        return None

    with mp.get_context("spawn").Pool(mp.cpu_count()) as f:
        output = f.map(run_on_cluster, subfolders_paths)


# initialize variable
if __name__ == "__main__":

    clusterdone = True
    moviefile = sys.argv[1]
    cutoff = np.float(sys.argv[2])
    directory_name = moviefile[:-4]
    if not os.path.isdir(directory_name):
        os.mkdir(directory_name)

    # run clustering
    if not os.path.isfile(os.path.join(directory_name, "plot overview.png")):
        while clusterdone:

            print(f"Making clusters with cutoff {cutoff}")
            get_clusters(moviefile, cutoff, directory_name)

            textin = input("y if cluster ok, newedgecutoff if not ")

            if not textin == "y":
                clusterdone = True
                cutoff = np.float(textin)
            else:
                clusterdone = False

    # run group analysis
    print("Running single-aggregate analysis")
    Group_analysis(directory_name)
