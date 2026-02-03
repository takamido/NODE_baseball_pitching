import numpy as np
import matplotlib.pyplot as plt

def _set_axes_equal_3d(ax, pts):
    pts = np.asarray(pts)
    xlim = (pts[:,0].min(), pts[:,0].max())
    ylim = (pts[:,1].min(), pts[:,1].max())
    zlim = (pts[:,2].min(), pts[:,2].max())
    xr, yr, zr = xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]
    m = max(xr, yr, zr) or 1.0
    xm, ym, zm = (xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2, (zlim[0]+zlim[1])/2
    ax.set_xlim(xm - m/2, xm + m/2)
    ax.set_ylim(ym - m/2, ym + m/2)
    ax.set_zlim(zm - m/2, zm + m/2)

def plot_stick_figure(
    positions_T_J_3, edge_index_E_2,
    frame_stride=20, elev=20, azim=-60,
    color="blue", show_nodes=True, ax=None):
    T, J, _ = positions_T_J_3.shape
    frames = list(range(0, T, frame_stride))
    if frames[-1] != T - 1:
        frames.append(T - 1)
    all_pts = positions_T_J_3.reshape(-1, 3)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    for f in frames:
        xyz = positions_T_J_3[f]
        for (i, j) in edge_index_E_2:
            xi, yi, zi = xyz[i]
            xj, yj, zj = xyz[j]
            ax.plot([xi, xj], [yi, yj], [zi, zj],
                    linewidth=1, color=color,
                    alpha=max(0.15, 1.0 - 0.8*(f/(T-1))))
        if show_nodes:
            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=6, c=color)

    ax.view_init(elev=elev, azim=azim)
    _set_axes_equal_3d(ax, all_pts)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"Stick figure (every {frame_stride} frames)")
    return ax

