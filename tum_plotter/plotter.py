#!/usr/bin/env python3
"""
tum_plotter  Visualise and compare SLAM trajectories from TUM files.

Usage:
  # Plot a single trajectory
  tum_plotter path/to/trajectory.tum

  # Compare multiple algorithms against ground truth
  tum_plotter --gt ground_truth.tum \
              --traj lidarslam.tum kiss_icp.tum odometry.tum \
              --labels "LiDAR SLAM" "KISS-ICP" "Odometry"

  # Save publication-ready figure
  tum_plotter --gt ground_truth.tum --traj lidarslam.tum --save plot.pdf

  # 3D plot
  tum_plotter --gt ground_truth.tum --traj lidarslam.tum --3d
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# -- TUM file loader -----------------------------------------------------------

def load_tum(path):
    """Load a TUM trajectory file. Returns (timestamps, xyz array)."""
    timestamps, xyz = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                timestamps.append(float(parts[0]))
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(timestamps), np.array(xyz)


def associate(t_est, xyz_est, t_gt, xyz_gt, max_diff=0.5):
    """Associate estimated and ground truth trajectories by timestamp."""
    matched_est, matched_gt = [], []
    for i, t_e in enumerate(t_est):
        diffs = np.abs(t_gt - t_e)
        idx = np.argmin(diffs)
        if diffs[idx] < max_diff:
            matched_est.append(xyz_est[i])
            matched_gt.append(xyz_gt[idx])
    return np.array(matched_est), np.array(matched_gt)


def align_umeyama(est, gt, correct_scale=False):
    """Umeyama alignment. Returns aligned trajectory."""
    mu_e = est.mean(axis=0)
    mu_g = gt.mean(axis=0)
    est_c = est - mu_e
    gt_c = gt - mu_g

    sigma_e = np.mean(np.sum(est_c**2, axis=1))
    if sigma_e < 1e-10:
        return est

    H = est_c.T @ gt_c / len(est)
    U, S, Vt = np.linalg.svd(H)
    det = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(det)])
    R = Vt.T @ D @ U.T

    if correct_scale:
        c = float(np.sum(S * np.diag(D))) / sigma_e
    else:
        c = 1.0

    t = mu_g - c * R @ mu_e
    aligned = (c * (R @ est.T)).T + t
    return aligned


def compute_ape(est_aligned, gt_matched):
    """Compute APE errors."""
    errors = np.linalg.norm(est_aligned - gt_matched, axis=1)
    return errors


# -- Plotting ------------------------------------------------------------------

# Colour palette � clean and publication-ready
COLOURS = [
    '#2196F3',  # blue
    '#F44336',  # red
    '#4CAF50',  # green
    '#FF9800',  # orange
    '#9C27B0',  # purple
    '#00BCD4',  # cyan
    '#795548',  # brown
    '#607D8B',  # blue-grey
]

GT_COLOUR = '#212121'  # near black for ground truth


def plot_2d(args, gt_data, traj_data):
    """2D top-down trajectory comparison plot."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    n_traj = len(traj_data)
    has_gt = gt_data is not None

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor('white')

    if n_traj > 0 and has_gt:
        gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
        ax_traj = fig.add_subplot(gs[0])
        ax_stats = fig.add_subplot(gs[1])
        ax_stats.axis('off')
    else:
        ax_traj = fig.add_subplot(111)

    ax_traj.set_facecolor('#fafafa')
    ax_traj.grid(True, alpha=0.3, linewidth=0.5)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('X (m)', fontsize=11)
    ax_traj.set_ylabel('Y (m)', fontsize=11)

    title = args.title if args.title else 'Trajectory Comparison'
    ax_traj.set_title(title, fontsize=13, fontweight='bold', pad=12)

    handles = []

    # Plot ground truth
    if has_gt:
        t_gt, xyz_gt = gt_data
        line, = ax_traj.plot(
            xyz_gt[:, 0], xyz_gt[:, 1],
            color=GT_COLOUR, linewidth=2.0,
            label='Ground Truth', zorder=5, alpha=0.8
        )
        # Start marker
        ax_traj.scatter(xyz_gt[0, 0], xyz_gt[0, 1],
                       color=GT_COLOUR, s=80, zorder=6, marker='o')
        handles.append(line)

    # Plot estimated trajectories
    stats_rows = []
    for i, (label, (t_est, xyz_est)) in enumerate(traj_data):
        colour = COLOURS[i % len(COLOURS)]

        if has_gt:
            t_gt, xyz_gt = gt_data
            est_m, gt_m = associate(t_est, xyz_est, t_gt, xyz_gt,
                                    max_diff=args.t_max_diff)
            if len(est_m) > 3:
                correct_scale = getattr(args, 'correct_scale', False)
                est_aligned = align_umeyama(est_m, gt_m, correct_scale)
                errors = compute_ape(est_aligned, gt_m)

                # Plot aligned trajectory
                line, = ax_traj.plot(
                    est_aligned[:, 0], est_aligned[:, 1],
                    color=colour, linewidth=1.5, alpha=0.85,
                    label=label, zorder=4
                )
                handles.append(line)

                align_str = "Sim(3)" if correct_scale else "SE(3)"
                stats_rows.append([
                    label,
                    f"{errors.mean():.4f}",
                    f"{np.sqrt(np.mean(errors**2)):.4f}",
                    f"{errors.max():.4f}",
                    f"{len(est_m)}",
                    align_str
                ])
            else:
                line, = ax_traj.plot(
                    xyz_est[:, 0], xyz_est[:, 1],
                    color=colour, linewidth=1.5, alpha=0.85,
                    label=f"{label} (no match)", zorder=4,
                    linestyle='--'
                )
                handles.append(line)
                stats_rows.append([label, "N/A", "N/A", "N/A", "0", "N/A"])
        else:
            line, = ax_traj.plot(
                xyz_est[:, 0], xyz_est[:, 1],
                color=colour, linewidth=1.5, alpha=0.85,
                label=label, zorder=4
            )
            handles.append(line)

    ax_traj.legend(handles=handles, loc='best', fontsize=9,
                   framealpha=0.9, edgecolor='#cccccc')

    # Stats table
    if has_gt and stats_rows and n_traj > 0:
        col_labels = ['Algorithm', 'APE Mean (m)', 'APE RMSE (m)',
                      'APE Max (m)', 'Matched', 'Alignment']
        table = ax_stats.table(
            cellText=stats_rows,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.8)

        # Style header
        for j in range(len(col_labels)):
            table[(0, j)].set_facecolor('#212121')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Colour rows
        for i, row in enumerate(stats_rows):
            colour = COLOURS[i % len(COLOURS)]
            for j in range(len(col_labels)):
                table[(i+1, j)].set_facecolor(colour + '22')

        ax_stats.set_title('APE Statistics', fontsize=11,
                           fontweight='bold', pad=8)

    plt.tight_layout()
    return fig


def plot_3d(args, gt_data, traj_data):
    """3D trajectory plot."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    title = args.title if args.title else '3D Trajectory Comparison'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    if gt_data is not None:
        t_gt, xyz_gt = gt_data
        ax.plot(xyz_gt[:, 0], xyz_gt[:, 1], xyz_gt[:, 2],
                color=GT_COLOUR, linewidth=2.0, label='Ground Truth', alpha=0.8)

    for i, (label, (t_est, xyz_est)) in enumerate(traj_data):
        colour = COLOURS[i % len(COLOURS)]
        if gt_data is not None:
            t_gt, xyz_gt = gt_data
            est_m, gt_m = associate(t_est, xyz_est, t_gt, xyz_gt,
                                    max_diff=args.t_max_diff)
            if len(est_m) > 3:
                correct_scale = getattr(args, 'correct_scale', False)
                est_aligned = align_umeyama(est_m, gt_m, correct_scale)
                ax.plot(est_aligned[:, 0], est_aligned[:, 1], est_aligned[:, 2],
                        color=colour, linewidth=1.5, label=label, alpha=0.85)
            else:
                ax.plot(xyz_est[:, 0], xyz_est[:, 1], xyz_est[:, 2],
                        color=colour, linewidth=1.5, label=label, alpha=0.85,
                        linestyle='--')
        else:
            ax.plot(xyz_est[:, 0], xyz_est[:, 1], xyz_est[:, 2],
                    color=colour, linewidth=1.5, label=label, alpha=0.85)

    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_ape_over_time(args, gt_data, traj_data):
    """APE error over time plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor('#fafafa')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('APE (m)', fontsize=11)
    title = args.title if args.title else 'APE Over Time'
    ax.set_title(title, fontsize=13, fontweight='bold')

    if gt_data is None:
        print("APE over time requires ground truth (--gt)")
        return fig

    t_gt, xyz_gt = gt_data

    for i, (label, (t_est, xyz_est)) in enumerate(traj_data):
        colour = COLOURS[i % len(COLOURS)]
        est_m, gt_m = associate(t_est, xyz_est, t_gt, xyz_gt,
                                max_diff=args.t_max_diff)
        if len(est_m) > 3:
            correct_scale = getattr(args, 'correct_scale', False)
            est_aligned = align_umeyama(est_m, gt_m, correct_scale)
            errors = compute_ape(est_aligned, gt_m)

            # Use relative time
            t_matched = []
            t_gt_arr = np.array([g for g in t_gt])
            for t_e in t_est:
                diffs = np.abs(t_gt - t_e)
                idx = np.argmin(diffs)
                if diffs[idx] < args.t_max_diff:
                    t_matched.append(t_gt[idx])

            t_rel = np.array(t_matched) - t_gt[0]
            if len(t_rel) == len(errors):
                ax.plot(t_rel, errors, color=colour, linewidth=1.2,
                        label=label, alpha=0.85)

    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='tum_plotter � Visualise and compare SLAM trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single trajectory
  tum_plotter trajectory.tum

  # Compare against ground truth
  tum_plotter --gt ground_truth.tum --traj slam.tum kiss_icp.tum

  # With custom labels
  tum_plotter --gt gt.tum --traj a.tum b.tum --labels "LiDAR SLAM" "KISS-ICP"

  # With scale correction (for monocular visual SLAM)
  tum_plotter --gt gt.tum --traj stella.tum --correct-scale

  # Save to file
  tum_plotter --gt gt.tum --traj slam.tum --save figure.pdf

  # 3D plot
  tum_plotter --gt gt.tum --traj slam.tum --3d

  # APE over time
  tum_plotter --gt gt.tum --traj slam.tum --mode ape-time
        """
    )

    parser.add_argument('traj_positional', nargs='?',
                        help='Single TUM file (positional argument)')
    parser.add_argument('--gt', help='Ground truth TUM file')
    parser.add_argument('--traj', nargs='+', help='Estimated trajectory TUM file(s)')
    parser.add_argument('--labels', nargs='+', help='Labels for each trajectory')
    parser.add_argument('--title', help='Plot title')
    parser.add_argument('--save', help='Save figure to file (pdf, png, svg)')
    parser.add_argument('--3d', dest='plot_3d', action='store_true',
                        help='3D trajectory plot')
    parser.add_argument('--mode', choices=['traj', 'ape-time'],
                        default='traj', help='Plot mode (default: traj)')
    parser.add_argument('--correct-scale', dest='correct_scale',
                        action='store_true',
                        help='Apply scale correction (Sim3) � use for monocular SLAM')
    parser.add_argument('--t-max-diff', dest='t_max_diff', type=float,
                        default=0.5,
                        help='Max timestamp difference for association (default: 0.5s)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (useful with --save)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figure (default: 150)')

    args = parser.parse_args()

    # Handle positional single-file mode
    traj_files = []
    if args.traj_positional:
        traj_files = [args.traj_positional]
    elif args.traj:
        traj_files = args.traj

    if not traj_files and not args.gt:
        parser.print_help()
        sys.exit(1)

    # Load ground truth
    gt_data = None
    if args.gt:
        if not os.path.exists(args.gt):
            print(f"Error: Ground truth file not found: {args.gt}")
            sys.exit(1)
        t_gt, xyz_gt = load_tum(args.gt)
        gt_data = (t_gt, xyz_gt)
        print(f"Loaded GT: {Path(args.gt).name} � {len(t_gt)} poses")

    # Load trajectories
    labels = args.labels or []
    traj_data = []
    for i, traj_file in enumerate(traj_files):
        if not os.path.exists(traj_file):
            print(f"Warning: Trajectory file not found: {traj_file}")
            continue
        t_est, xyz_est = load_tum(traj_file)
        label = labels[i] if i < len(labels) else Path(traj_file).stem
        traj_data.append((label, (t_est, xyz_est)))
        print(f"Loaded: {label} � {len(t_est)} poses")

    if not gt_data and not traj_data:
        print("Error: No valid trajectory files loaded.")
        sys.exit(1)

    # If only GT provided, treat it as a trajectory to plot
    if gt_data and not traj_data:
        traj_data = [('Trajectory', gt_data)]
        gt_data = None

    import matplotlib
    if args.no_show or args.save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Generate plot
    if args.plot_3d:
        fig = plot_3d(args, gt_data, traj_data)
    elif args.mode == 'ape-time':
        fig = plot_ape_over_time(args, gt_data, traj_data)
    else:
        fig = plot_2d(args, gt_data, traj_data)

    # Save
    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches='tight',
                    facecolor='white')
        print(f"Saved: {args.save}")

    # Show
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
