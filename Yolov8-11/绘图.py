import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体为黑体（Windows 系统）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示异常（如果有负数坐标或标签）
plt.rcParams['axes.unicode_minus'] = False


def plot_metrics_and_loss(experiment_names, custom_legend_names, metrics_info, loss_info, metrics_subplot_layout,
                          loss_subplot_layout,
                          metrics_figure_size=(15, 10), loss_figure_size=(15, 10), base_directory='runs/train'):
    # Plot metrics
    plt.figure(figsize=metrics_figure_size)
    markers = ['o', 's', '^', 'D', 'v', '*']  # 定义标记样式
    target_marker_count = 6  # 目标标记数量
    target_model = 'YOLO11-ARE-PCCDSConv'
    target_index = custom_legend_names.index(target_model)
    linewidth = 1  # 统一设置线条粗细
    for i, (metric_name, title) in enumerate(metrics_info):
        plt.subplot(*metrics_subplot_layout, i + 1)
        for j, (name, legend_name) in enumerate(zip(experiment_names, custom_legend_names)):
            file_path = os.path.join(base_directory, name, 'results.csv')
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == metric_name][0]
            markevery = len(data[column_name]) // target_marker_count
            markersize = 8 if legend_name == target_model else 5
            color = 'red' if legend_name == target_model else None
            plt.plot(data[column_name], marker=markers[j % len(markers)], linestyle='-', label=legend_name,
                     markevery=markevery, linewidth=linewidth, markersize=markersize, color=color)
        plt.xlabel('Epoch')
        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels)
    plt.tight_layout()
    metrics_filename = 'metrics_curves.png'
    plt.savefig(metrics_filename)
    plt.show()

    # Plot loss
    plt.figure(figsize=loss_figure_size)
    for i, (loss_name, title) in enumerate(loss_info):
        plt.subplot(*loss_subplot_layout, i + 1)
        for j, (name, legend_name) in enumerate(zip(experiment_names, custom_legend_names)):
            file_path = os.path.join(base_directory, name, 'results.csv')
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == loss_name][0]
            markevery = len(data[column_name]) // target_marker_count
            markersize = 8 if legend_name == target_model else 5
            color = 'red' if legend_name == target_model else None
            plt.plot(data[column_name], marker=markers[j % len(markers)], linestyle='-', label=legend_name,
                     markevery=markevery, linewidth=linewidth, markersize=markersize, color=color)
        plt.xlabel('Epoch')
        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels)
    plt.tight_layout()
    loss_filename = 'loss_curves.png'
    plt.savefig(loss_filename)
    plt.show()

    return metrics_filename, loss_filename


# Metrics to plot
metrics_info = [
    ('metrics/precision(B)', 'Precision'),
    ('metrics/recall(B)', 'Recall'),
    ('metrics/mAP50(B)', 'mAP at IoU=0.5'),
    ('metrics/mAP50-95(B)', 'mAP for IoU Range 0.5-0.95')
]

# Loss to plot
loss_info = [
    ('train/box_loss', 'Training Box Loss'),
    ('train/cls_loss', 'Training Classification Loss'),
    ('train/dfl_loss', 'Training DFL Loss'),
    ('val/box_loss', 'Validation Box Loss'),
    ('val/cls_loss', 'Validation Classification Loss'),
    ('val/dfl_loss', 'Validation DFL Loss')
]

# 自定义图例名称
custom_legend_names = ['YOLO11', 'YOLOvn10', 'YOLOv8', 'YOLOv6', 'YOLOv5', 'YOLO11-ARE-PCCDSConv']

# Plot the metrics and loss from multiple experiments
metrics_filename, loss_filename = plot_metrics_and_loss(
    experiment_names=['exp', 'exp13', 'exp18', 'exp23', 'exp25', 'exp24'],
    custom_legend_names=custom_legend_names,
    metrics_info=metrics_info,
    loss_info=loss_info,
    metrics_subplot_layout=(2, 2),
    loss_subplot_layout=(2, 3)
)
