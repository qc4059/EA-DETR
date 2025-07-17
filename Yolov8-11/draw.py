# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 设置中文字体为黑体（Windows 系统）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # 解决负号显示异常（如果有负数坐标或标签）
# plt.rcParams['axes.unicode_minus'] = False
#
#
# def plot_metrics_and_loss(experiment_names, custom_legend_names, metrics_info, loss_info, metrics_subplot_layout,
#                           loss_subplot_layout,
#                           metrics_figure_size=(15, 10), loss_figure_size=(15, 10), base_directory='runs/train'):
#     # Plot metrics
#     plt.figure(figsize=metrics_figure_size)
#     for i, (metric_name, title) in enumerate(metrics_info):
#         plt.subplot(*metrics_subplot_layout, i + 1)
#         for name, legend_name in zip(experiment_names, custom_legend_names):
#             file_path = os.path.join(base_directory, name, 'results.csv')
#             data = pd.read_csv(file_path)
#             column_name = [col for col in data.columns if col.strip() == metric_name][0]
#             plt.plot(data[column_name], label=legend_name)
#         plt.xlabel('Epoch')
#         plt.title(title)
#         plt.legend()
#     plt.tight_layout()
#     metrics_filename = 'metrics_curves.png'
#     plt.savefig(metrics_filename)
#     plt.show()
#
#     # Plot loss
#     plt.figure(figsize=loss_figure_size)
#     for i, (loss_name, title) in enumerate(loss_info):
#         plt.subplot(*loss_subplot_layout, i + 1)
#         for name, legend_name in zip(experiment_names, custom_legend_names):
#             file_path = os.path.join(base_directory, name, 'results.csv')
#             data = pd.read_csv(file_path)
#             column_name = [col for col in data.columns if col.strip() == loss_name][0]
#             plt.plot(data[column_name], label=legend_name)
#         plt.xlabel('Epoch')
#         plt.title(title)
#         plt.legend()
#     plt.tight_layout()
#     loss_filename = 'loss_curves.png'
#     plt.savefig(loss_filename)
#     plt.show()
#
#     return metrics_filename, loss_filename
#
#
# # Metrics to plot
# metrics_info = [
#     ('metrics/precision(B)', 'Precision'),
#     ('metrics/recall(B)', 'Recall'),
#     ('metrics/mAP50(B)', 'mAP at IoU=0.5'),
#     ('metrics/mAP50-95(B)', 'mAP for IoU Range 0.5-0.95')
# ]
#
# # Loss to plot
# loss_info = [
#     ('train/box_loss', 'Training Box Loss'),
#     ('train/cls_loss', 'Training Classification Loss'),
#     ('train/dfl_loss', 'Training DFL Loss'),
#     ('val/box_loss', 'Validation Box Loss'),
#     ('val/cls_loss', 'Validation Classification Loss'),
#     ('val/dfl_loss', 'Validation DFL Loss')
# ]
#
# # 自定义图例名称
# custom_legend_names = ['YOLO11', 'YOLOv10n', 'YOLOv8', 'YOLOv6','YOLOv5', 'YOLOv3-tiny', 'YOLO11-ARE-PCCDSConv', ]
# # custom_legend_names = ['YOLO11',  'YOLO11-PCCDSConv','YOLO11-ARE', 'YOLO11-ARE-PCCDSConv']
#
# # Plot the metrics and loss from multiple experiments
# metrics_filename, loss_filename = plot_metrics_and_loss(
#     experiment_names=['exp', 'exp13', 'exp34', 'exp23', 'exp35',  'exp31', 'exp33',],
#     # experiment_names=['exp', 'exp8', 'exp26','exp33',],
#     custom_legend_names=custom_legend_names,
#     metrics_info=metrics_info,
#     loss_info=loss_info,
#     metrics_subplot_layout=(2, 2),
#     loss_subplot_layout=(2, 3)
# )
import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体为黑体（Windows 系统）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示异常（如果有负数坐标或标签）
plt.rcParams['axes.unicode_minus'] = False


def plot_metrics_and_loss(
        experiment_names,
        custom_legend_names,
        metrics_info,
        loss_info,
        metrics_subplot_layout,
        loss_subplot_layout,
        specific_colors=None,  # 新增参数：字典格式{图例名称: 颜色值}
        metrics_figure_size=(15, 10),
        loss_figure_size=(15, 10),
        base_directory='runs/train'
):
    # 获取Matplotlib默认颜色循环（确保其他曲线颜色与默认一致）
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 生成颜色映射：默认颜色循环 + 自定义颜色覆盖
    def get_color(legend_name, idx):
        if specific_colors and legend_name in specific_colors:
            return specific_colors[legend_name]  # 使用指定颜色
        return default_colors[idx % len(default_colors)]  # 使用默认循环颜色

    # Plot metrics
    plt.figure(figsize=metrics_figure_size)
    for i, (metric_name, title) in enumerate(metrics_info):
        plt.subplot(*metrics_subplot_layout, i + 1)
        for idx, (name, legend_name) in enumerate(zip(experiment_names, custom_legend_names)):
            file_path = os.path.join(base_directory, name, 'results.csv')
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == metric_name][0]
            # 根据图例名称获取颜色
            color = get_color(legend_name, idx)
            plt.plot(data[column_name], label=legend_name, color=color)
        plt.xlabel('Epoch')
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    metrics_filename = 'metrics_curves.png'
    plt.savefig(metrics_filename)
    plt.show()

    # Plot loss（逻辑与metrics一致）
    plt.figure(figsize=loss_figure_size)
    for i, (loss_name, title) in enumerate(loss_info):
        plt.subplot(*loss_subplot_layout, i + 1)
        for idx, (name, legend_name) in enumerate(zip(experiment_names, custom_legend_names)):
            file_path = os.path.join(base_directory, name, 'results.csv')
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == loss_name][0]
            color = get_color(legend_name, idx)
            plt.plot(data[column_name], label=legend_name, color=color)
        plt.xlabel('Epoch')
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    loss_filename = 'loss_curves.png'
    plt.savefig(loss_filename)
    plt.show()

    return metrics_filename, loss_filename


# Metrics to plot（保持不变）
metrics_info = [
    ('metrics/precision(B)', 'Precision'),
    ('metrics/recall(B)', 'Recall'),
    ('metrics/mAP50(B)', 'mAP at IoU=0.5'),
    ('metrics/mAP50-95(B)', 'mAP for IoU Range 0.5-0.95')
]

# Loss to plot（保持不变）
loss_info = [
    ('train/box_loss', 'Training Box Loss'),
    ('train/cls_loss', 'Training Classification Loss'),
    ('train/dfl_loss', 'Training DFL Loss'),
    ('val/box_loss', 'Validation Box Loss'),
    ('val/cls_loss', 'Validation Classification Loss'),
    ('val/dfl_loss', 'Validation DFL Loss')
]

# 自定义图例名称（保持不变）
custom_legend_names = [
    # 'YOLO11n', 'YOLOv10n', 'YOLOv8', 'YOLOv6',
    # 'YOLOv5', 'YOLOv3-tiny', 'YOLO11-ARE-PCCDSConv'
        'YOLO11n','YOLO11-DSConv', 'YOLO11-PCCDSConv','YOLO11-ARE', 'YOLO11-ARE-PCCDSConv'
]

# 关键设置：指定'YOLO11-ARE-PCCDSConv'为大红色（#FF0000是标准大红色）
specific_colors = {'YOLO11-ARE-PCCDSConv': '#FF0000'}

# 调用函数（新增specific_colors参数）
metrics_filename, loss_filename = plot_metrics_and_loss(
    # experiment_names=['exp', 'exp13', 'exp34', 'exp23', 'exp35', 'exp31', 'exp33'],
    experiment_names=['exp', 'exp7', 'exp8', 'exp26',  'exp33'],
    custom_legend_names=custom_legend_names,
    metrics_info=metrics_info,
    loss_info=loss_info,
    metrics_subplot_layout=(2, 2),
    loss_subplot_layout=(2, 3),
    specific_colors=specific_colors  # 传入颜色指定字典
)
