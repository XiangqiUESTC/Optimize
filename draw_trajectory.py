import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.express as px
import os

i = 0
results_folder = "results"  # 请替换为实际的文件夹路径
task_name = "toy"  # 请替换为实际的任务名称
MAX_NORM = 0.032
MIN_NORM = MAX_NORM/32


# 加载数据
x = torch.load(f"{results_folder}\\{task_name}\\toy{i}.pt")
y = torch.load(f"{results_folder}\\{task_name}\\toy_all_task_grads{i}.pt")
z = torch.load(f"{results_folder}\\{task_name}\\toy_all_task_fixer_grads{i}.pt")
w = torch.load(f"{results_folder}\\{task_name}\\param_delta{i}.pt")  # 新增的实际更新梯度

methods = {
    # "sgd": (2575, 3200),
    # "cagrad": (2575, 3200),
    "pcgrad": (2575, 3200),
}

# 梯度缩放因子 - 可以根据需要调整这个值
GRADIENT_SCALE_FACTOR = 1

for method in methods:
    poses = x[method][methods[method][0]:methods[method][1]]
    task_grads = y[method][methods[method][0]:methods[method][1]]
    fixed_grads = z[method][methods[method][0]:methods[method][1]]
    actual_grads = w[method][methods[method][0]:methods[method][1]]  # 新增的实际更新梯度

    # 转换为numpy数组
    poses_np = poses.numpy()
    task_grads_np = task_grads.numpy()
    fixed_grads_np = fixed_grads.numpy()
    actual_grads_np = actual_grads.numpy()  # 新增的实际更新梯度

    # 获取步数
    n_steps = poses_np.shape[0]
    task_num = task_grads_np.shape[-1]

    print(f"Processing method: {method}, Steps: {n_steps}, Tasks: {task_num}")

    # 由于数据量可能很大，我们可以进行下采样以提高性能
    # 保持每250步一个点用于路径绘制，但保留所有数据用于梯度显示
    sample_rate = 1
    sampled_indices = np.arange(0, n_steps, sample_rate)

    # 创建颜色映射表示时间进展
    colors = np.arange(n_steps)

    # 创建子图
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"type": "scatter"}]]
    )

    # 添加优化路径（带颜色渐变，使用下采样数据）
    fig.add_trace(
        go.Scatter(
            x=poses_np[sampled_indices, 0],
            y=poses_np[sampled_indices, 1],
            mode='lines+markers',
            marker=dict(
                size=4,
                color=colors[sampled_indices],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="步数")
            ),
            line=dict(width=1, color='rgba(0,0,0,0.2)'),
            name='优化路径',
            hovertemplate=
            'x: %{x:.4f}<br>' +
            'y: %{y:.4f}<br>' +
            f'步数: {methods[method][0]}<extra></extra>'  # 修改为使用起始步数
        )
    )

    # 添加当前点（初始位置）
    current_point = go.Scatter(
        x=[poses_np[0, 0]],
        y=[poses_np[0, 1]],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='当前位置',
        hovertemplate=
        'x: %{x:.4f}<br>' +
        'y: %{y:.4f}<br>' +
        '步数: 0<extra></extra>'
    )
    fig.add_trace(current_point)


    # 创建梯度向量绘制函数
    def create_gradient_trace(x, y, dx, dy, name, color, step, scalar=1, is_actual=False):
        # 对于实际更新梯度，不需要反转方向（因为已经是实际更新方向）
        if not is_actual:
            # 反转梯度方向（因为参数更新是往梯度反方向去的）
            dx, dy = -dx, -dy

        # 计算向量长度用于调整箭头大小
        scale = scalar

        # 计算箭头的终点
        end_x = x + dx * scale
        end_y = y + dy * scale

        # 根据是否为实际更新梯度显示不同的信息
        hover_info = (
                f'{name}<br>' +
                f'梯度值: ({dx:.6f}, {dy:.6f})<br>' +
                f'步数: {step}<extra></extra>'
        )

        return go.Scatter(
            x=[x, end_x],
            y=[y, end_y],
            mode='lines+markers',
            line=dict(width=3, color=color),
            marker=dict(size=4, color=color),
            name=name,
            hovertemplate=hover_info
        )


    # 添加初始梯度向量
    for i in range(task_num):
        grad_trace = create_gradient_trace(
            poses_np[0, 0], poses_np[0, 1],
            task_grads_np[0, 0, i], task_grads_np[0, 1, i],
            f'任务{i + 1}梯度', "orange", 0
        )
        fig.add_trace(grad_trace)

    # 添加总梯度向量
    total_grad_trace = create_gradient_trace(
        poses_np[0, 0], poses_np[0, 1],
        fixed_grads_np[0, 0], fixed_grads_np[0, 1],
        '总梯度', 'red', 0
    )
    fig.add_trace(total_grad_trace)

    # 添加实际更新梯度向量
    actual_grad_trace = create_gradient_trace(
        poses_np[0, 0], poses_np[0, 1],
        actual_grads_np[0, 0], actual_grads_np[0, 1],
        '实际更新', 'purple', 0, is_actual=True
    )
    fig.add_trace(actual_grad_trace)

    # 创建动画帧
    frames = []
    slider_steps = []  # 用于存储滑块步骤

    # 由于数据量可能很大，我们可以对动画帧进行下采样
    frame_sample_rate = max(1, n_steps // 2500)  # 最多1000帧

    for step in range(0, n_steps, frame_sample_rate):
        frame_data = [
            # 优化路径（保持不变）
            go.Scatter(
                x=poses_np[sampled_indices, 0],
                y=poses_np[sampled_indices, 1],
                mode='lines+markers',
                marker=dict(
                    size=4,
                    color=colors[sampled_indices],
                    colorscale='Viridis'
                )
            ),
            # 当前点
            go.Scatter(
                x=[poses_np[step, 0]],
                y=[poses_np[step, 1]],
                mode='markers',
                marker=dict(size=12, color='red'),
                hovertemplate=f'x: %{{x:.4f}}<br>y: %{{y:.4f}}<br>步数: {step + methods[method][0]}<extra></extra>'
            )
        ]

        task_grad_norms = np.array([
            np.sqrt(task_grads_np[step, 0, i] ** 2 + task_grads_np[step, 1, i] ** 2)
            for i in range(task_grads_np.shape[-1])
        ])

        max_norm = task_grad_norms.max()

        tasks_scale = np.maximum(task_grad_norms/max_norm*MAX_NORM, MIN_NORM)

        tasks_scale = tasks_scale / (1e-8 + task_grad_norms)

        max_scale = tasks_scale[np.argmax(task_grad_norms)]

        # 添加任务梯度
        for i in range(task_num):
            frame_data.append(
                create_gradient_trace(
                    poses_np[step, 0], poses_np[step, 1],
                    task_grads_np[step, 0, i], task_grads_np[step, 1, i],
                    f'任务{i + 1}梯度', "brown", step + methods[method][0], scalar=tasks_scale[i]
                )
            )

        # 添加总梯度
        frame_data.append(
            create_gradient_trace(
                poses_np[step, 0], poses_np[step, 1],
                fixed_grads_np[step, 0], fixed_grads_np[step, 1],
                '总梯度', 'red', step + methods[method][0], scalar=max_scale
            )
        )

        actual_grads_norms = np.sqrt(actual_grads_np[step, 0] ** 2 + actual_grads_np[step, 1] ** 2)

        # 添加实际更新梯度
        frame_data.append(
            create_gradient_trace(
                poses_np[step, 0], poses_np[step, 1],
                actual_grads_np[step, 0], actual_grads_np[step, 1],
                '实际更新', 'green', step + methods[method][0], is_actual=True, scalar = 1
            )
        )

        frame = go.Frame(data=frame_data, name=str(step))
        frames.append(frame)

        # 添加滑块步骤
        slider_step = {
            "args": [
                [str(step)],
                {
                    "frame": {"duration": 0, "redraw": True},  # 设置为0，取消自动过渡
                    "mode": "immediate",
                    "transition": {"duration": 0}  # 设置为0，取消过渡效果
                }
            ],
            "label": str(step+methods[method][0]),
            "method": "animate"
        }
        slider_steps.append(slider_step)

    fig.frames = frames

    # 获取所有帧的名称列表
    frame_names = [frame.name for frame in frames]

    # 更新布局，包括滑块和按钮
    fig.update_layout(
        title=f'{method.upper()}方法 - 多任务优化',
        xaxis_title='x',
        yaxis_title='y',
        width=1000,
        height=700,
        hovermode='closest',
        updatemenus=[
            # 播放/暂停按钮
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        label="播放",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="暂停",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]),
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top"
            ),
        ],
        # 滑块
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "prefix": "步数: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0, "easing": "cubic-in-out"},  # 设置为0，取消过渡效果
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": slider_steps  # 使用预先构建的滑块步骤列表
        }]
    )

    # 保存为HTML文件
    fig.write_html(f"optimization_path_{method}_from_{methods[method][0]}_to_{methods[method][1]}.html")
    print(f"Saved visualization for {method} as optimization_path_{method}.html")

print("All visualizations completed!")