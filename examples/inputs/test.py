import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --- 1. 数据与颜色配置 ---
colors = {
    'AutoFigure': '#f0d9e2', 'GPT-Image': '#1d3d25', 'SVG Code': '#3a633d',
    'HTML Code': '#c2c9b4', 'PPTX Code': '#adc2d1', 'Diagram Agent': '#82a6c2',
    'Reference': '#eeeeee', 'None': '#cad9e0'
}

# (a) 数据
win_labels = ['Reference', 'AutoFigure', 'GPT-Image', 'SVG Code', 'HTML Code', 'PPTX Code', 'Diagram Agent']
win_vals = [96.8, 83.3, 55.6, 46.8, 48.4, 7.9, 11.1]

# (b) 数据
pub_labels = ['AutoFigure', 'GPT-Image', 'SVG Code', 'HTML Code', 'None']
pub_vals = [66.7, 4.8, 4.8, 4.8, 28.6]

# (c, d, e) 数据 (均值)
score_labels = ['AutoFigure', 'GPT-Image', 'SVG Code', 'HTML Code', 'PPTX Code', 'Diagram Agent']
acc_scores = [4.00, 2.05, 2.52, 2.52, 1.19, 1.19]
cla_scores = [4.14, 2.57, 2.10, 2.62, 1.05, 1.29]
aes_scores = [4.24, 2.33, 1.81, 1.95, 1.00, 1.00]
std_err = 0.12 # 误差线粗细

# --- 2. 绘图核心逻辑 ---
fig = plt.figure(figsize=(15, 9), dpi=100)
gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.5)

def style_ax(ax, title, ylabel, ylim):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=14, pad=20)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([])
    ax.grid(axis='y', linestyle='--', alpha=0.3)

def add_labels_and_logo(ax, rects, is_score=False):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        label = f"{height:.1f}" + ("" if is_score else "%")
        ax.text(rect.get_x() + rect.get_width()/2., height + (0.1 if is_score else 2),
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 模拟 AutoFigure Logo (如果当前柱子是第一个且是 AutoFigure)
        if i == (1 if ax.get_title().startswith("(a)") else 0):
            ax.text(rect.get_x() + rect.get_width()/2., height + (1.2 if is_score else 15), 
                    'Auto\nFigure', ha='center', va='bottom', color='#e91e63', 
                    fontweight='black', fontsize=10, linespacing=0.8)

# (a) Overall Win Rate
ax0 = fig.add_subplot(gs[0, :3])
bars0 = ax0.bar(win_labels, win_vals, color=[colors[l] for l in win_labels])
bars0[0].set_hatch('////')
bars0[0].set_edgecolor('#bbbbbb')
style_ax(ax0, "(a) Overall Win Rate Analysis", "Win Rate (%)", 110)
add_labels_and_logo(ax0, bars0)
ax0.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# (b) Publication Intent
ax1 = fig.add_subplot(gs[0, 3:])
bars1 = ax1.bar(pub_labels, pub_vals, color=[colors[l] for l in pub_labels])
bars1[-1].set_hatch('\\\\\\\\')
bars1[-1].set_edgecolor('#adc2d1')
style_ax(ax1, "(b) Publication Intent Distribution", "Selection Rate (%)", 110)
add_labels_and_logo(ax1, bars1)
ax1.axvline(3.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# (c, d, e) Scores
for i, (data, title, pos) in enumerate(zip([acc_scores, cla_scores, aes_scores], 
                                          ["(c) Accuracy Performance", "(d) Clarity Performance", "(e) Aesthetics Performance"],
                                          [gs[1, :2], gs[1, 2:4], gs[1, 4:]])):
    ax = fig.add_subplot(pos)
    bars = ax.bar(score_labels, data, color=[colors[l] for l in score_labels], yerr=std_err, capsize=3)
    style_ax(ax, title + " Score", "Score (1-5)", 6)
    add_labels_and_logo(ax, bars, is_score=True)

# --- 3. 图例与保存 ---
legend_elements = [
    Patch(facecolor=colors['AutoFigure'], label='AutoFigure'),
    Patch(facecolor=colors['GPT-Image'], label='GPT-Image'),
    Patch(facecolor=colors['SVG Code'], label='SVG Code'),
    Patch(facecolor=colors['HTML Code'], label='HTML Code'),
    Patch(facecolor=colors['PPTX Code'], label='PPTX Code'),
    Patch(facecolor=colors['Diagram Agent'], label='Diagram Agent'),
    Patch(facecolor=colors['Reference'], hatch='////', edgecolor='#bbbbbb', label='Reference'),
    Patch(facecolor=colors['None'], hatch='\\\\\\\\', edgecolor='#adc2d1', label='None'),
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=True, 
           bbox_to_anchor=(0.5, 0.02), fontsize=12)

plt.tight_layout(rect=[0, 0.08, 1, 0.95])

# 保存文件
save_path = "test.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"图表已成功保存至: {save_path}")

plt.show()