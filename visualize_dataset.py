#!/usr/bin/env python3
"""
可视化 jailbreak_diffusion_bench_filtered_400.json 数据集
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from collections import Counter
import numpy as np

def visualize_dataset(file_path):
    """生成数据集可视化图表"""
    
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = data['prompts']
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 来源分布饼图
    ax1 = plt.subplot(2, 3, 1)
    source_counter = Counter(p['source'] for p in prompts)
    sources = list(source_counter.keys())
    counts = list(source_counter.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(sources)))
    ax1.pie(counts, labels=sources, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Data Source Distribution', fontsize=12, fontweight='bold')
    
    # 2. 类别分布柱状图
    ax2 = plt.subplot(2, 3, 2)
    category_counter = Counter()
    for p in prompts:
        for cat in p['category']:
            category_counter[cat] += 1
    
    categories = list(category_counter.keys())
    cat_counts = list(category_counter.values())
    y_pos = np.arange(len(categories))
    ax2.barh(y_pos, cat_counts, color='steelblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([cat[:30] + '...' if len(cat) > 30 else cat for cat in categories], fontsize=8)
    ax2.set_xlabel('Count')
    ax2.set_title('Category Distribution', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    # 3. 文本长度分布直方图
    ax3 = plt.subplot(2, 3, 3)
    text_lengths = [len(p['text']) for p in prompts]
    ax3.hist(text_lengths, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Text Length (characters)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Text Length Distribution', fontsize=12, fontweight='bold')
    ax3.axvline(np.mean(text_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(text_lengths):.1f}')
    ax3.axvline(np.median(text_lengths), color='green', linestyle='--', label=f'Median: {np.median(text_lengths):.1f}')
    ax3.legend()
    
    # 4. 翻译状态饼图
    ax4 = plt.subplot(2, 3, 4)
    translated_count = sum(1 for p in prompts if p.get('translate', '').strip())
    not_translated_count = len(prompts) - translated_count
    ax4.pie([translated_count, not_translated_count], 
            labels=['Translated', 'Not Translated'], 
            autopct='%1.1f%%', 
            colors=['lightgreen', 'lightcoral'],
            startangle=90)
    ax4.set_title('Translation Status', fontsize=12, fontweight='bold')
    
    # 5. 各来源的类别分布堆叠柱状图
    ax5 = plt.subplot(2, 3, 5)
    source_categories = {}
    for p in prompts:
        source = p['source']
        if source not in source_categories:
            source_categories[source] = Counter()
        for cat in p['category']:
            source_categories[source][cat] += 1
    
    # 获取所有类别
    all_categories = sorted(set(cat for p in prompts for cat in p['category']))
    sources = sorted(source_categories.keys())
    
    # 准备数据
    bottom = np.zeros(len(sources))
    colors_map = plt.cm.tab10(np.linspace(0, 1, len(all_categories)))
    
    for i, cat in enumerate(all_categories):
        values = [source_categories[src][cat] for src in sources]
        ax5.bar(sources, values, bottom=bottom, label=cat[:20], color=colors_map[i])
        bottom += values
    
    ax5.set_ylabel('Count')
    ax5.set_title('Category Distribution by Source', fontsize=12, fontweight='bold')
    ax5.set_xticklabels(sources, rotation=45, ha='right', fontsize=8)
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    
    # 6. 文本长度箱线图（按来源）
    ax6 = plt.subplot(2, 3, 6)
    source_lengths = {}
    for p in prompts:
        source = p['source']
        if source not in source_lengths:
            source_lengths[source] = []
        source_lengths[source].append(len(p['text']))
    
    sources = sorted(source_lengths.keys())
    data_to_plot = [source_lengths[src] for src in sources]
    bp = ax6.boxplot(data_to_plot, labels=sources, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax6.set_ylabel('Text Length (characters)')
    ax6.set_title('Text Length by Source', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis_visualization.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存为: dataset_analysis_visualization.png")
    
    # 生成额外的统计表
    print("\n" + "="*80)
    print("各来源的文本长度统计")
    print("="*80)
    for source in sorted(source_lengths.keys()):
        lengths = source_lengths[source]
        print(f"\n{source}:")
        print(f"  数量: {len(lengths)}")
        print(f"  平均长度: {np.mean(lengths):.1f} 字符")
        print(f"  中位数: {np.median(lengths):.1f} 字符")
        print(f"  最短: {min(lengths)} 字符")
        print(f"  最长: {max(lengths)} 字符")

if __name__ == "__main__":
    file_path = "data/jailbreak_diffusion_bench/jailbreak_diffusion_bench_filtered_400.json"
    visualize_dataset(file_path)

