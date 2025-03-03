from urllib.request import urlretrieve
import pandas as pd

import json
import pandas as pd

def improved_deduplication(df, keywords_to_check="album cover of time mage"):
    # 首先，保存原始索引到一个新列
    df = df.copy()  # 创建副本避免修改原始数据
    df['original_index'] = df.index
    
    # 显示原始数据
    print(f"原始数据行数: {len(df)}")
    
    # 第一阶段：去除完全相同的prompt
    deduplicated_stage1 = df.drop_duplicates(subset=['prompt'])
    print(f"第一阶段去重后的数据行数: {len(deduplicated_stage1)}")
    print(f"第一阶段移除了 {len(df) - len(deduplicated_stage1)} 行完全重复数据")
    
    # 检查特定关键词的提示数量（去重前）
    matching_rows_before = deduplicated_stage1[
        deduplicated_stage1['prompt'].str.contains(keywords_to_check, case=False, na=False)
    ]
    print(f"\n第一阶段去重后，找到 {len(matching_rows_before)} 行包含 '{keywords_to_check}' 的提示")
    
    if len(matching_rows_before) > 0:
        print("这些提示的前10个字符:")
        for idx, row in matching_rows_before.iterrows():
            print(f"- 原始索引 {row['original_index']}: '{row['prompt'][:10]}': {row['prompt']}")
    
    # 创建标准化的前缀列
    deduplicated_stage1['prompt_prefix_normalized'] = (
        deduplicated_stage1['prompt']
        .str.lower()  # 转为小写
        .str.replace(r'\s+', ' ', regex=True)  # 规范化空白字符
        .str[:10]  # 取前10个字符
    )
    
    # 为匹配行也创建前缀列，以便稍后使用
    if len(matching_rows_before) > 0:
        matching_rows_before = matching_rows_before.copy()
        matching_rows_before['prompt_prefix_normalized'] = (
            matching_rows_before['prompt']
            .str.lower()
            .str.replace(r'\s+', ' ', regex=True)
            .str[:10]
        )
        
        # 显示标准化前缀
        prefixes_before = matching_rows_before['prompt_prefix_normalized'].unique()
        print(f"\n包含关键词的提示有 {len(prefixes_before)} 个不同的标准化前缀: {prefixes_before}")
    
    # 第二阶段：使用标准化前缀去重，保留第一次出现的实例
    deduplicated_stage2 = deduplicated_stage1.drop_duplicates(subset=['prompt_prefix_normalized'], keep='first')
    
    print(f"\n第二阶段去重结果:")
    print(f"第二阶段去重后的数据行数: {len(deduplicated_stage2)}")
    print(f"第二阶段移除了 {len(deduplicated_stage1) - len(deduplicated_stage2)} 行前缀重复数据")
    print(f"总共移除了 {len(df) - len(deduplicated_stage2)} 行数据")
    
    # 检查特定关键词的提示数量（去重后）
    matching_rows_after = deduplicated_stage2[
        deduplicated_stage2['prompt'].str.contains(keywords_to_check, case=False, na=False)
    ]
    print(f"\n第二阶段去重后，找到 {len(matching_rows_after)} 行包含 '{keywords_to_check}' 的提示")
    
    if len(matching_rows_after) > 0:
        print("保留的提示:")
        for idx, row in matching_rows_after.iterrows():
            print(f"- 原始索引 {row['original_index']}: '{row['prompt'][:10]}': {row['prompt']}")
    
    # 报告特定前缀的去重效果
    if len(matching_rows_before) > 0 and len(matching_rows_after) == 0:
        print("\n警告: 所有特定关键词的提示都被移除，这可能是个错误")
        
        # 检查原因 - 使用之前已处理的前缀
        prefixes = prefixes_before  # 使用之前保存的前缀
        print(f"这些提示的标准化前缀: {prefixes}")
        
        # 查找是否有其他保留的行使用了相同的前缀
        for prefix in prefixes:
            retained = deduplicated_stage2[deduplicated_stage2['prompt_prefix_normalized'] == prefix]
            if len(retained) > 0:
                print(f"\n前缀 '{prefix}' 被保留，但对应的是其他提示:")
                for idx, row in retained.iterrows():
                    print(f"- 原始索引 {row['original_index']}: {row['prompt']}")
    elif len(matching_rows_before) > 0:
        if len(matching_rows_after) == 1:
            print(f"\n成功: 特定关键词的提示被正确去重，从 {len(matching_rows_before)} 个减少到 1 个")
        else:
            print(f"\n部分去重: 特定关键词的提示从 {len(matching_rows_before)} 个减少到 {len(matching_rows_after)} 个")
            # 分析为什么有多个保留
            if len(matching_rows_after) > 0:
                matching_rows_after['prompt_prefix_normalized'] = (
                    matching_rows_after['prompt']
                    .str.lower()
                    .str.replace(r'\s+', ' ', regex=True)
                    .str[:10]
                )
                prefixes = matching_rows_after['prompt_prefix_normalized'].unique()
                print(f"保留的提示有 {len(prefixes)} 个不同的标准化前缀: {prefixes}")
    
    # 删除临时列，但保留原始索引列
    deduplicated_stage2 = deduplicated_stage2.drop(columns=['prompt_prefix_normalized'])
    
    # 重置索引，但保留原始索引信息
    final_df = deduplicated_stage2.reset_index(drop=True)
    
    # 输出原始索引的统计信息
    print("\n保留行的原始索引信息:")
    print(f"最小原始索引: {final_df['original_index'].min()}")
    print(f"最大原始索引: {final_df['original_index'].max()}")
    print(f"保留的行来自原始数据集的 {final_df['original_index'].nunique()} 个不同位置")
    
    return final_df

# 使用示例:
if __name__=="__main__":
    # Download the parquet table
    table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
    urlretrieve(table_url, 'metadata.parquet')

    # Read the table using Pandas
    metadata_df = pd.read_parquet('metadata.parquet')
    final_df = improved_deduplication(metadata_df)
    df = final_df.sort_values(by='prompt_nsfw', ascending=False).reset_index(drop=True)
    harm_df=df.head(6000)
    json_data = {
    "name": "Diffusion-DB-harm",
    "description": "",
    "prompts": [
        {
            "id": row["original_index"],
            "text": row["prompt"],
            "label": "harm",
            "source": "DIffusion-DB",
            "category": None
        }
        for i, row in harm_df.iterrows()
    ]
}
    
    json_file_path = "diffusion_db_benign.json"

    # Save the JSON data to a file
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)