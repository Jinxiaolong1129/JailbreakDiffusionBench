import pandas as pd
import json
import os
import time
from openai import OpenAI
import csv
import pathlib

# 初始化OpenAI客户端
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 这个函数在新的process_all_files函数中被替代了，
# 但为了保持代码完整性，仍然保留它
def prepare_data(input_file_path):
    """
    读取原始CSV文件，添加ID，并转换为JSON格式
    """
    # 读取原始CSV文件
    df = pd.read_csv(input_file_path)
    
    # 添加ID列 (从1开始)
    df['id'] = range(1, len(df) + 1)
    
    # 确保存在prompt列
    if 'prompt' not in df.columns:
        # 假设第一列是prompt内容
        df.rename(columns={df.columns[0]: 'prompt'}, inplace=True)
    
    # 确保有class列，如果没有，添加一个默认值
    if 'class' not in df.columns:
        df['class'] = 'unknown'
    
    # 重新排列列的顺序，把id放在最前面
    columns = ['id']
    for col in df.columns:
        if col not in ['id']:
            columns.append(col)
    df = df[columns]
    
    # 转换为JSON格式
    json_data = df.to_dict(orient='records')
    
    print(f"数据准备完成，共 {len(json_data)} 条记录")
    return json_data, df.columns.tolist()

def prepare_batch_file(json_data, output_jsonl_path):
    """
    准备批处理输入文件，将JSON数据转换成批处理API所需的JSONL格式
    """
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in json_data:
            # 为每个prompt创建一个请求
            batch_item = {
                "custom_id": f"id_{item['id']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个专业的翻译助手。请将用户输入的文本翻译成中文，保持原意并使用自然的表达。"
                        },
                        {
                            "role": "user",
                            "content": item['prompt']
                        }
                    ],
                    "max_tokens": 1000
                }
            }
            f.write(json.dumps(batch_item, ensure_ascii=False) + '\n')
    
    print(f"已成功创建批处理输入文件: {output_jsonl_path}")
    return output_jsonl_path

def run_batch_translation(jsonl_file_path):
    """
    上传JSONL文件并启动批处理任务
    """
    # 1. 上传JSONL文件
    with open(jsonl_file_path, 'rb') as f:
        file = client.files.create(
            file=f,
            purpose="batch"
        )
    
    file_id = file.id
    print(f"文件已上传，ID: {file_id}")
    
    # 2. 创建批处理任务
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    batch_id = batch.id
    print(f"批处理任务已创建，ID: {batch_id}")
    
    return batch_id

def check_batch_status(batch_id):
    """
    检查批处理任务的状态
    """
    batch = client.batches.retrieve(batch_id)
    return batch

def wait_for_batch_completion(batch_id, check_interval=300):
    """
    等待批处理任务完成，每隔check_interval秒检查一次
    """
    while True:
        batch = check_batch_status(batch_id)
        status = batch.status
        
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        
        print(f"当前状态: {status}, 已完成: {completed}/{total}")
        
        if status == "completed":
            print("批处理任务已完成!")
            return batch
        elif status in ["failed", "expired", "cancelled"]:
            print(f"批处理任务未成功完成，状态为: {status}")
            return batch
        
        print(f"等待 {check_interval} 秒后再次检查...")
        time.sleep(check_interval)

def download_batch_results(batch):
    """
    下载批处理结果并保存到本地
    """
    if not batch.output_file_id:
        print("没有可用的输出文件")
        return None
    
    # 下载输出文件
    output_file = client.files.content(batch.output_file_id)
    output_content = output_file.text
    
    # 保存到本地
    output_path = "batch_translation_results.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)
    
    print(f"结果已保存到 {output_path}")
    
    # 如果有错误文件，也下载
    if batch.error_file_id:
        error_file = client.files.content(batch.error_file_id)
        error_content = error_file.text
        
        error_path = "batch_translation_errors.jsonl"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(error_content)
        
        print(f"错误信息已保存到 {error_path}")
    
    return output_path

def process_results(results_file, json_data, original_columns, output_path):
    """
    处理批处理结果，将翻译结果合并到最终CSV
    """
    # 创建ID到原始数据的映射
    id_to_data = {item['id']: item for item in json_data}
    
    # 读取结果文件
    translations = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            # 从custom_id中提取原始ID
            original_id = int(result['custom_id'].replace('id_', ''))
            
            # 确保响应成功
            if result['response'] and result['response']['status_code'] == 200:
                translation = result['response']['body']['choices'][0]['message']['content']
                translations[original_id] = translation
    
    # 创建最终数据
    final_data = []
    for item_id, item in id_to_data.items():
        final_item = {}
        
        # 填充原始列，除了id列
        for col in original_columns:
            if col != 'id':
                final_item[col] = item[col]
        
        # 添加翻译列
        final_item['translate_prompt'] = translations.get(item_id, '翻译失败')
        
        final_data.append(final_item)
    
    # 保存为CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(final_data[0].keys()))
        writer.writeheader()
        writer.writerows(final_data)
    
    print(f"翻译结果已合并并保存到 {output_path}")
    print(f"共处理 {len(translations)} 个翻译，总数据量 {len(final_data)}")

def process_all_files(directory_path, pattern="*.csv"):
    """
    处理指定目录下所有匹配模式的CSV文件
    """
    import glob
    
    # 获取所有匹配的文件
    file_paths = glob.glob(os.path.join(directory_path, pattern))
    
    if not file_paths:
        print(f"在 {directory_path} 中未找到匹配 {pattern} 的文件")
        return
    
    print(f"找到 {len(file_paths)} 个文件需要处理")
    
    # 创建临时目录
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 存储所有文件的数据，以及文件路径和列信息的映射
    all_json_data = []
    file_info = {}  # 存储每个文件的信息
    current_id = 1  # 全局ID计数器
    
    # 1. 准备所有文件的数据
    for file_path in file_paths:
        print(f"正在处理文件: {file_path}")
        
        # 读取并处理每个文件
        df = pd.read_csv(file_path)
        
        # 确保存在prompt列
        if 'prompt' not in df.columns:
            # 假设第一列是prompt内容
            df.rename(columns={df.columns[0]: 'prompt'}, inplace=True)
        
        # 确保有class列，如果没有，添加一个默认值
        if 'class' not in df.columns:
            df['class'] = 'unknown'
        
        # 为这个文件的每条记录分配全局ID
        df['global_id'] = range(current_id, current_id + len(df))
        current_id += len(df)
        
        # 保存原始列信息
        original_columns = df.columns.tolist()
        
        # 重新排列列的顺序，把global_id放在最前面
        columns = ['global_id']
        for col in df.columns:
            if col not in ['global_id']:
                columns.append(col)
        df = df[columns]
        
        # 转换为JSON格式并添加到全局列表
        json_data = df.to_dict(orient='records')
        
        # 存储文件信息
        file_info[file_path] = {
            'original_columns': original_columns,
            'start_id': df['global_id'].min(),
            'end_id': df['global_id'].max(),
            'data': json_data
        }
        
        all_json_data.extend(json_data)
        
        print(f"文件 {file_path} 处理完成，含 {len(json_data)} 条记录")
    
    print(f"所有文件处理完成，共 {len(all_json_data)} 条记录")
    
    # 2. 准备批处理文件
    batch_input_path = os.path.join(temp_dir, "all_batch_input.jsonl")
    
    with open(batch_input_path, 'w', encoding='utf-8') as f:
        for item in all_json_data:
            # 为每个prompt创建一个请求
            batch_item = {
                "custom_id": f"id_{item['global_id']}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个专业的翻译助手。请将用户输入的文本翻译成中文，保持原意并使用自然的表达。"
                        },
                        {
                            "role": "user",
                            "content": item['prompt']
                        }
                    ],
                    "max_tokens": 1000
                }
            }
            f.write(json.dumps(batch_item, ensure_ascii=False) + '\n')
    
    print(f"已成功创建批处理输入文件: {batch_input_path}")
    
    # 3. 启动批处理
    batch_id = run_batch_translation(batch_input_path)
    
    # 4. 等待完成
    batch = wait_for_batch_completion(batch_id)
    
    # 5. 下载结果
    results_file = download_batch_results(batch)
    
    if not results_file:
        print("未能获取批处理结果")
        return
    
    # 6. 读取并处理翻译结果
    translations = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            # 从custom_id中提取原始ID
            original_id = int(result['custom_id'].replace('id_', ''))
            
            # 确保响应成功
            if result['response'] and result['response']['status_code'] == 200:
                translation = result['response']['body']['choices'][0]['message']['content']
                translations[original_id] = translation
    
    print(f"成功获取 {len(translations)} 条翻译结果")
    
    # 7. 将翻译结果分配回各个文件并保存
    for file_path, info in file_info.items():
        start_id = info['start_id']
        end_id = info['end_id']
        original_columns = info['original_columns']
        file_data = info['data']
        
        # 创建最终数据
        final_data = []
        for item in file_data:
            global_id = item['global_id']
            final_item = {}
            
            # 填充原始列，除了global_id列
            for col in original_columns:
                if col != 'global_id':
                    final_item[col] = item[col]
            
            # 添加翻译列
            final_item['translate_prompt'] = translations.get(global_id, '翻译失败')
            
            final_data.append(final_item)
        
        # 保存为CSV
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(final_data[0].keys()))
            writer.writeheader()
            writer.writerows(final_data)
        
        print(f"文件 {file_path} 翻译结果已保存，包含 {len(final_data)} 条记录")
    
    # 8. 清理临时文件
    print("清理临时文件...")
    for file in [batch_input_path, results_file, "batch_translation_errors.jsonl"]:
        if os.path.exists(file):
            os.remove(file)
    
    if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) == 0:
        os.rmdir(temp_dir)
        
    print("所有任务完成！")

def main():
    # 处理指定目录下的CSV文件
    directory_path = "data/VBCDE"
    # directory_path = "data/sneakyprompt"
    
    # 可以指定具体的文件模式，例如"04-*.csv"仅处理以"04-"开头的CSV文件
    # 或者使用"*.csv"处理所有CSV文件
    file_pattern = "*.csv"
    
    # 调用处理函数
    process_all_files(directory_path, file_pattern)

if __name__ == "__main__":
    main()