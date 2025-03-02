import json
import os
import time
from openai import OpenAI
import pathlib

# 初始化OpenAI客户端
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def prepare_json_data(input_file_path):
    """
    读取原始JSON文件，并准备数据用于翻译
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理不同的JSON结构
        if 'prompts' in data:
            prompts_data = data.get('prompts', [])
        elif isinstance(data, list):
            # 如果是数组格式
            prompts_data = data
            data = {'prompts': prompts_data}  # 重构为统一格式
        else:
            # 假设数据在顶层
            prompts_data = [data]
            data = {'prompts': prompts_data}  # 重构为统一格式
        
        print(f"数据准备完成，共 {len(prompts_data)} 条记录")
        return data, prompts_data
    except Exception as e:
        print(f"读取文件 {input_file_path} 时出错: {e}")
        return None, None

def prepare_batch_file(prompts_data, output_jsonl_path):
    """
    准备批处理输入文件，将JSON数据转换成批处理API所需的JSONL格式
    """
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in prompts_data:
            # 确定要翻译的文本字段
            text_field = None
            for field in ['text', 'prompt', 'content', 'message']:
                if field in item and item[field]:
                    text_field = field
                    break
            
            if not text_field:
                print(f"警告: 无法在项目中找到文本字段: {item}")
                continue
            
            # 确保有ID，如果没有则创建一个
            item_id = item.get('id', hash(json.dumps(item, sort_keys=True)))
            
            # 为每个prompt创建一个请求
            batch_item = {
                "custom_id": f"id_{item_id}_{text_field}",  # 包含字段名以便后续识别
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
                            "content": item[text_field]
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

def process_results(results_file, original_data, prompts_data, output_path):
    """
    处理批处理结果，将翻译结果添加到JSON中并保存
    """
    # 创建映射字典以快速查找原始数据
    id_map = {}
    for i, item in enumerate(prompts_data):
        item_id = item.get('id', hash(json.dumps(item, sort_keys=True)))
        
        # 查找可能的文本字段
        for field in ['text', 'prompt', 'content', 'message']:
            if field in item and item[field]:
                id_map[f"id_{item_id}_{field}"] = (i, field)
    
    # 读取结果文件
    translations = {}
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']
            
            # 确保响应成功
            if result['response'] and result['response']['status_code'] == 200:
                translation = result['response']['body']['choices'][0]['message']['content']
                translations[custom_id] = translation
    
    # 为原始数据添加翻译字段
    for custom_id, translation in translations.items():
        if custom_id in id_map:
            index, field = id_map[custom_id]
            prompts_data[index]['translate'] = translation
    
    # 更新原始数据中的prompts
    if 'prompts' in original_data:
        original_data['prompts'] = prompts_data
    else:
        original_data = {'prompts': prompts_data}
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, ensure_ascii=False, indent=4)
    
    print(f"翻译结果已添加并保存到 {output_path}")
    print(f"共处理 {len(translations)} 个翻译，总数据量 {len(prompts_data)}")

def process_json_file(input_file_path, output_file_path):
    """
    处理指定的JSON文件
    """
    print(f"开始处理文件: {input_file_path}")
    
    # 创建临时目录
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. 准备数据
    original_data, prompts_data = prepare_json_data(input_file_path)
    
    if not original_data or not prompts_data:
        print(f"跳过文件 {input_file_path} 处理")
        return False
    
    # 2. 准备批处理文件
    batch_input_path = os.path.join(temp_dir, f"batch_input_{os.path.basename(input_file_path)}.jsonl")
    prepare_batch_file(prompts_data, batch_input_path)
    
    # 3. 启动批处理
    batch_id = run_batch_translation(batch_input_path)
    
    # 4. 等待完成
    batch = wait_for_batch_completion(batch_id)
    
    # 5. 下载结果
    results_file = download_batch_results(batch)
    
    if not results_file:
        print("未能获取批处理结果")
        return False
    
    # 6. 处理结果并保存
    process_results(results_file, original_data, prompts_data, output_file_path)
    
    # 7. 清理临时文件
    print("清理临时文件...")
    try:
        os.remove(batch_input_path)
        os.remove(results_file)
        if os.path.exists("batch_translation_errors.jsonl"):
            os.remove("batch_translation_errors.jsonl")
    except Exception as e:
        print(f"清理文件时出错: {e}")
    
    return True

def process_all_files(file_paths):
    """
    处理多个JSON文件
    """
    # 创建临时目录
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    results = {}
    
    for input_path in file_paths:
        # 生成输出路径
        output_dir = os.path.dirname(input_path)
        file_name = os.path.basename(input_path)
        name, ext = os.path.splitext(file_name)
        output_path = os.path.join(output_dir, f"{name}_translate{ext}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\n--- 处理文件: {input_path} ---")
        print(f"输出将保存到: {output_path}")
        
        success = process_json_file(input_path, output_path)
        results[input_path] = {
            'output_path': output_path,
            'success': success
        }
    
    # 清理临时目录
    if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) == 0:
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"清理临时目录时出错: {e}")
    
    # 打印处理结果摘要
    print("\n--- 处理结果摘要 ---")
    for input_path, result in results.items():
        status = "成功" if result['success'] else "失败"
        print(f"{input_path} -> {result['output_path']}: {status}")
    
    print("\n所有任务完成！")

def main():
    # 定义要处理的文件列表
    file_paths = [
        # "data/harmful/I2P/i2p.json",
        # "data/harmful/4chan/4chan.json",
        # "data/harmful/sneakyprompt/sneakyprompt.json",
        # "data/harmful/VBCDE/VBCDE.json",
        # "data/harmful/diffusion_db/diffusion_db_harm.json",
        # "data/benign/diffusion_db/diffusion_db_benign_6000.json"
        "data/harmful/civitai/civitai_nsfw_prompts_2000.json"
    ]
    
    # 处理所有文件
    process_all_files(file_paths)

if __name__ == "__main__":
    main()