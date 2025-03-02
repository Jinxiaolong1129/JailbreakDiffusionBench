import json
import os

def merge_annotations(original_file, annotated_file, output_file):
    """
    将Label Studio标注结果合并到原始数据中
    
    Args:
        original_file: 原始数据文件路径
        annotated_file: Label Studio标注数据文件路径
        output_file: 输出文件路径
    """
    # 读取原始数据
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 读取标注数据
    with open(annotated_file, 'r', encoding='utf-8') as f:
        annotated_data = json.load(f)
    
    # 创建ID到标注结果的映射
    annotation_map = {}
    for item in annotated_data:
        original_id = item['data']['id']
        # 检查是否有标注结果
        if item['annotations'] and len(item['annotations']) > 0:
            # 获取第一个标注结果
            annotation = item['annotations'][0]
            if 'result' in annotation and len(annotation['result']) > 0:
                for result in annotation['result']:
                    if 'value' in result and 'choices' in result['value'] and len(result['value']['choices']) > 0:
                        # 获取所有标注选择作为列表
                        choices = result['value']['choices']
                        # 直接存储为列表
                        annotation_map[original_id] = choices
    
    # 更新原始数据的category字段
    for prompt in original_data['prompts']:
        prompt_id = prompt['id']
        if prompt_id in annotation_map:
            prompt['category'] = annotation_map[prompt_id]
    
    # 保存更新后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成! 已标注 {len(annotation_map)} 个条目.")
    print(f"结果已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    original_file = "data/harmful/diffusion_db/diffusion_db_harm_2000.json"
    annotated_file = "labeled_data/diffusion_db_harm.json"  # 标注后的数据文件
    output_file = "data/harmful/diffusion_db/diffusion_db_harm_2000_updated.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    merge_annotations(original_file, annotated_file, output_file)
    
    
    