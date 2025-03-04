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
    
    # 计算统计信息
    total_items = len(original_data['prompts']) if 'prompts' in original_data else 0
    labeled_count = 0
    
    # 更新原始数据的category字段
    if 'prompts' in original_data:
        for prompt in original_data['prompts']:
            prompt_id = prompt['id']
            if prompt_id in annotation_map:
                prompt['category'] = annotation_map[prompt_id]
                labeled_count += 1
    
    # 保存更新后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成! 已标注 {labeled_count}/{total_items} 个条目 ({labeled_count/total_items*100:.2f}%).")
    print(f"结果已保存到 {output_file}")
    
    return labeled_count, total_items

def process_multiple_files(file_pairs):
    """
    处理多个文件对
    
    Args:
        file_pairs: 包含原始文件、标注文件和输出文件路径的列表，每个元素是一个元组 (original_file, annotated_file, output_file)
    """
    total_labeled = 0
    total_items = 0
    
    for original_file, annotated_file, output_file in file_pairs:
        print(f"\n处理文件: {os.path.basename(original_file)} 和 {os.path.basename(annotated_file)}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            labeled, items = merge_annotations(original_file, annotated_file, output_file)
            total_labeled += labeled
            total_items += items
        except Exception as e:
            print(f"处理文件时出错: {e}")
    
    print(f"\n总计处理完成! 所有文件共标注 {total_labeled}/{total_items} 个条目 ({total_labeled/total_items*100:.2f}% 完成).")

# 使用示例
if __name__ == "__main__":
    # 定义文件对列表: (原始文件, 标注文件, 输出文件)
    file_pairs = [
        ("data/harmful/4chan/4chan_translate.json", "labeled_data/4chan.json", "data/harmful/4chan/4chan_translate_labeled.json"),
        ("data/harmful/VBCDE/VBCDE_translate.json", "labeled_data/VBCDE.json", "data/harmful/VBCDE/VBCDE_translate_labeled.json")
    ]
    
    # 处理所有文件对
    process_multiple_files(file_pairs)