import os
import json

file_paths = [
    # "data/harmful/I2P/i2p_translate.json",
    # "data/harmful/VBCDE/VBCDE_translate.json"
    # "data/harmful/4chan/4chan_translate.json"
    # "data/harmful/civitai/civitai_nsfw_prompts_2000_translate.json"
    # "data/harmful/diffusion_db/diffusion_db_harm_translate.json"
    # "data/harmful/sneakyprompt/sneakyprompt_translate.json"
    "data/harmful/diffusion_db/diffusion_db_harm_translate_2000.json"
]

# 输出目录
output_dir = "label_studio/"
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在，跳过: {file_path}")
        continue

    # 读取原始 JSON 数据
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 检查 JSON 结构
    if "prompts" not in raw_data:
        print(f"❌ 无 'prompts' 键，跳过: {file_path}")
        continue

    # 转换数据格式
    converted_data = []
    for item in raw_data["prompts"]:
        converted_data.append({
            "data": {
                "id": item.get("id"),
                "text": item.get("text"),
                "translated_prompt": item.get("translate"),
                "orignal_class": item.get("label"),
                "source": item.get("source"),
                "category": item.get("category"),
            }
        })

    # 生成输出文件路径
    file_name = os.path.basename(file_path)  # 仅获取文件名
    output_path = os.path.join(output_dir, file_name)

    # 保存转换后的 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 转换完成: {output_path}")
