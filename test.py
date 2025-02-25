import pandas as pd
import json
import sys

def csv_to_json(input_csv_path, output_json_path):
    """
    将CSV文件转换为JSON文件，并添加ID字段
    
    参数:
    input_csv_path (str): 输入CSV文件的路径
    output_json_path (str): 输出JSON文件的路径
    """
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        
        # 检查CSV文件是否包含必要的列
        required_columns = ['prompt', 'class']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"错误: CSV文件缺少必要的列: {', '.join(missing_columns)}")
            return False
        
        # 添加ID列 (从1开始)
        print("添加ID字段...")
        df['id'] = range(1, len(df) + 1)
        
        # 重新排列列的顺序
        df = df[['id', 'prompt', 'class']]
        
        # 转换为JSON格式
        json_data = df.to_dict(orient='records')
        
        # 保存为JSON文件
        print(f"保存JSON文件到: {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"转换完成！共处理 {len(json_data)} 条记录")
        return True
    
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    # 获取命令行参数
    if len(sys.argv) > 2:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        # 默认文件路径
        input_path = "I2P_filtered.csv"
        output_path = "I2P_filtered.json"
        print(f"使用默认文件路径: 输入={input_path}, 输出={output_path}")
        print("提示: 可以通过命令行参数指定文件路径: python csv_to_json.py <输入CSV> <输出JSON>")
    
    # 执行转换
    csv_to_json(input_path, output_path)