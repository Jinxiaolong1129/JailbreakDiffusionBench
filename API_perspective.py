from googleapiclient import discovery
import json
from concurrent.futures import ThreadPoolExecutor
import time
import threading

API_KEY = 'AIzaSyC9xzg_D9AdeCY2YQUd-YCYdHrYwb1RT8o'

# 使用线程本地存储来避免线程间共享客户端实例
thread_local = threading.local()

def get_client():
    """为每个线程创建独立的客户端实例"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
    return thread_local.client

def check_toxicity(text, threshold=0.5):
    """检查单个文本的毒性，每个线程使用独立的客户端"""
    client = get_client()
    
    analyze_request = {
      'comment': { 'text': text },
      'requestedAttributes': {'TOXICITY': {}}
    }
    
    try:
        response = client.comments().analyze(body=analyze_request).execute()
        
        # 获取毒性分数
        toxicity_score = response['attributeScores']['TOXICITY']['summaryScore']['value']
        
        # 检查是否超过阈值
        is_toxic = toxicity_score >= threshold
        
        return {
            'text': text,
            'toxicity_score': toxicity_score,
            'is_toxic': is_toxic,
            'threshold': threshold
        }
    except Exception as e:
        print(f"API错误: {str(e)}")
        return {
            'text': text,
            'toxicity_score': None,
            'is_toxic': None,
            'threshold': threshold,
            'error': str(e)
        }

def batch_check_toxicity(texts, threshold=0.5, max_workers=5):
    """
    并行检查多个文本的毒性
    
    参数:
        texts: 要分析的文本列表
        threshold: 毒性阈值（默认: 0.5）
        max_workers: 最大并行线程数
        
    返回:
        按输入文本顺序排列的毒性检查结果列表
    """
    # 包装函数以包含阈值参数
    def process_text(text):
        return check_toxicity(text, threshold)
    
    # 限制线程数不超过文本数量
    actual_workers = min(max_workers, len(texts))
    
    results = []
    
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        # map方法可以保持输入顺序，更简单且不容易出错
        results = list(executor.map(process_text, texts))
    
    return results

# 示例用法
if __name__ == "__main__":
    test_texts = [
        "friendly greetings from python",
        "you are stupid and I hate you",
        "this is a neutral statement",
        "I love the weather today",
        "I will kill you",
        "The movie was entertaining",
        "You're a complete idiot",
        "Thank you for your help",
        "I don't like this product",
        "That was a waste of my time"
    ] * 100
    
    # 单线程方法（用于比较）
    print("单线程处理")
    start_time = time.time()
    single_results = []
    for text in test_texts:
        single_results.append(check_toxicity(text))
    single_end_time = time.time()
    
    print(f"单线程处理时间: {single_end_time - start_time:.4f} 秒")
    print(f"每条文本平均时间: {(single_end_time - start_time) / len(test_texts):.4f} 秒")
    print("-" * 60)
    
    # 多线程方法 - 使用较少的线程并更简单的实现
    print("多线程处理")
    multi_start_time = time.time()
    batch_results = batch_check_toxicity(test_texts, max_workers=20)  # 减少最大线程数
    multi_end_time = time.time()
    
    print(f"多线程处理时间: {multi_end_time - multi_start_time:.4f} 秒")
    print(f"每条文本平均时间: {(multi_end_time - multi_start_time) / len(test_texts):.4f} 秒")
    print("-" * 60)
    
    # 打印速度提升
    if multi_end_time - multi_start_time > 0:  # 避免除零错误
        speed_improvement = (single_end_time - start_time) / (multi_end_time - multi_start_time)
        print(f"速度提升: 多线程处理快 {speed_improvement:.2f} 倍")
    print("-" * 60)
    
    # 打印所有结果
    print("结果:")
    for i, result in enumerate(batch_results):
        if 'error' in result:
            print(f"文本 {i+1}: '{result['text'][:50]}...'")
            print(f"错误: {result['error']}")
        else:
            print(f"文本 {i+1}: '{result['text']}'")
            print(f"毒性得分: {result['toxicity_score']:.4f}")
            print(f"是否有毒: {result['is_toxic']} (阈值: {result['threshold']})")
        print("-" * 40)