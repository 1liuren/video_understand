import os
import hashlib
import json
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import base64
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dashscope import MultiModalConversation
from llm_video_single import process_single_video_qa
import time

def process_video_folder(folder_path: str, max_workers: int = 4) -> list:
    """批量处理视频文件夹"""
    # 转换为Path对象并获取绝对路径
    # folder = Path(folder_path).absolute()
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"文件夹不存在: {folder}")
    
    # 获取所有视频文件
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(folder.glob(f'**/*{ext}'))
    
    if not video_files:
        raise ValueError(f"文件夹中没有找到视频文件: {folder}")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建future到文件名的映射
        future_to_file = {
            executor.submit(process_single_video_qa, video_file): video_file
            for video_file in video_files
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(video_files), desc="处理视频") as pbar:
            for future in as_completed(future_to_file):
                video_file = future_to_file[future]
                result = future.result()
                if result and isinstance(result, dict) and "annotaion" in result:
                    logger.info(f"处理视频 {video_file.name} 成功！")
                    results.append(result)
                else:
                    logger.error(f"处理视频 {video_file.name} 失败")
                pbar.update(1)
    
    return results

if __name__ == "__main__":
    load_dotenv()
    
    # 配置logger级别（可通过环境变量LOG_LEVEL设置）
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'ERROR').upper()
    logger.remove()  # 移除默认的logger
    logger.add(lambda msg: print(msg, end=""), level=LOG_LEVEL)
    
    # 设置输入文件夹路径（相对路径）
    input_folder = "gemini 预标注"
    
    # 确保result目录存在
    result_dir = Path("result")
    result_dir.mkdir(exist_ok=True)
    
    # 记录整体开始时间
    total_start_time = time.time()
    
    # 处理视频文件夹
    results = process_video_folder(input_folder, max_workers=4)
    
    # 记录整体结束时间
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # 保存结果到文件（移除统计信息，只保留纯净结果）
    clean_results = []
    for result in results:
        if result:
            clean_result = {
                "video_path": result["video_path"],
                "annotaion": result["annotaion"]
            }
            clean_results.append(clean_result)
    
    output_file = "result/qa_results_0619.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    # 收集统计信息
    stats = {
        "batch_processing_info": {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(total_start_time)),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(total_end_time)),
            "total_duration_seconds": round(total_duration, 2),
            "total_videos_processed": len(results),
            "input_folder": input_folder,
            "max_workers": 4
        },
        "video_statistics": []
    }
    
    # 收集每个视频的统计信息
    total_video_size_mb = 0
    total_video_duration_seconds = 0
    total_processing_duration_seconds = 0
    
    for result in results:
        if result and "_video_info" in result and result["_video_info"]:
            video_info = result["_video_info"]
            processing_info = result.get("_processing_info", {})
            
            video_stat = {
                "video_path": result["video_path"],
                "video_name": Path(result["video_path"]).name,
                "file_size_mb": video_info.get("file_size_mb", 0),
                "duration_seconds": video_info.get("duration_seconds", 0),
                "resolution": video_info.get("resolution", "unknown"),
                "fps": video_info.get("fps", 0),
                "processing_duration_seconds": processing_info.get("duration_seconds", 0)
            }
            
            stats["video_statistics"].append(video_stat)
            
            # 累加统计
            total_video_size_mb += video_info.get("file_size_mb", 0)
            total_video_duration_seconds += video_info.get("duration_seconds", 0)
            total_processing_duration_seconds += processing_info.get("duration_seconds", 0)
    
    # 添加汇总统计
    stats["summary"] = {
        "total_video_size_mb": round(total_video_size_mb, 2),
        "total_video_duration_seconds": round(total_video_duration_seconds, 2),
        "total_processing_duration_seconds": round(total_processing_duration_seconds, 2),
        "average_processing_time_per_video_seconds": round(total_processing_duration_seconds / len(results), 2) if results else 0,
        "processing_speed_ratio": round(total_video_duration_seconds / total_processing_duration_seconds, 2) if total_processing_duration_seconds > 0 else 0
    }
    
    # 保存统计信息到单独的文件
    stats_file = "result/processing_statistics_0619.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有问答对已生成完成，结果已保存到 {output_file}")
    print(f"处理统计信息已保存到 {stats_file}")
    print(f"\n=== 处理统计 ===")
    print(f"总计处理视频: {len(results)} 个")
    print(f"总视频大小: {stats['summary']['total_video_size_mb']} MB")
    print(f"总视频时长: {stats['summary']['total_video_duration_seconds']:.2f} 秒")
    print(f"总处理时长: {stats['summary']['total_processing_duration_seconds']:.2f} 秒")
    print(f"平均每个视频处理时间: {stats['summary']['average_processing_time_per_video_seconds']:.2f} 秒")
    print(f"处理速度比（视频时长/处理时长）: {stats['summary']['processing_speed_ratio']:.2f}")