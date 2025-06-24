# -*- coding: utf-8 -*-
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
import time
import cv2

# 配置logger级别（可通过环境变量LOG_LEVEL设置）
LOG_LEVEL = os.getenv('LOG_LEVEL', 'ERROR').upper()
logger.remove()  # 移除默认的logger
logger.add(lambda msg: print(msg, end=""), level=LOG_LEVEL)

def parse_json_simple(content: str, content_type: str = "unknown") -> dict:
    """
    简单解析JSON字符串，带基本修复功能
    
    Args:
        content: 原始内容字符串
        content_type: 内容类型（用于日志）
        
    Returns:
        dict: 解析后的JSON对象，解析失败时返回None
    """
    if not content or not content.strip():
        logger.error(f"{content_type}内容为空")
        return None
    
    # 简单清理：移除首尾空白和markdown标记
    cleaned_content = content.strip()
    
    if cleaned_content.startswith('```json'):
        cleaned_content = cleaned_content[7:]
    elif cleaned_content.startswith('```'):
        cleaned_content = cleaned_content[3:]
    
    if cleaned_content.endswith('```'):
        cleaned_content = cleaned_content[:-3]
    
    cleaned_content = cleaned_content.strip()
    
    # 直接解析JSON
    try:
        parsed_json = json.loads(cleaned_content)
        logger.debug(f"{content_type}JSON解析成功")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.warning(f"{content_type}JSON直接解析失败: {str(e)}，尝试修复")
        
        # 尝试修复常见的JSON问题
        fixed_content = fix_json_quotes(cleaned_content)
        try:
            parsed_json = json.loads(fixed_content)
            logger.info(f"{content_type}JSON修复后解析成功")
            return parsed_json
        except json.JSONDecodeError as e2:
            logger.error(f"{content_type}JSON修复后仍解析失败: {str(e2)}")
            return None
    except Exception as e:
        logger.error(f"{content_type}JSON解析时发生未知错误: {str(e)}")
        return None

def fix_json_quotes(content: str) -> str:
    """
    修复JSON中的引号问题
    """
    import re
    
    # 修复策略：针对包含选择题选项的行进行特殊处理
    lines = content.split('\n')
    fixed_lines = []
    
    for line in lines:
        # 检查是否是包含选项的行（包含 A. B. C. D.）
        if '"question":' in line and ('A. "' in line or 'B. "' in line or 'C. "' in line or 'D. "' in line):
            # 对这种行进行特殊处理：将选项内的引号转义
            # 找到字段值的开始和结束
            start_quote = line.find(': "') + 3
            # 找到最后一个引号（字段值的结束）
            end_quote = line.rfind('"')
            
            if start_quote < end_quote:
                field_value = line[start_quote:end_quote]
                # 转义内部的引号
                fixed_value = field_value.replace('"', '\\"')
                fixed_line = line[:start_quote] + fixed_value + line[end_quote:]
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        elif '"answer":' in line and ('"A. "' in line or '"B. "' in line or '"C. "' in line or '"D. "' in line):
            # 同样处理answer字段中的引号问题
            start_quote = line.find(': "') + 3
            end_quote = line.rfind('"')
            
            if start_quote < end_quote:
                field_value = line[start_quote:end_quote]
                # 转义内部的引号，但保留首尾
                if field_value.startswith('"') and field_value.endswith('"'):
                    # 如果已经有引号包围，只转义内部的
                    inner_value = field_value[1:-1]
                    fixed_inner = inner_value.replace('"', '\\"')
                    fixed_value = f'"{fixed_inner}"'
                else:
                    fixed_value = field_value.replace('"', '\\"')
                fixed_line = line[:start_quote] + fixed_value + line[end_quote:]
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        else:
            # 对于其他行，进行通用的引号转义处理
            # 查找字段值中的未转义引号
            if '": "' in line and line.count('"') > 4:
                # 这表明可能有内部引号需要转义
                # 找到字段值部分
                colon_quote_pos = line.find('": "')
                if colon_quote_pos != -1:
                    start_pos = colon_quote_pos + 4
                    end_pos = line.rfind('"')
                    if start_pos < end_pos:
                        field_value = line[start_pos:end_pos]
                        # 转义内部引号
                        fixed_value = field_value.replace('"', '\\"')
                        fixed_line = line[:start_pos] + fixed_value + line[end_pos:]
                        fixed_lines.append(fixed_line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def get_video_info(video_path):
    """
    获取视频的基本信息：时长、大小、分辨率等
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        dict: 包含视频信息的字典
    """
    try:
        # 获取文件大小（字节）
        file_size = os.path.getsize(video_path)
        file_size_mb = round(file_size / (1024 * 1024), 2)  # 转换为MB
        
        # 使用OpenCV获取视频信息
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return None
            
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 宽度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高度
        
        # 计算时长（秒）
        duration_seconds = frame_count / fps if fps > 0 else 0
        duration_minutes = duration_seconds / 60
        
        cap.release()
        
        video_info = {
            "file_size_bytes": file_size,
            "file_size_mb": file_size_mb,
            "duration_seconds": round(duration_seconds, 2),
            "fps": round(fps, 2),
            "frame_count": int(frame_count),
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height
        }
        
        return video_info
        
    except Exception as e:
        logger.error(f"获取视频信息时出错: {str(e)}")
        return None

def call_openai(prompt, img_file, qa_type):
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复
    
    video_path = f"file://{img_file}"
    print(video_path)
    
    # 根据qa_type构建不同的提示语
    qa_prompts = {
        "caption": "请详尽分析并描述视频内容，需要包含以下方面：\n1. 整体概述：视频的主要内容和核心信息\n2. 视觉元素：场景、人物、物体、动作、色彩等\n3. 听觉元素：语音、音乐、音效、环境声等\n4. 文本内容：字幕、标题、显示的文字等\n5. 叙事结构：情节发展、转折点、高潮等\n6. 时空背景：拍摄时间、地点、环境等\n7. 主题分析：核心主题、寓意、情感表达等\n8. 因果关系：事件之间的逻辑联系\n9. 技术特点：拍摄手法、剪辑风格等\n请确保描述准确、全面且结构清晰。",
        "qa_event": "请分析视频中的事件，并生成一个问答对。问题应该关注视频中发生的具体事件和行为。",
        "qa_plot_predict": "请分析视频情节，预测可能的发展，并生成一个问答对。问题应该关注情节发展和预测。",
        "qa_emotion": "请分析视频中人物的情感表现，并生成一个问答对。问题应该关注人物的情感状态和表现。",
        "qa_intention": "请分析视频中人物的意图，并生成一个问答对。问题应该关注人物的动机和目的。",
        "qa_role": "请分析视频中人物的身份和性格特征，并生成一个问答对。问题应该关注人物的性格和角色定位。",
        "qa_casual": "请分析视频中的因果关系，并生成一个问答对。问题应该关注事件之间的因果联系。",
        "qa_intervention": "请分析如果采取某种干预措施会产生什么影响，并生成一个问答对。问题应该关注干预措施的效果。",
        "qa_conterfactual": "请分析如果改变某些条件会产生什么影响，进行反事实推演，并生成一个问答对。问题应该关注条件变化的影响。",
        "qa_scene": "请分析视频中的场景细节，并生成一个问答对。问题应该关注场景的多模态特征。"
    }
    if qa_type == "caption":
        current_prompt = qa_prompts.get(qa_type, "") + "\n" + f"""**输出要求：**
必须严格按照以下JSON格式输出，不要添加任何其他内容：

{{
    "caption": "详细的视频描述内容"
}}

**注意：**
- 输出必须是有效的JSON格式
- 只包含一个caption字段
- 所有内容使用中文
- 字符串用双引号包围
- 不要添加markdown标记或其他格式"""
    else:
        current_prompt = qa_prompts.get(qa_type, "") + "\n" + prompt
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": video_path
                },
                {"text": current_prompt}
            ]
        }
    ]

    try:
        # 调用主要回答API
        response = MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model="qwen-vl-max-latest",
            messages=messages,
            result_format='message',
            response_format={'type': 'json_object'}
        )

        clean_content = response["output"]["choices"][0]["message"].content[0]["text"]
        logger.info(f"模型输出: {clean_content}")
        
        # 解析主要回答的JSON
        answer = parse_json_simple(clean_content, "主要回答")
        if not answer:
            return "JSON解析失败", None
        
        # 生成思考过程（简化处理）
        thinking_content = ""
        think_prompt = f"""请根据以下问答对和视频内容，生成详细的思考分析过程：

问答对内容：
{clean_content}

**输出要求：**
必须严格按照以下JSON格式输出，不要添加任何其他内容：

{{
    "think": "完整的分析过程，包括规划、字幕分析、推理和总结"
}}

**分析内容应包含：**
1. <规划>视频整体结构和关键点分析</规划>
2. <字幕>按时间段分析视频内容，如：从00:00-00:02，视频展示了...</字幕>
3. <推理>基于视频内容的深入分析推理</推理>  
4. <总结>对视频内容的全面总结</总结>

**注意：**
- 输出必须是有效的JSON格式
- 所有分析内容放在think字段的字符串中
- 使用中文描述
- 时间段分析要准确连贯
- 字符串用双引号包围
- 不要添加markdown标记"""
        
        # 尝试生成思考过程，失败时使用空字符串
        try:
            response_thinking = MultiModalConversation.call(
                api_key=os.getenv('DASHSCOPE_API_KEY'),
                model="qwen-vl-max-latest",
                messages=[{
                    "role": "user",
                    "content": [{"video": video_path}, {"text": think_prompt}]
                }],
                result_format='message',
                response_format={'type': 'json_object'}
            )
            
            clean_content_think = response_thinking["output"]["choices"][0]["message"].content[0]["text"]
            logger.info(f"思考过程: {clean_content_think}")
            think_json = parse_json_simple(clean_content_think, "思考过程")
            if think_json and "think" in think_json:
                thinking_content = think_json.get("think", "")
        except:
            logger.warning("思考过程生成失败，使用空字符串")
        
        return answer, thinking_content

    except Exception as e:
        error_message = str(e)
        if "Input data may contain inappropriate content" in error_message:
            logger.warning(f"视频内容审核未通过，文件路径: {img_file}")
            return error_message, None
        else:
            logger.error(f"调用API失败: {error_message}，文件路径: {img_file}")
            return error_message, None

def process_single_video_qa(video_file: str, qa_types: list = None) -> dict:
    """
    处理单个视频的问答生成
    
    Args:
        video_file (str): 视频文件路径
        qa_types (list, optional): 问答类型列表。如果为None，则使用默认类型列表
        
    Returns:
        dict: 包含所有问答对的结果字典
    """
    # 记录开始时间
    start_time = time.time()
    
    # 如果没有提供qa_types，使用默认列表
    abs_video_path = video_file.absolute()
    # abs_video_path = video_file
    if qa_types is None:
        qa_types = [
            "caption",
            "qa_event", 
            "qa_plot_predict", 
            "qa_emotion", 
            "qa_intention", 
            "qa_role",
            "qa_casual",
            "qa_intervention",
            "qa_conterfactual",
            "qa_scene"
        ]
    
    # 获取视频信息
    video_info = get_video_info(abs_video_path)
    
    # 构建提示语
    prompt = """请根据视频内容生成问答对。

**要求：**
1. 深度分析视频的视觉和听觉信息
2. 结合字幕和音频内容
3. 提供4个选项（A/B/C/D），每个选项都要合理且有区分度
4. 答案需要深入分析视频内容才能得出

**输出要求：**
必须严格按照以下JSON格式输出，不要添加任何其他内容：

{
    "question": "基于视频中的具体事件和证据，[问题内容]？\\nA. [选项A]\\nB. [选项B]\\nC. [选项C]\\nD. [选项D]",
    "answer": "[选项答案]",
    "gt": "[A/B/C/D]"
}

**注意：**
- 输出必须是有效的JSON格式
- question字段包含完整问题和所有选项
- answer字段只包含选项内容，不要解释
- gt字段只包含一个字母（A、B、C或D）
- 不要使用换行符，使用\\n表示换行
- 所有字符串都要用双引号包围"""
    
    # 初始化结果字典
    results = {
        "video_path": str(video_file),
        "annotaion": {}
    }
    
    # 保存视频信息和处理信息用于统计（不包含在主结果中）
    results["_video_info"] = video_info
    results["_processing_info"] = {
        "start_timestamp": start_time
    }
    
    # 处理每种类型的问答
    with ThreadPoolExecutor(max_workers=1) as executor:
        # 创建future到qa_type的映射
        future_to_qa = {
            executor.submit(call_openai, prompt, abs_video_path, qa_type): qa_type
            for qa_type in qa_types
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(qa_types), desc="生成问答对") as pbar:
            for future in as_completed(future_to_qa):
                qa_type = future_to_qa[future]
                result, thinking = future.result()
                # 检查result是否为有效的字典结果
                if result and isinstance(result, dict):
                    results["annotaion"][qa_type] = result
                    results["annotaion"][qa_type]["thinking"] = thinking
                    print(f"{video_file}：{qa_type} 生成成功！")
                else:
                    print(f"{video_file}：{qa_type} 生成失败！原因: {result}")
                pbar.update(1)
    
    # 记录结束时间和计算处理时长
    end_time = time.time()
    processing_duration = end_time - start_time
    
    results["_processing_info"].update({
        "end_timestamp": end_time,
        "duration_seconds": round(processing_duration, 2)
    })
    
    return results

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv()
    
    # 设置视频文件路径
    video_file = r"E:\bilibili\0428\《千里江山图》vs《富春山居图》山水画也有代沟？.mp4"
    video_file = Path(video_file)
    # 
    # a = "E:\download\0428\2022-23赛季CHBL耐高全国总决赛：浙江回浦中学VS清华附中.mp4"
    results = process_single_video_qa(video_file)
    
    # 保存结果到文件
    output_file = "qa_results_test.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n所有问答对已生成完成，结果已保存到 {output_file}")