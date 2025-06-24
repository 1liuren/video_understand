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

def load_config(config_path: str = "config.json") -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置信息
    """
    try:
        if not os.path.exists(config_path):
            logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return get_default_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证配置完整性
        required_sections = ['api_config', 'model_config', 'prompt_config', 'output_config']
        for section in required_sections:
            if section not in config:
                logger.warning(f"配置文件缺少 {section} 部分，使用默认值")
                config[section] = get_default_config()[section]
        
        logger.info(f"成功加载配置文件: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return get_default_config()

def get_default_config() -> dict:
    """
    获取默认配置
    
    Returns:
        dict: 默认配置信息
    """
    return {
        "api_config": {
            "api_key": os.getenv('DASHSCOPE_API_KEY', 'YOUR_DASHSCOPE_API_KEY_HERE'),
            "base_url": os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/api/v1'),
            "timeout": 60
        },
        "model_config": {
            "model_name": "qwen-vl-max",
            "temperature": 0.1,
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0,
            "presence_penalty": 0
        },
        "prompt_config": {
            "system_prompt": "你是一个专业的视频内容分析助手。请根据提供的视频内容，生成准确的问答对。",
            "max_retry": 3,
            "retry_delay": 1
        },
        "output_config": {
            "output_dir": "result",
            "log_level": "INFO",
            "save_intermediate": False
        }
    }

# 加载全局配置
CONFIG = load_config()

# 配置logger级别（优先使用配置文件，然后是环境变量）
LOG_LEVEL = CONFIG['output_config'].get('log_level', os.getenv('LOG_LEVEL', 'ERROR')).upper()
logger.remove()  # 移除默认的logger
logger.add(lambda msg: print(msg, end=""), level=LOG_LEVEL)

def parse_json_simple(content: str, content_type: str = "unknown") -> dict:
    """
    简化的JSON解析函数，包含两层修复机制
    
    Args:
        content: 原始内容字符串
        content_type: 内容类型（用于日志）
        
    Returns:
        dict: 解析后的JSON对象，解析失败时返回None
    """
    if not content or not content.strip():
        logger.error(f"{content_type}内容为空")
        return None
    
    # 清理markdown标记
    cleaned_content = content.strip()
    if cleaned_content.startswith('```json'):
        cleaned_content = cleaned_content[7:]
    elif cleaned_content.startswith('```'):
        cleaned_content = cleaned_content[3:]
    if cleaned_content.endswith('```'):
        cleaned_content = cleaned_content[:-3]
    cleaned_content = cleaned_content.strip()
    
    # 第一次尝试：直接解析
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        logger.warning(f"{content_type}JSON直接解析失败: {str(e)}，尝试修复")
    
    
    try:
        fixed_content2 = fix_missing_quotes(cleaned_content)
        result = json.loads(fixed_content2)
        logger.info(f"{content_type}JSON修复后解析成功")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"{content_type}JSON所有修复尝试失败: {str(e)}")
        return None


def fix_missing_quotes(content: str) -> str:
    """
    修复JSON中缺少引号、引号多余，以及换行符未转义的问题
    """
    import re

    fixed = content

    # 1. 转义换行符，只对 "question": "..." 的值做处理
    def escape_newlines_in_question(match):
        key = match.group(1)
        value = match.group(2)
        # 替换换行符为 \n，同时移除前后空格
        escaped_value = value.replace('\n', '\\n').strip()
        return f'"{key}": "{escaped_value}"'

    fixed = re.sub(r'"(question)":\s*"((?:[^"\\]|\\.)*)"', escape_newlines_in_question, fixed, flags=re.DOTALL)

    # 2. 修复 "answer": 后面缺少引号的情况
    fixed = re.sub(r'"answer":\s*([^"\s][^,}\n]*)([,}])', r'"answer": "\1"\2', fixed)

    # 3. 修复多余引号的情况（如多个双引号）
    fixed = re.sub(r'"answer":\s*"([^"]*?)"+([,}])', r'"answer": "\1"\2', fixed)

    # 4. 修复 "gt" 字段缺少引号
    fixed = re.sub(r'"gt":\s*([ABCD])([,}])', r'"gt": "\1"\2', fixed)

    return fixed

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
    
    # 从配置获取系统提示词（如果需要的话）
    prompt_config = CONFIG['prompt_config']
    
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
        # 从配置获取API参数
        api_config = CONFIG['api_config']
        model_config = CONFIG['model_config']
        
        # 调用主要回答API
        response = MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model=model_config.get('model_name', "qwen-vl-max"),
            messages=messages,
            temperature=model_config.get('temperature', 0.1),
            top_p=model_config.get('top_p', 0.2),
            result_format='message',
            response_format={'type': 'json_object'}
        )
        logger.info(f"response: {response}")
        clean_content = response.output.choices[0].message.content[0]["text"]
        logger.info(f"模型{qa_type}输出: {clean_content}")
        
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
                model=model_config.get('model_name', "qwen-vl-max"),
                messages=[{
                    "role": "user",
                    "content": [{"video": video_path}, {"text": think_prompt}]
                }],
                temperature=model_config.get('temperature', 0.1),
                top_p=model_config.get('top_p', 0.2),
                result_format='message',
                response_format={'type': 'json_object'}
            )
            logger.info(f"response_thinking: {response_thinking}")
            clean_content_think = response_thinking.output.choices[0].message.content[0]["text"]  
            logger.info(f"模型{qa_type}思考过程: {clean_content_think}")
            think_json = parse_json_simple(clean_content_think, "思考过程")
            if think_json and "think" in think_json:
                thinking_content = think_json.get("think", "")
        except Exception as e:
            logger.warning(f"思考过程生成失败，使用空字符串，错误原因: {str(e)}")
        
        return answer, thinking_content

    except Exception as e:
        error_message = str(e)
        if "Input data may contain inappropriate content" in error_message:
            logger.warning(f"视频内容审核未通过，文件路径: {img_file}")
            return error_message, None
        else:
            # logger.exception会自动记录当前异常的堆栈跟踪信息，不需要手动获取
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
    "question": "基于视频中的具体事件和证据，[问题内容]？\\nA. [选项A内容]\\nB. [选项B内容]\\nC. [选项C内容]\\nD. [选项D内容]",
    "answer": "[选项答案内容]",
    "gt": "[A/B/C/D]"
}

**JSON格式严格要求：**
- 输出必须是完全标准的JSON格式，能被json.loads()直接解析
- 所有字符串值必须用双引号包围，包括answer字段的值
- 选项内容中绝对不要使用双引号(")或单引号(')
- 选项格式严格为: A. 选项内容（不要在选项内容外加引号）
- answer字段的值必须用双引号包围，只包含选项的具体内容，不包含A./B./C./D.前缀
- gt字段的值必须用双引号包围，只包含一个字母（A、B、C或D）
- 使用\\n表示换行，不要使用实际换行符
- 所有字段名和字符串值都必须用双引号包围

**完全正确的示例格式：**
{
    "question": "基于视频内容，最能描述场景的是？\\nA. 宁静的自然环境\\nB. 繁华的城市街道\\nC. 现代化的办公空间\\nD. 热闹的购物中心",
    "answer": "宁静的自然环境",
    "gt": "A"
}

**错误示例（绝对不要这样）：**
- "answer": 宁静的自然环境 （错误：缺少双引号）
- "gt": A （错误：缺少双引号）
- "A. "选项内容"" （错误：选项内容有引号）"""
    
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