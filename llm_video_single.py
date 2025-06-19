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

# 配置logger级别（可通过环境变量LOG_LEVEL设置）
LOG_LEVEL = os.getenv('LOG_LEVEL', 'ERROR').upper()
logger.remove()  # 移除默认的logger
logger.add(lambda msg: print(msg, end=""), level=LOG_LEVEL)

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
        current_prompt = qa_prompts.get(qa_type, "") + "\n" + f"""返回的结果使用json格式，并使用中文，格式如下，只需要整体caption：\n + {{
            "caption":  
            }}"""
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
        response = MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model="qwen-vl-max-latest",
            messages=messages,
            # stream=True
            result_format='message',
            response_format={'type': 'json_object'}
        )
        # print("=" * 20 + "思考过程" + "=" * 20)
        # for chunk in response:
        #     # 如果思考过程与回复皆为空，则忽略
        #     message = chunk.output.choices[0].message
        #     reasoning_content_chunk = message.get("reasoning_content", None)

        #     if (chunk.output.choices[0].message.content == [] and
        #         reasoning_content_chunk == ""):
        #         pass
        #     else:
        #         # 如果当前为思考过程
        #         if reasoning_content_chunk != None and chunk.output.choices[0].message.content == []:
        #             print(chunk.output.choices[0].message.reasoning_content, end="")
        #             reasoning_content += chunk.output.choices[0].message.reasoning_content
        #         # 如果当前为回复
        #         elif chunk.output.choices[0].message.content != []:
        #             if not is_answering:
        #                 print("\n" + "=" * 20 + "完整回复" + "=" * 20)
        #                 is_answering = True
        #             print(chunk.output.choices[0].message.content[0]["text"], end="")
        #             answer_content += chunk.output.choices[0].message.content[0]["text"]
        # ... existing code ...
        clean_content = response["output"]["choices"][0]["message"].content[0]["text"]
        # if clean_content.startswith('```json'):
        #     clean_content = clean_content[7:]
        # if clean_content.endswith('```'):
        #     clean_content = clean_content[:-3]
        logger.info(f"模型输出: {clean_content}")
        
        think_prompt = f"""
        根据以下的问答对和视频内容，生成思考过程：
        {clean_content}

        请按以下格式生成分析内容：
        1. 输出格式要求：使用JSON格式，结构为：
        {{
            "think": "在此填写完整的分析内容"
        }}

        2. 分析内容结构要求：
        <规划>
        分析视频的整体结构和关键点
        </规划>

        <字幕>
        按时间段分析视频内容，格式如：
        从00:00-00:02，视频展示了华为Mate 60和iPhone 15的渲染图，并将它们框定为一场竞争。
        ...
        </字幕>

        <推理>
        基于视频内容进行深入分析和推理
        </推理>

        <总结>
        对整个视频内容进行全面总结
        </总结>

        注意：
        1. 所有内容需要整合在一个完整的JSON字符串中
        2. 时间段分析要准确且连贯
        3. 确保所有XML标签配对完整
        4. 分析要全面且有逻辑性
        """
        # f"""
        # 根据以下的问答对和视频内容，生成思考过程
        # {clean_content}
        # 其中这个问答对同样是根据视频内容得出。请按时间段分析视频中各个部分的小主题，如
        # ‘’‘从00:00-00:02，展示了华为Mate 60的渲染图，显示手机背部有一个圆形相机模块，位于上半部分中央。’‘’
        # ...
        # 在分析结束后总结，在think的输出中体现，think中需要以<规划><规划><字幕><字幕><推理><推理><总结><总结>的格式进行输出，在总结中需要对视频内容进行总结。输出结果以json格式输出，格式如下：{{
        #     "think":
        # }}
        # 在<字幕><字幕>请按时间段分析视频中各个部分的小主题，如
        # ‘’‘从0:01-0:05，视频展示了华为Mate 60和iPhone 15的渲染图，并将它们框定为一场竞争。’‘’
        # ...
        # """
        response_thinking = MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model="qwen-vl-max-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "video": video_path
                        },
                        {"text": think_prompt}
                    ]
                }
            ],
            # stream=True,
            result_format='message',
            response_format={'type': 'json_object'}
        )
        
        reasoning_content_think = ""  # 定义完整思考过程
        answer_content_think = ""     # 定义完整回复
        
        # print("=" * 20 + "think过程" + "=" * 20)
        # for chunk in response_thinking:
        #     # 如果思考过程与回复皆为空，则忽略
        #     message = chunk.output.choices[0].message
        #     reasoning_content_chunk = message.get("reasoning_content", None)

        #     if (chunk.output.choices[0].message.content == [] and
        #         reasoning_content_chunk == ""):
        #         pass
        #     else:
        #         # 如果当前为思考过程
        #         if reasoning_content_chunk != None and chunk.output.choices[0].message.content == []:
        #             print(chunk.output.choices[0].message.reasoning_content, end="")
        #             reasoning_content_think += chunk.output.choices[0].message.reasoning_content
        #         # 如果当前为回复
        #         elif chunk.output.choices[0].message.content != []:
        #             if not is_answering:
        #                 print("\n" + "=" * 20 + "think完整回复" + "=" * 20)
        #                 is_answering = True
        #             print(chunk.output.choices[0].message.content[0]["text"], end="")
        #             answer_content_think += chunk.output.choices[0].message.content[0]["text"]
        # print("=" * 20 + "think过程答案" + "=" * 20)
        #... existing code...
        clean_content_think = response_thinking["output"]["choices"][0]["message"].content[0]["text"]
        logger.info(f"think过程答案: {clean_content_think}")
        think_json = json.loads(clean_content_think)
        thinking_content = think_json.get("think", "")
        
        answer = json.loads(clean_content)
        return answer, thinking_content

    
    except Exception as e:
        error_message = str(e)
        import traceback
        stack_trace = traceback.format_exc()
        
        if "Input data may contain inappropriate content" in error_message:
            logger.warning(f"视频内容审核未通过，文件路径: {img_file}")
            logger.debug(f"错误堆栈:\n{stack_trace}")
            return None, None
        else:
            logger.error(f"调用API失败: {error_message}，文件路径: {img_file}")
            logger.error(f"错误堆栈:\n{stack_trace}")
            return None, None

def process_single_video_qa(video_file: str, qa_types: list = None) -> dict:
    """
    处理单个视频的问答生成
    
    Args:
        video_file (str): 视频文件路径
        qa_types (list, optional): 问答类型列表。如果为None，则使用默认类型列表
        
    Returns:
        dict: 包含所有问答对的结果字典
    """
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
    
    # 构建提示语
    prompt = """要求如下：
            1. 问题设计要求：
            - 需要整合视觉和听觉信息来回答，请深度思考视频中的内容，请按时间段分析视频中各个部分的小主题,并在分析结束后总结
            - 需要适当的结合字幕和音频来总结信息
            - 提供4个选项（A/B/C/D），每个选项都要合理且有区分度
            - 答案应该需要深入分析视频内容才能得出
            2. 输出格式：
            {
                "question": "基于视频中的具体事件和证据，[具体问题]？\\nA. [选项A]\\nB. [选项B]\\nC. [选项C]\\nD. [选项D]",
                "answer": "[简短答案]",
                "gt": "[正确选项字母]"
            }
            注意确保问题和选项的措辞清晰、准确、无歧义，question字段中需要包含具体的问题以及选项，answer字段中只需要提供选项中的答案即可，不需要多余的解释，gt字段中只需要提供正确选项的字母。"""
    
    # 初始化结果字典
    results = {
        "video_path": str(video_file),
        "annotaion": {}
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
                try:
                    result, thinking = future.result()
                    if result:
                        results["annotaion"][qa_type] = result
                        results["annotaion"][qa_type]["thinking"] = thinking
                        print(f"{video_file}：{qa_type} 生成成功！")
                    else:
                        print(f"{video_file}：{qa_type} 生成失败！")
                except Exception as e:
                    logger.error(f"处理 {qa_type} 问答对时出错: {str(e)}")
                pbar.update(1)
    
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
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n所有问答对已生成完成，结果已保存到 {output_file}")
    except Exception as e:
        logger.error(f"保存结果文件时出错: {str(e)}")