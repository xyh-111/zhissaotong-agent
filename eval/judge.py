"""
LLM-as-Judge 评测裁判
使用大模型作为裁判，对 Agent 回答的「任务完成度」与「幻觉情况」进行打分。

注意：
1. 当前复用项目的 chat_model 作为裁判，存在「自我偏好」偏差（用同一模型评自己的回答容易虚高）；
   生产环境建议换用独立、更强的裁判模型，并固定 temperature=0 以保证可复现。
2. 裁判输出统一要求为 JSON，解析做了容错处理（若模型多输出解释文字，也会尝试提取 JSON 块）。
"""
import json
import re

from model.factory import ChatModelFactory
from utils.config_handler import eval_conf

# 独立裁判模型：可配置独立模型名 + temperature=0，减少自我偏好偏差、保证打分可复现
_judge_model = ChatModelFactory(
    model_name=eval_conf.get("judge_model_name"),
    temperature=eval_conf.get("judge_temperature", 0),
).generator()


def _extract_json(text: str) -> dict:
    """容错地从模型输出中提取 JSON 对象"""
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取第一个 {...} 块
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _invoke_judge(prompt: str) -> str:
    """调用裁判模型并返回纯文本内容"""
    resp = _judge_model.invoke(prompt)
    content = resp.content
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return content or ""


def judge_task_completion(query: str, reference_points: list, answer: str) -> tuple[bool, str]:
    """判断 Agent 回答是否完成了用户核心需求，返回 (是否完成, 理由)"""
    ref = "\n".join(f"- {p}" for p in reference_points) if reference_points else "（无参考要点）"
    prompt = f"""你是一个评测裁判，负责判断一个智能客服 Agent 的回答是否完成了用户的核心需求。

【用户问题】
{query}

【参考要点】（回答应覆盖这些关键信息）
{ref}

【Agent 回答】
{answer}

请判断该回答是否准确、完整地解决了用户的核心需求。
只输出 JSON，不要输出其他任何内容：
{{"completed": true/false, "reason": "一句话说明理由"}}
"""
    result = _extract_json(_invoke_judge(prompt))
    return result.get("completed", False), result.get("reason", "judge 解析失败")


def judge_hallucination(query: str, reference_points: list, answer: str) -> tuple[bool, str]:
    """判断 Agent 回答是否包含幻觉，返回 (是否有幻觉, 理由)"""
    ref = "\n".join(f"- {p}" for p in reference_points) if reference_points else "（无参考要点）"
    prompt = f"""你是一个评测裁判，负责判断智能客服回答是否包含「幻觉」（无依据的事实）。

【用户问题】
{query}

【参考要点】（可信的事实来源）
{ref}

【Agent 回答】
{answer}

判断标准：
1. 回答中出现与参考要点相矛盾的事实 -> 幻觉；
2. 回答中出现参考要点完全未提及、且无法由常识推断的具体事实 -> 幻觉；
3. 基于常识、与参考要点不冲突的一般性建议不算幻觉。

只输出 JSON，不要输出其他任何内容：
{{"has_hallucination": true/false, "reason": "一句话说明幻觉内容，或确认无幻觉"}}
"""
    result = _extract_json(_invoke_judge(prompt))
    return result.get("has_hallucination", False), result.get("reason", "judge 解析失败")
