import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

# 新增：更便于监控的分块策略
MAX_ROWS_PER_CHUNK = 20          # 每块最多多少条（越小越“像流式逐条”）
WRITE_PARTIAL_EACH_CHUNK = True  # 每块完成立刻落盘
PARTIAL_ITEMS_CSV = "llm_items.partial.csv"
PRINT_SAMPLE_PER_CHUNK = 3       # 每块打印前 N 条样例，便于你验收


def _clean_json_maybe(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        return t[l : r + 1]
    return t


def openrouter_chat(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout: int = 60,
    retries: int = 3,
) -> str:
    api_key = os.environ.get(OPENROUTER_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"未检测到环境变量 {OPENROUTER_KEY_ENV}。请先 setx {OPENROUTER_KEY_ENV} \"...\"")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE", ""),
        "X-Title": os.environ.get("OPENROUTER_TITLE", "Zhihu Annual Report"),
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * attempt)
            else:
                raise RuntimeError(f"OpenRouter 调用失败（已重试 {retries} 次）：{e}") from e
    raise RuntimeError(last_err)


def load_csv(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df["upvote_dt"] = pd.to_datetime(df["upvote_time"], errors="coerce")
    df = df.dropna(subset=["upvote_dt"]).copy()
    df = df.reset_index(drop=True)
    df["row_id"] = df.index + 1
    df["full_text"] = (df["question_title"].fillna("") + "\n" + df["answer_excerpt"].fillna("")).str.strip()
    return df


def chunk_rows(
    df: pd.DataFrame,
    max_chars: int = 24000,
    max_rows: int = MAX_ROWS_PER_CHUNK,
) -> List[pd.DataFrame]:
    """
    分块策略：同时限制字符数 & 行数（更利于“准流式”监控）。
    """
    chunks: List[pd.DataFrame] = []
    buf: List[Dict[str, Any]] = []
    cur_chars = 0

    for _, row in df.iterrows():
        record = {
            "row_id": int(row["row_id"]),
            "upvote_time": str(row["upvote_time"]),
            "question_title": str(row.get("question_title", "")),
            "answer_excerpt": str(row.get("answer_excerpt", "")),
            "url": str(row.get("url", "")),
        }
        s = json.dumps(record, ensure_ascii=False)

        need_flush = False
        if buf and (cur_chars + len(s) > max_chars):
            need_flush = True
        if buf and (len(buf) >= max_rows):
            need_flush = True

        if need_flush:
            chunks.append(pd.DataFrame(buf))
            buf = []
            cur_chars = 0

        buf.append(record)
        cur_chars += len(s)

    if buf:
        chunks.append(pd.DataFrame(buf))
    return chunks


def prompt_analyze_chunk(chunk_df: pd.DataFrame) -> List[Dict[str, str]]:
    records = chunk_df.to_dict(orient="records")
    schema_hint = {
        "chunk_id": 1,
        "items": [
            {
                "row_id": 1,
                "category": "科技与AI/职场与学习/情感与家庭/健康与生命/社会与人文/生活方式/其他",
                "keywords": ["关键词1", "关键词2"],
                "sentiment": "积极/中性/消极/混合",
                "entities": ["实体1", "实体2"],
                "reason": "一句话说明为什么这样分类（尽量引用标题语义）",
            }
        ],
        "chunk_keywords": [{"word": "xx", "weight": 0.0}],
    }

    prompt = (
        "你是中文语义分析引擎。请对我提供的知乎赞同记录逐条做语义标注。\n"
        "要求：\n"
        "1) 必须逐条输出 items，row_id 必须原样保留；\n"
        "2) category 从给定集合里选一个；\n"
        "3) keywords 为该条内容的核心词（2-6 个），不要太泛；\n"
        "4) entities 可为空数组；\n"
        "5) reason 一句话即可；\n"
        "6) 输出必须是严格 JSON（不要 Markdown、不要多余文字）。\n"
        "category 集合：科技与AI、职场与学习、情感与家庭、健康与生命、社会与人文、生活方式、其他。\n\n"
        f"records(JSON 数组)：\n{json.dumps(records, ensure_ascii=False)}\n\n"
        f"按此 JSON 结构返回：\n{json.dumps(schema_hint, ensure_ascii=False)}"
    )
    return [{"role": "user", "content": prompt}]


def _items_df_from_items(items: List[Dict[str, Any]]) -> pd.DataFrame:
    items_df = pd.DataFrame(items)

    for col in ["row_id", "category", "keywords", "sentiment", "entities", "reason"]:
        if col not in items_df.columns:
            items_df[col] = None

    items_df["keywords"] = items_df["keywords"].apply(lambda x: ",".join(x) if isinstance(x, list) else (x or ""))
    items_df["entities"] = items_df["entities"].apply(lambda x: ",".join(x) if isinstance(x, list) else (x or ""))
    items_df["row_id"] = pd.to_numeric(items_df["row_id"], errors="coerce").astype("Int64")
    return items_df


def analyze_all_items_via_llm(
    df: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    max_chars_per_chunk: int = 24000,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    chunks = chunk_rows(df, max_chars=max_chars_per_chunk, max_rows=MAX_ROWS_PER_CHUNK)
    all_items: List[Dict[str, Any]] = []
    chunk_outputs: List[Dict[str, Any]] = []

    # 可选：清理旧的 partial
    if WRITE_PARTIAL_EACH_CHUNK and os.path.exists(PARTIAL_ITEMS_CSV):
        os.remove(PARTIAL_ITEMS_CSV)

    for idx, chunk_df in enumerate(chunks, start=1):
        row_ids = chunk_df["row_id"].tolist()
        print(f"Chunk {idx}/{len(chunks)} 开始：rows={len(row_ids)}，row_id范围={min(row_ids)}-{max(row_ids)}")

        messages = prompt_analyze_chunk(chunk_df)
        raw = openrouter_chat(messages, model=model, temperature=0.1, max_tokens=2500, timeout=90, retries=3)
        cleaned = _clean_json_maybe(raw)
        obj = json.loads(cleaned)

        obj["chunk_id"] = idx
        chunk_outputs.append(obj)

        items = obj.get("items", [])
        all_items.extend(items)

        # 每块立即转 DF、落盘、打印样例
        chunk_items_df = _items_df_from_items(items)

        if WRITE_PARTIAL_EACH_CHUNK:
            header = not os.path.exists(PARTIAL_ITEMS_CSV)
            chunk_items_df.to_csv(PARTIAL_ITEMS_CSV, mode="a", header=header, index=False, encoding="utf-8-sig")

        if PRINT_SAMPLE_PER_CHUNK and not chunk_items_df.empty:
            sample = chunk_items_df.head(PRINT_SAMPLE_PER_CHUNK)[["row_id", "category", "keywords", "reason"]]
            print("样例：")
            for _, r in sample.iterrows():
                print(f"  - row_id={int(r['row_id'])} | {r['category']} | {r['keywords']} | {r['reason']}")

        print(f"Chunk {idx}/{len(chunks)} 完成：items={len(items)}")

    items_df = _items_df_from_items(all_items)
    return items_df, chunk_outputs

def prompt_reduce_summary(
    stats: Dict[str, Any],
    sample_titles: List[str],
) -> List[Dict[str, str]]:
    schema_hint = {
        "persona": "四到八字的人设标签",
        "annual_comment": "120-200字年度评语（温暖、可行动）",
        "theme_insights": ["主题洞察1", "主题洞察2", "主题洞察3"],
        "time_insight": "基于活跃时间的作息/节律洞察（带一点幽默）",
        "suggestions": ["建议1", "建议2", "建议3"],
        "interest_shift": "简述一年中兴趣迁移/阶段变化（可按季度或月份）",
        "risk_or_bias": "可选：信息摄入偏差或风险提示（不吓人、可行动）",
    }

    prompt = (
        "你是一名数据叙事专家。基于以下统计摘要（来自逐条语义标注后的聚合结果），"
        "生成一份高质量、克制但有洞察的年度复盘。\n"
        "要求：输出严格 JSON，不要 Markdown。\n\n"
        f"统计摘要 stats(JSON)：\n{json.dumps(stats, ensure_ascii=False)}\n\n"
        f"代表性标题 sample_titles(JSON)：\n{json.dumps(sample_titles, ensure_ascii=False)}\n\n"
        f"返回结构示例：\n{json.dumps(schema_hint, ensure_ascii=False)}"
    )
    return [{"role": "user", "content": prompt}]


def build_stats(df: pd.DataFrame, items_df: pd.DataFrame) -> Dict[str, Any]:
    # 时间统计（Python 算，避免 LLM 口胡）
    df2 = df.copy()
    df2["weekday"] = df2["upvote_dt"].dt.weekday
    df2["hour"] = df2["upvote_dt"].dt.hour
    weekday_counts = df2["weekday"].value_counts().to_dict()
    hour_counts = df2["hour"].value_counts().to_dict()

    # 语义分类统计（LLM 标注结果）
    cat_counts = items_df["category"].value_counts(dropna=True).to_dict()

    # 关键词统计（LLM 的逐条 keywords 聚合）
    kw_counter: Dict[str, int] = {}
    for s in items_df["keywords"].fillna("").astype(str).tolist():
        for w in [x.strip() for x in s.split(",") if x.strip()]:
            kw_counter[w] = kw_counter.get(w, 0) + 1
    top_kw = sorted(kw_counter.items(), key=lambda x: x[1], reverse=True)[:50]

    period = {
        "start": df2["upvote_dt"].min().strftime("%Y-%m-%d"),
        "end": df2["upvote_dt"].max().strftime("%Y-%m-%d"),
    }

    return {
        "period": period,
        "total": int(len(df2)),
        "active_days": int(df2["upvote_dt"].dt.date.nunique()),
        "weekday_counts": weekday_counts,
        "hour_counts": hour_counts,
        "category_counts": cat_counts,
        "top_keywords": [{"word": w, "count": c} for w, c in top_kw],
    }


def generate_annual_report_via_api(
    csv_file: str = "cleaned_activities.csv",
    model: str = DEFAULT_MODEL,
    out_items_csv: str = "llm_items.csv",
    out_analysis_json: str = "llm_analysis.json",
):
    df = load_csv(csv_file)

    # 1) 逐条标注（多次 API 调用，全量分块喂）
    items_df, chunk_outputs = analyze_all_items_via_llm(df, model=model, max_chars_per_chunk=24000)
    items_df.to_csv(out_items_csv, index=False, encoding="utf-8-sig")

    # 2) 汇总复盘（再一次 API 调用）
    stats = build_stats(df, items_df)
    sample_titles = df["question_title"].dropna().astype(str).tolist()[:40]

    messages = prompt_reduce_summary(stats, sample_titles)
    raw = openrouter_chat(messages, model=model, temperature=0.6, max_tokens=1200, timeout=90, retries=3)
    cleaned = _clean_json_maybe(raw)
    analysis = json.loads(cleaned)

    payload = {
        "model": model,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats,
        "analysis": analysis,
        "note": "items 来自分块逐条标注；analysis 来自聚合后二次复盘。",
    }
    with open(out_analysis_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"已生成：{out_items_csv}")
    print(f"已生成：{out_analysis_json}")


if __name__ == "__main__":
    generate_annual_report_via_api()