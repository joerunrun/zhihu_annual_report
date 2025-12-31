import base64
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import jieba
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from wordcloud import WordCloud
import requests
import os


CATEGORY_KEYWORDS = {
    "科技与AI": ["ai", "算法", "科技", "芯片", "模型", "网络", "计算机", "自动化"],
    "职场与学习": ["工作", "职场", "学习", "博士", "论文", "科研", "offer"],
    "情感与家庭": ["爱情", "婚姻", "女朋友", "恋爱", "家庭", "孩子", "生娃", "父母"],
    "健康与生命": ["医生", "疾病", "癌", "手术", "健康", "生命"],
    "社会与人文": ["历史", "社会", "政治", "文化", "城市", "犯罪"],
    "生活方式": ["旅游", "水果", "美食", "生活", "消费", "购物"],
}

STOPWORDS = {
    "什么", "没有", "觉得", "怎么", "一个", "我们", "他们", "就是", "可以", "因为", "如果", "不会", "这样",
    "这个", "那些", "自己", "以及", "还有", "已经", "可能", "需要", "这是", "这些", "然后", "为什么",
    "问题", "很多", "时候", "现在", "不是", "回答", "如何", "一下", "知道", "com", "www", "http", "https",
    "span", "class", "div", "br", "href", "target", "blank" # 过滤 HTML 垃圾
}

WEEKDAY_MAP = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五", 5: "周六", 6: "周日"}
FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
]

# OpenRouter / Gemini 3 Flash 配置
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-3.0-flash-001"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"


def pick_font_path() -> str | None:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def summarize_for_llm(df: pd.DataFrame, keyword_counter: Counter) -> str:
  top_kw = ", ".join([f"{k}({v})" for k, v in list(keyword_counter.items())[:15]])
  category_share = df["category"].value_counts(normalize=True).mul(100).round(1)
  category_lines = "; ".join([f"{cat}: {pct}%" for cat, pct in category_share.items()])
  busiest_weekday = df.groupby("weekday").size().idxmax()
  busiest_hour = int(df.groupby("hour").size().idxmax())
  period = f"{df['upvote_dt'].min().strftime('%Y-%m-%d')} 至 {df['upvote_dt'].max().strftime('%Y-%m-%d')}"
  return (
    f"时间范围: {period}\n"
    f"总赞同: {len(df)}，活跃天数: {df['date'].nunique()}\n"
    f"最活跃: {busiest_weekday} 的 {busiest_hour}:00 时段\n"
    f"主题分布: {category_lines}\n"
    f"高频关键词: {top_kw}\n"
  )


def call_llm_commentary(df: pd.DataFrame, keyword_counter: Counter) -> str:
  api_key = os.environ.get(OPENROUTER_KEY_ENV)
  if not api_key:
    return "未检测到 OPENROUTER_API_KEY，已跳过 AI 评语生成。"

  summary = summarize_for_llm(df, keyword_counter)
  prompt = (
    "你是一名洞察力很强的数据叙事者，请基于以下知乎年度阅读摘要，生成一段 120-200 字的中文评语。"
    "要求：\n"
    "1) 先给出读者的年度主题倾向，语气自信但克制；\n"
    "2) 点出时间作息特征；\n"
    "3) 给出 2-3 条轻量建议，保持温暖、具体、可行动；\n"
    "4) 不要罗列原始数据，也不要出现数字列表。\n"
    f"\n年度摘要：\n{summary}"
  )

  payload = {
    "model": OPENROUTER_MODEL,
    "messages": [
      {"role": "user", "content": prompt}
    ],
  }

  headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": os.environ.get("OPENROUTER_SITE", ""),
    "X-Title": os.environ.get("OPENROUTER_TITLE", "Zhihu Annual Report"),
  }

  try:
    resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "") or "AI 返回为空"
  except Exception as e:
    return f"AI 评语生成失败：{e}"


def load_llm_items(path: str = "llm_items.csv") -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "row_id" in df.columns:
        df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").astype("Int64")
    return df


def load_llm_analysis(path: str = "llm_analysis.json") -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["upvote_dt"] = pd.to_datetime(df["upvote_time"], errors="coerce")
    df = df.dropna(subset=["upvote_dt"]).copy()
    df = df.reset_index(drop=True)
    df["row_id"] = df.index + 1

    df["date"] = df["upvote_dt"].dt.date
    df["month"] = df["upvote_dt"].dt.to_period("M").dt.to_timestamp()
    df["weekday_num"] = df["upvote_dt"].dt.weekday
    df["weekday"] = df["weekday_num"].map(WEEKDAY_MAP)
    df["hour"] = df["upvote_dt"].dt.hour
    df["title_excerpt"] = (df["question_title"].fillna("") + " " + df["answer_excerpt"].fillna("")).str.strip()
    return df


def categorize_row(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return category
    return "其他"


def normalize_category(cat: str) -> str:
    """清洗 LLM 可能输出的脏数据，例如 '职场 with 学习'"""
    if not isinstance(cat, str):
        return "其他"
    cat = cat.strip()
    # 修复 LLM 偶尔的中英夹杂
    cat = cat.replace(" with ", "与").replace(" and ", "与")
    return cat


def count_keywords_smart(df: pd.DataFrame, top_k: int = 200) -> Counter:
    """
    优先使用 LLM 提取的 keywords 列。
    如果 keywords 列不存在或为空，才回退到 jieba 分词。
    """
    counter: Counter[str] = Counter()
    
    # 1. 优先尝试使用 LLM 的 keywords 列
    if "keywords" in df.columns:
        print("✅ 正在使用 AI 提取的关键词进行统计...")
        # keywords 列通常是 "词1,词2,词3" 的字符串
        valid_keywords = df["keywords"].dropna().astype(str)
        for kw_str in valid_keywords:
            # 按逗号分割，去除空白
            words = [w.strip() for w in kw_str.replace("，", ",").split(",") if w.strip()]
            for w in words:
                # 再次过滤停用词（虽然 LLM 已经过滤过，但双重保险）
                if w.lower() not in STOPWORDS and len(w) > 1:
                    counter[w] += 1
    
    # 2. 如果计数器为空（说明没合并成功或没数据），回退到 Jieba
    if not counter:
        print("⚠️ 未检测到 AI 关键词，回退到 Jieba 原始分词...")
        texts = df["title_excerpt"].fillna("").tolist()
        for text in texts:
            for token in jieba.lcut(text):
                token = token.strip().lower()
                if len(token) <= 1 or token.isdigit() or token in STOPWORDS:
                    continue
                counter[token] += 1
                
    return Counter(dict(counter.most_common(top_k)))


def build_wordcloud(freqs: Counter) -> str:
    font_path = pick_font_path()
    wc = WordCloud(
        font_path=font_path,
        width=900,
        height=500,
        background_color="white",
        max_words=150, # 减少一点词数，让大词更突出
        colormap="viridis",
        margin=2
    )
    wc.generate_from_frequencies(freqs)
    buffer = BytesIO()
    wc.to_image().save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def make_figures(df: pd.DataFrame, keyword_counter: Counter):
    # 1. 月度图表（修复截断问题）
    month_df = df.groupby("month").size().reset_index(name="count")
    fig_month = px.bar(
        month_df,
        x="month",
        y="count",
        title="按月阅读活跃度",
        labels={"month": "月份", "count": "赞同次数"},
        color="count",
        color_continuous_scale="Blues",
    )
    # 修复：设置日期格式，并增加边距防止截断
    fig_month.update_xaxes(tickformat="%Y-%m")
    fig_month.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # 2. 热力图
    fig_week_hour = px.density_heatmap(
        df,
        x="hour",
        y="weekday",
        histfunc="count",
        category_orders={"weekday": list(WEEKDAY_MAP.values())},
        nbinsx=24,
        color_continuous_scale="Viridis",
        title="一周内的活跃时段热力图",
        labels={"hour": "小时", "weekday": "周几", "count": "次数"},
    )

    # 3. 周分布
    fig_weekday = px.bar(
        df.groupby("weekday").size().reindex(list(WEEKDAY_MAP.values())).reset_index(name="count"),
        x="weekday",
        y="count",
        title="周内活跃分布",
        labels={"weekday": "周几", "count": "赞同次数"},
        color="count",
        color_continuous_scale="Teal",
    )

    # 4. 关键词条形图
    top_keywords = list(keyword_counter.items())[:15] # 只取前15个，避免太长
    kw_df = pd.DataFrame(top_keywords, columns=["keyword", "count"])
    fig_keywords = px.bar(
        kw_df,
        x="count",
        y="keyword",
        orientation="h",
        title="AI 提取核心关键词 Top 15",
        labels={"keyword": "关键词", "count": "出现次数"},
        color="count",
        color_continuous_scale="Oranges",
    )
    fig_keywords.update_layout(yaxis=dict(autorange="reversed"))

    # 5. 分类图表
    category_df = df.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False)
    fig_category = px.bar(
        category_df,
        x="category",
        y="count",
        title="关注主题分布",
        labels={"category": "主题", "count": "赞同次数"},
        color="count",
        color_continuous_scale="PuBuGn",
    )

    figs = {
        "month": plot(fig_month, include_plotlyjs=False, output_type="div"),
        "week_hour": plot(fig_week_hour, include_plotlyjs=False, output_type="div"),
        "weekday": plot(fig_weekday, include_plotlyjs=False, output_type="div"),
        "keywords": plot(fig_keywords, include_plotlyjs=False, output_type="div"),
        "category": plot(fig_category, include_plotlyjs=False, output_type="div"),
    }
    return figs


def render_html(df: pd.DataFrame, figs: dict, wordcloud_data_uri: str, keyword_counter: Counter, output_html: str):
    total = int(len(df))

    if total > 0:
        first_dt = df["upvote_dt"].min()
        last_dt = df["upvote_dt"].max()
        first_date = first_dt.strftime("%Y-%m-%d")
        last_date = last_dt.strftime("%Y-%m-%d")
        span_days = int((last_dt.normalize() - first_dt.normalize()).days) + 1
        active_days = int(df["date"].nunique())
        busiest_weekday = df.groupby("weekday").size().idxmax()
        busiest_hour = int(df.groupby("hour").size().idxmax())
    else:
        first_date = last_date = "N/A"
        span_days = 0
        active_days = 0
        busiest_weekday = "N/A"
        busiest_hour = 0

    # Top 5 关注问题
    if "question_title" in df.columns and total > 0:
        top_questions = (
            df["question_title"]
            .fillna("")
            .astype(str)
            .loc[lambda s: s.str.strip().ne("")]
            .value_counts()
            .head(5)
            .index
            .tolist()
        )
    else:
        top_questions = []
    question_list = "".join(f"<li>{q}</li>" for q in top_questions) or "<li>暂无</li>"

    # 高频词列表
    top_kw_list = "".join(
        f"<li>{k} <span style='color:#666; font-size:0.8em'>x{v}</span></li>" for k, v in list(keyword_counter.items())[:15]
    ) or "<li>暂无</li>"

    # 读取 LLM 汇总
    llm_analysis = df.attrs.get("llm_analysis") or {}
    analysis_core = (llm_analysis.get("analysis") if isinstance(llm_analysis, dict) else {}) or {}
    persona = analysis_core.get("persona", "")
    annual_comment = analysis_core.get(
        "annual_comment",
        df.attrs.get("llm_comment", "未生成，请检查 API Key 或先运行 api_analyse.py。"),
    )
    suggestions = analysis_core.get("suggestions", [])
    theme_insights = analysis_core.get("theme_insights", [])
    time_insight = analysis_core.get("time_insight", "")

    suggestions_html = "".join(f"<li>{s}</li>" for s in suggestions) if isinstance(suggestions, list) else ""
    theme_html = "".join(f"<li>{t}</li>" for t in theme_insights) if isinstance(theme_insights, list) else ""

    ai_theme_section = f"""
    <section>
      <h2>AI 主题洞察</h2>
      <div class="card"><ul>{theme_html}</ul></div>
    </section>
    """ if theme_html else ""

    ai_suggestion_section = f"""
    <section>
      <h2>给明年的建议</h2>
      <div class="card"><ul>{suggestions_html}</ul></div>
    </section>
    """ if suggestions_html else ""

    time_insight_html = f"<p style='margin-top:12px; color: var(--muted);'>{time_insight}</p>" if time_insight else ""

    ai_comment_section = f"""
    <section>
      <h2>AI 生成的评语</h2>
      <div class="card">
        <p style="margin:0; white-space: pre-line;">{annual_comment}</p>
        {time_insight_html}
      </div>
    </section>
    """

    html = f"""
<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>2025 知乎阅读年度报告</title>
  <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
  <style>
    :root {{
      --bg: #0b1021;
      --panel: rgba(255, 255, 255, 0.04);
      --accent: #8ee1ff;
      --accent-2: #ffb86c;
      --text: #e7ecf2;
      --muted: #a3acc2;
      --card-shadow: 0 14px 45px rgba(0,0,0,0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 0;
      background: radial-gradient(circle at 20% 20%, rgba(100,149,237,0.12), transparent 35%),
                  radial-gradient(circle at 80% 0%, rgba(255,182,108,0.12), transparent 25%),
                  var(--bg);
      color: var(--text);
      font-family: 'Space Grotesk', 'Noto Sans SC', sans-serif;
      line-height: 1.65;
    }}
    header {{
      padding: 48px 32px;
      background: linear-gradient(120deg, rgba(142,225,255,0.3), rgba(255,182,108,0.35));
      color: #0b1021;
    }}
    header h1 {{ margin: 0 0 8px; font-size: 32px; }}
    header p {{ margin: 0; max-width: 900px; font-weight: 500; }}
    .container {{ padding: 32px; max-width: 1200px; margin: 0 auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 18px; box-shadow: var(--card-shadow); backdrop-filter: blur(6px); }}
    .stat-value {{ font-size: 26px; font-weight: 700; color: #ffffff; }}
    .stat-label {{ color: var(--muted); }}
    section {{ margin: 32px 0; }}
    section h2 {{ margin-bottom: 12px; }}
    .figure {{ background: var(--panel); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; box-shadow: var(--card-shadow); }}
    .flex {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
    .wordcloud img {{ width: 100%; border-radius: 12px; box-shadow: var(--card-shadow); }}
    ul {{ padding-left: 20px; }}
    a {{ color: var(--accent-2); }}
  </style>
</head>
<body>
  <header>
    <h1>2025 知乎阅读年度报告</h1>
    <p>基于你的赞同记录，洞察一年来的兴趣轨迹、时间节律与关键词画像（含 Gemini 3 Flash 语义分析）。</p>
    {f"<p><strong>年度人设：</strong>{persona}</p>" if persona else ""}
  </header>

  <div class=\"container\">
    <div class=\"grid\">
      <div class=\"card\"><div class=\"stat-value\">{total}</div><div class=\"stat-label\">年度赞同</div></div>
      <div class=\"card\"><div class=\"stat-value\">{first_date} — {last_date}</div><div class=\"stat-label\">时间范围</div></div>
      <div class=\"card\"><div class=\"stat-value\">{active_days} 天 / 共 {span_days} 天</div><div class=\"stat-label\">活跃天数</div></div>
      <div class=\"card\"><div class=\"stat-value\">{busiest_weekday}</div><div class=\"stat-label\">最忙的星期</div></div>
      <div class=\"card\"><div class=\"stat-value\">{busiest_hour}:00</div><div class=\"stat-label\">高频时段</div></div>
    </div>

    <section>
      <h2>时间脉络</h2>
      <div class=\"grid\">
        <div class=\"figure\">{figs['month']}</div>
        <div class=\"figure\">{figs['weekday']}</div>
      </div>
      <div class=\"figure\" style=\"margin-top:16px;\">{figs['week_hour']}</div>
    </section>

    <section>
      <h2>兴趣画像</h2>
      <div class=\"figure\">{figs['category']}</div>
      <h3>Top 5 关注问题</h3>
      <ul>{question_list}</ul>
    </section>

    <section>
      <h2>关键词与阅读气质</h2>
      <div class=\"flex\">
        <div class=\"figure\" style=\"flex:1; min-width:340px;\">{figs['keywords']}</div>
        <div class=\"card wordcloud\" style=\"flex:1; min-width:280px;\">
          <h3>词云</h3>
          <img src=\"{wordcloud_data_uri}\" alt=\"wordcloud\" />
        </div>
      </div>
      <h3>高频词汇</h3>
      <ul>{top_kw_list}</ul>
    </section>

    {ai_comment_section}
    {ai_theme_section}
    {ai_suggestion_section}

    <section>
      <h2>原始数据片段</h2>
      <div class=\"card\">
        <pre style=\"white-space: pre-wrap; color: var(--muted); margin:0;\">{json.dumps(df[['upvote_time','question_title']].head(8).to_dict(orient='records'), ensure_ascii=False)}</pre>
      </div>
    </section>
  </div>
</body>
</html>
"""
    Path(output_html).write_text(html, encoding="utf-8")


def generate_report(csv_path: str = "cleaned_activities.csv", output_html: str = "annual_report.html"):
    df = load_data(csv_path)
    if df.empty:
        raise ValueError("数据为空，无法生成报告")

    # 1) 先用本地规则给个默认 category（兜底）
    df["category"] = df["title_excerpt"].apply(categorize_row)

    # 2) 若存在 LLM 逐条标注结果，则用它覆盖 category（更准）
    llm_items = load_llm_items("llm_items.csv")
    if llm_items is not None and "row_id" in llm_items.columns:
        # 关键修改：同时合并 category 和 keywords
        cols_to_merge = ["row_id", "category"]
        if "keywords" in llm_items.columns:
            cols_to_merge.append("keywords")
            
        df = df.merge(llm_items[cols_to_merge], on="row_id", how="left", suffixes=("", "_llm"))
        
        # 覆盖 Category
        if "category_llm" in df.columns:
            df["category"] = df["category_llm"].fillna(df["category"])
            df.drop(columns=["category_llm"], inplace=True)
            
        # 覆盖 Keywords (如果有)
        if "keywords_llm" in df.columns:
            df["keywords"] = df["keywords_llm"] # 优先用 LLM 的
            df.drop(columns=["keywords_llm"], inplace=True)

    # 3) 清洗 Category (修复 "职场 with 学习" 等脏数据)
    df["category"] = df["category"].apply(normalize_category)

    # 4) 统计关键词 (优先用 LLM 的 keywords 列)
    keyword_counter = count_keywords_smart(df, top_k=300)
    
    wordcloud_data_uri = build_wordcloud(keyword_counter)
    figs = make_figures(df, keyword_counter)

    # 5) 加载 LLM 汇总
    llm_analysis = load_llm_analysis("llm_analysis.json")
    if llm_analysis:
        df.attrs["llm_analysis"] = llm_analysis

    # 6) 额外的“短评语”
    llm_comment = call_llm_commentary(df, keyword_counter)
    df.attrs["llm_comment"] = llm_comment
    render_html(df, figs, wordcloud_data_uri, keyword_counter, output_html)
    print(f"年度报告已生成，保存到 {output_html}")
if __name__ == "__main__":
    generate_report()