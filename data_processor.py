import json
import pandas as pd
from datetime import datetime

def process_raw_data(input_file="raw_activities.json", output_file="cleaned_activities.csv"):
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    processed_list = []
    for item in raw_data:
        verb = item.get("verb", "")
        # 只处理赞同回答的动态
        if verb != "MEMBER_VOTEUP_ANSWER":
            continue
            
        target = item.get("target", {})
        
        # 提取点赞时间（item 级别的时间）
        upvote_timestamp = item.get("created_time")
        
        # 过滤非 2025 年的数据 (1735689600 为 2025年元旦)
        if not upvote_timestamp or upvote_timestamp < 1735689600:
            continue

        dt_object = datetime.fromtimestamp(upvote_timestamp)
        time_str = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        
        title = target.get("question", {}).get("title") or target.get("title") or ""
        # 建议使用 excerpt 替代 content，避免 CSV 文件过大且包含大量 HTML 标签
        content = target.get("excerpt") or ""
        
        processed_list.append({
            "upvote_time": time_str,
            "verb": "赞同了回答",
            "question_title": title,
            "answer_excerpt": content,
            "url": target.get("url", "").replace("api.zhihu.com/answers", "www.zhihu.com/question/0/answer")
        })
    
    df = pd.DataFrame(processed_list)
    # 去重
    df.drop_duplicates(subset=["question_title", "upvote_time"], inplace=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"数据清洗完成，共 {len(df)} 条有效动态，已保存到 {output_file}")

if __name__ == "__main__":
    process_raw_data()
