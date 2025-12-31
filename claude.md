# 项目目标：知乎个人动态爬取与 AI 智能总结助手

## 1. 核心任务描述
开发一个自动化工具，登录并爬取我的知乎动态数据，利用 Claude AI 对内容进行深度分析、分类和兴趣建模。

## 2. 技术栈要求
- **语言**: Python 3.10+
- **爬虫框架**: Playwright (由于知乎反爬严厉，优先使用 Playwright 模拟浏览器行为)
- **数据处理**: Pandas
- **AI 集成**: Anthropic Python SDK (Claude API)
- **存储**: CSV 或 JSONL

## 3. 分阶段路线图 (Roadmap)

### 第一阶段：验证与登录 (Data Acquisition)
- [ ] 初始化 Python 环境，安装 `playwright`, `pandas`。
- [ ] 编写 `crawler.py`：
    - 实现“头戴模式（Headful）”启动，允许用户手动扫码登录知乎。
    - 成功登录后保存 Cookies 到本地 `cookies.json`。
    - 验证是否能成功访问 `https://www.zhihu.com/people/ni-hao-26-89-96` 并获取第一页数据。
- [ ] 应对无限滚动（Infinite Scroll），实现自动下拉抓取历史动态。

### 第二阶段：内容清洗与 AI 总结 (AI Processing)
- [ ] 编写 `analyzer.py`：
    - 提取动态中的关键信息（问题标题、回答内容摘要、点赞时间、收藏夹名称）。
    - 接入 Claude API。
    - 设计 Prompt：针对单条或多条动态进行摘要，提取核心观点。

### 第三阶段：分类与兴趣画像 (Interest Profiling)
- [ ] 构建分类逻辑：
    - 根据内容自动归类（如：数学、算法、Agent 研究、生活、物理等）。
    - 统计兴趣占比，生成 Markdown 格式的年度/月度总结报告。
- [ ] (进阶) 制作简单的可视化图表，展示兴趣偏移曲线。

## 4. 约束与安全
- **隐私**: 不要将 `cookies.json` 和 `API_KEY` 提交到 Git。
- **频率**: 爬取过程中加入随机等待，模拟真实人类行为，避免触发知乎封禁。
- **错误处理**: 遇到验证码时需提醒用户手动介入。