import asyncio
import json
import os
from playwright.async_api import async_playwright


class ZhihuCrawler:
    def __init__(self, user_url):
        self.user_url = user_url
        self.cookies_path = "cookies.json"
        self.browser = None
        self.context = None
        self.page = None

    async def init_browser(self, headless=True):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        context_args = {"user_agent": ua}
        
        # 修改此处：使用 storage_state 加载 cookies
        if os.path.exists(self.cookies_path):
            with open(self.cookies_path, 'r') as f:
                cookies = json.load(f)
            context_args["storage_state"] = {"cookies": cookies}

        self.context = await self.browser.new_context(**context_args)
        self.page = await self.context.new_page()


    async def login(self):
        """手动登录并保存 cookies"""
        print("正在启动浏览器进行登录...")
        await self.init_browser(headless=False)
        await self.page.goto("https://www.zhihu.com/signin")
        print("请在浏览器中完成扫码登录...")
        
        # 等待登录成功（通过检查是否跳转到首页或个人页）
        try:
            await self.page.wait_for_url("https://www.zhihu.com/", timeout=60000)
            print("登录成功！")
            cookies = await self.context.cookies()
            with open(self.cookies_path, 'w') as f:
                json.dump(cookies, f)
            print(f"Cookies 已保存到 {self.cookies_path}")
        except Exception as e:
            print(f"登录超时或失败: {e}")
        finally:
            await self.close()

    async def crawl_activities(self, count=50):
        """通过拦截 API 请求抓取动态"""
        if not os.path.exists(self.cookies_path):
            print("未找到 cookies，请先运行登录程序。")
            return

        await self.init_browser(headless=False)
        activities = []

        # 监听响应
        async def handle_response(response):
            # 放宽匹配条件，并打印所有匹配到的 URL 方便调试
            if "activities" in response.url and "api" in response.url:
                try:
                    # 确保响应状态是 200
                    if response.status == 200:
                        data = await response.json()
                        print(f"捕获到活动响应: {response.url}")
                        print(f"响应数据示例: {str(data)[:100]}")  # 打印部分响应数据
                        if "data" in data:
                            for item in data["data"]:
                                activities.append(item)
                                # 打印更详细的信息
                                target = item.get('target', {})
                                title = target.get('title') or target.get('excerpt') or "无标题内容"
                                print(f"捕获到动态: {str(title)[:30]}")
                except Exception as e:
                    # print(f"解析 JSON 出错: {e}")
                    pass

        self.page.on("response", handle_response)

        print(f"正在访问用户主页: {self.user_url}")
        try:
            # 增加 wait_until 参数，确保页面基本加载完成
            await self.page.goto(self.user_url, wait_until="domcontentloaded")
            # 给页面一点缓冲时间处理重定向
            await asyncio.sleep(5) 
        except Exception as e:
            print(f"访问页面出错: {e}")
        
        last_height = 0
        stop_crawling = False
        while not stop_crawling:
            try:
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(3)
                
                # 检查最后一条数据的时间
                if activities:
                    last_item_time = activities[-1].get("created_time", 0)
                    # 1735689600 是 2025-01-01 00:00:00 的时间戳
                    if last_item_time < 1735689600:
                        print("已到达 2025 年之前的数据，停止爬取。")
                        stop_crawling = True
            except Exception as e:
                print(f"滚动时发生异常（可能页面在跳转）: {e}")
                await asyncio.sleep(2)
                continue
            
            new_height = await self.page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            if len(activities) >= count:
                break

        # 保存原始数据
        with open("raw_activities.json", "w", encoding="utf-8") as f:
            json.dump(activities, f, ensure_ascii=False, indent=2)
        
        print(f"抓取完成，共捕获 {len(activities)} 条原始动态数据。")
        await self.close()
        return activities

    async def close(self):
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'pw'):
            await self.pw.stop()

async def main():
    # 示例用户 URL
    user_url = "https://www.zhihu.com/people/ni-hao-26-89-96"
    crawler = ZhihuCrawler(user_url)
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "login":
        await crawler.login()
    else:
        if not os.path.exists("cookies.json"):
            print("未发现 cookies.json，请先运行: python crawler.py login")
        else:
            await crawler.crawl_activities(count=3000)

if __name__ == "__main__":
    asyncio.run(main())
