import warnings
warnings.filterwarnings('ignore', message='.*allowed_objects.*')

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS  # 新增：搜索工具

# 1. 加载配置
load_dotenv('agent.env')

# 2. 初始化LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 3. 定义工具

# 工具1：计算器
@tool
def calculator(expression: str) -> str:
    """计算数学表达式，输入格式如"123 * 456"
    
    Args:
        expression: 要计算的数学表达式
        
    Returns:
        计算结果字符串
    """
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 工具2：网页搜索（新增！）
@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息，回答实时问题
    
    Args:
        query: 搜索关键词，比如"2026年5月比特币价格"
        
    Returns:
        搜索结果摘要
    """
    try:
        results = DDGS().text(query, max_results=3)
        summary = "\n\n".join([
            f"标题: {r['title']}\n内容: {r['body'][:200]}" 
            for r in results
        ])
        return f"搜索结果:\n{summary}"
    except Exception as e:
        return f"搜索失败: {str(e)}"

# 4. 创建Agent（现在有两个工具了！）
tools = [calculator, web_search]
agent = create_react_agent(llm, tools=tools)

# 5. 运行Agent
def run_agent(question: str):
    print(f"\n用户问题: {question}\n")
    print("=" * 50)
    
    response = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    
    # 打印完整思考过程
    for msg in response["messages"]:
        if msg.type == "human":
            continue
        elif msg.type == "ai":
            if msg.tool_calls:
                print(f"[AI 思考] 决定调用工具: {msg.tool_calls[0]['name']}")
                print(f"         参数: {msg.tool_calls[0]['args']}")
            else:
                print(f"\n[AI 回答] {msg.content}")
        elif msg.type == "tool":
            print(f"[工具返回] {msg.content}")
    
    print("\n" + "=" * 50)

# 6. 测试
if __name__ == "__main__":
    run_agent("12345乘以67890等于多少？")      # 应该用calculator
    run_agent("2的10次方是多少？")              # 应该用calculator
    run_agent("2026年5月比特币价格是多少？")    # 应该用web_search