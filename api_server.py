import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent  # 改这里：从langchain.agents导入，不是langgraph
from langchain_core.messages import HumanMessage
from duckduckgo_search import DDGS
from fastapi import FastAPI
from pydantic import BaseModel

# 消除所有警告（放最前面
warnings.filterwarnings("ignore")

# 加载配置
load_dotenv('agent.env')

# 初始化LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return f"计算结果: {eval(expression)}"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息"""
    try:
        results = DDGS().text(query, max_results=3)
        summary = "\n\n".join([f"标题: {r['title']}\n内容: {r['body'][:200]}" for r in results])
        return f"搜索结果:\n{summary}"
    except Exception as e:
        return f"搜索失败: {str(e)}"

# 创建Agent
tools = [calculator, web_search]
agent = create_agent(llm, tools=tools)  # 改这里：create_agent，不是create_react_agent

# 创建FastAPI应用
app = FastAPI(title="Agent API", version="1.0")

# 请求格式
class QuestionRequest(BaseModel):
    question: str

# API接口
@app.post("/ask")
async def ask_agent(request: QuestionRequest):
    """调用Agent回答问题"""
    response = agent.invoke({
        "messages": [HumanMessage(content=request.question)]
    })
    
    final_answer = response["messages"][-1].content
    
    thoughts = []
    for msg in response["messages"]:
        if msg.type == "ai" and msg.tool_calls:
            thoughts.append({
                "tool": msg.tool_calls[0]["name"],
                "args": msg.tool_calls[0]["args"]
            })
        elif msg.type == "tool":
            thoughts.append({
                "result": msg.content[:100] + "..."
            })
    
    return {
        "question": request.question,
        "answer": final_answer,
        "thought_process": thoughts
    }

@app.get("/")
async def root():
    return {"message": "Agent API is running!", "endpoints": ["/ask"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)