import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from duckduckgo_search import DDGS

warnings.filterwarnings("ignore")
load_dotenv('agent.env')

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

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

tools = [calculator, web_search]
agent = create_agent(llm, tools=tools)

# ========== 新增：对话记忆 ==========
class AgentWithMemory:
    def __init__(self):
        self.chat_history = []  # 保存历史对话
    
    def ask(self, question: str):
        # 把历史对话 + 当前问题一起发给Agent
        messages = self.chat_history + [HumanMessage(content=question)]
        
        response = agent.invoke({"messages": messages})
        
        # 把这次对话加入记忆
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["messages"][-1].content))
        
        # 只保留最近10轮对话，避免上下文太长
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        return response["messages"][-1].content

# ========== 测试 ==========
if __name__ == "__main__":
    agent_mem = AgentWithMemory()
    
    print("第一轮：")
    print(agent_mem.ask("我叫楷哥"))
    
    print("\n第二轮：")
    print(agent_mem.ask("我叫什么？"))  # Agent应该记得！
    
    print("\n第三轮：")
    print(agent_mem.ask("2的10次方是多少？"))
    
    print("\n第四轮：")
    print(agent_mem.ask("再乘以100等于多少？"))  # Agent应该记得上一轮的结果！