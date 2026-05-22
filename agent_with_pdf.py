import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from duckduckgo_search import DDGS
from pypdf import PdfReader  # 新增

warnings.filterwarnings("ignore")
load_dotenv('agent.env')

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 工具1：计算器
@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return f"计算结果: {eval(expression)}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 工具2：网页搜索
@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息"""
    try:
        results = DDGS().text(query, max_results=3)
        summary = "\n\n".join([f"标题: {r['title']}\n内容: {r['body'][:200]}" for r in results])
        return f"搜索结果:\n{summary}"
    except Exception as e:
        return f"搜索失败: {str(e)}"

# 工具3：读PDF（新增）
@tool
def read_pdf(file_path: str, question: str) -> str:
    """读取PDF文件内容，回答关于该PDF的问题
    
    Args:
        file_path: PDF文件的路径，比如"论文.pdf"
        question: 关于PDF的问题，比如"这篇论文的摘要是什么？"
        
    Returns:
        从PDF中提取的相关内容
    """
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages[:5]:  # 先读前5页
            text += page.extract_text() + "\n\n"
        
        # 把PDF内容和问题一起给LLM回答
        prompt = f"""
        请根据以下PDF内容回答问题。如果PDF中没有答案，请明确说"PDF中未找到相关信息"。
        
        PDF内容（前5页）：
        {text[:3000]}
        
        问题：{question}
        """
        
        response = llm.invoke(prompt)
        return f"PDF分析结果：\n{response.content}"
        
    except Exception as e:
        return f"读取PDF失败: {str(e)}"

# 现在有3个工具了！
tools = [calculator, web_search, read_pdf]
agent = create_agent(llm, tools=tools)

# 对话记忆
class AgentWithMemory:
    def __init__(self):
        self.chat_history = []
    
    def ask(self, question: str):
        messages = self.chat_history + [HumanMessage(content=question)]
        response = agent.invoke({"messages": messages})
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["messages"][-1].content))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        return response["messages"][-1].content

# 测试
if __name__ == "__main__":
    agent_mem = AgentWithMemory()
    
    print("=" * 60)
    print("测试1：数学计算")
    print(agent_mem.ask("12345乘以67890等于多少？"))
    
    print("\n" + "=" * 60)
    print("测试2：网页搜索")
    print(agent_mem.ask("2025年清华大学计算机系录取分数线是多少？"))
    
    print("\n" + "=" * 60)
    print("测试3：你可以找一个PDF放项目文件夹里，然后问：")
    print('agent_mem.ask("帮我读一下论文.pdf，摘要是什么？")')