import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from duckduckgo_search import DDGS

warnings.filterwarnings("ignore")
load_dotenv('agent.env')

llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 定义工具
@tool
def web_search(query: str) -> str:
    """搜索互联网获取信息"""
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return "没有找到相关信息"
        summary = "\n\n".join([f"标题: {r['title']}\n内容: {r['body'][:200]}" for r in results])
        return f"搜索结果:\n{summary}"
    except Exception as e:
        return f"搜索失败: {str(e)}"

tools = [web_search]

# ========== LangGraph核心开始 ==========

# 1. 定义状态（Agent的记忆）
class AgentState(TypedDict):
    messages: List[str]
    search_count: int  # 搜索次数，用于限制重试
    question: str

# 2. 定义节点（工作流的每一步）

# 节点1：判断是否需要搜索
def should_search(state: AgentState):
    """LLM判断这个问题是否需要搜索"""
    prompt = f"""
    问题：{state['question']}
    
    判断：这个问题是否需要联网搜索才能准确回答？
    只回答"是"或"否"，不要解释。
    
    是 = 需要实时信息、最新数据、具体事件
    否 = 常识问题、数学计算、逻辑推理
    """
    
    response = llm.invoke(prompt)
    need_search = "是" in response.content
    
    if need_search:
        print("🤔 判断：需要搜索")
        return "search"
    else:
        print("🤔 判断：不需要搜索，直接回答")
        return "answer"

# 节点2：执行搜索
def do_search(state: AgentState):
    """执行搜索"""
    print(f"🔍 正在搜索：{state['question']}")
    
    result = web_search.invoke({"query": state["question"]})
    
    state["messages"].append(f"搜索结果: {result}")
    state["search_count"] += 1
    
    print(f"✅ 搜索完成 (第{state['search_count']}次)")
    return state

# 节点3：判断搜索结果够不够
def check_result(state: AgentState):
    """检查搜索结果质量，不行就重试"""
    if state["search_count"] >= 3:
        print("⚠️ 已达最大搜索次数，停止搜索")
        return "answer"
    
    prompt = f"""
    问题：{state['question']}
    搜索结果：{state['messages'][-1]}
    
    判断：这个搜索结果是否足够回答问题？
    只回答"够"或"不够"，不要解释。
    """
    
    response = llm.invoke(prompt)
    
    if "不够" in response.content and state["search_count"] < 3:
        print("🔄 搜索结果不够，换关键词重试")
        return "search"
    else:
        print("✅ 搜索结果足够，开始回答")
        return "answer"

# 节点4：生成最终回答
def generate_answer(state: AgentState):
    """根据所有信息生成回答"""
    context = "\n".join(state["messages"])
    
    prompt = f"""
    请根据以下信息回答问题。
    
    问题：{state['question']}
    
    参考信息：
    {context}
    """
    
    response = llm.invoke(prompt)
    print("\n" + "=" * 60)
    print("📝 最终回答：")
    print(response.content)
    print("=" * 60)
    
    state["messages"].append(f"最终回答: {response.content}")
    return state

# 3. 构建工作流图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("search", do_search)
workflow.add_node("answer", generate_answer)

# 添加边
workflow.set_conditional_entry_point(
    should_search,
    {
        "search": "search",
        "answer": "answer"
    }
)

workflow.add_conditional_edges(
    "search",
    check_result,
    {
        "search": "search",
        "answer": "answer"
    }
)

workflow.add_edge("answer", END)

# 4. 编译图
app = workflow.compile()

# ========== 测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("测试1：不需要搜索的问题（常识）")
    print("=" * 60)
    result = app.invoke({
        "messages": [],
        "search_count": 0,
        "question": "水的沸点是多少度？"
    })
    
    print("\n" + "=" * 60)
    print("测试2：需要搜索的问题（实时信息）")
    print("=" * 60)
    result = app.invoke({
        "messages": [],
        "search_count": 0,
        "question": "2025年比特币最新价格是多少？"
    })