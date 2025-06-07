from langgraph.graph import StateGraph, END
from typing import Any, Dict, List, TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
import json
import re
import time
import logging
import uuid
from api_utils import get_reddit_instance, serpapi_search
from format_utils import clean_deepseek_output, strip_markdown_code_fence

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# ---- Shared State ----
class GraphState(TypedDict, total=False):
    user_request: Annotated[str, "input"]
    instruction: str
    plan: str
    plan_webAgent: str
    search_query: str
    search_error: bool
    error_message: str
    web_feedback_required: bool
    search_feedback: str
    desired_search_results: int
    web_agent_eval: bool
    web_results: List[Dict[str, str]]
    subreddits: List[str]
    keywords: List[str]
    regex: List[str]
    praw_code: str
    dataset: list[dict]
    messages: Dict[str, List[dict]]
    cache: Dict[str, any]
    message_log: Dict[str, List[str]]


# ---- OHCacheManager ----
class OHCacheManager:
    def __init__(self, state: Dict):
        self.state = state
        self.state.setdefault("messages", {})
        self.state.setdefault("cache", {})
        self.state.setdefault("message_log", {})

    def format_message(self, sender: str, content: str) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "sender": sender,
            "content": content
        }

    def send(self, sender: str, receivers: List[str], content: str):
        message = self.format_message(sender, content)
        logging.info(f"[SEND] {sender} → {receivers}: {content}")
        for receiver in receivers:
            self.state["messages"].setdefault(receiver, []).append(message)

    def receive(self, receiver: str) -> List[dict]:
        seen_ids = set(self.state["message_log"].get(receiver, []))
        all_msgs = self.state["messages"].get(receiver, [])
        new_msgs = [m for m in all_msgs if m["id"] not in seen_ids]
        self.state["message_log"].setdefault(receiver, [])
        self.state["message_log"][receiver].extend([m["id"] for m in new_msgs])
        for m in new_msgs:
            logging.info(f"[RECEIVE] {receiver} ← {m['sender']}: {m['content']}")
        return new_msgs

    def store_artifact(self, data: Any) -> str:
        artifact_id = f"artifact_{uuid.uuid4()}"
        self.state["cache"][artifact_id] = data
        logging.info(f"[CACHE STORE] {artifact_id} with {len(str(data))} characters")
        return artifact_id

    def get_artifact(self, artifact_id: str) -> Any:
        artifact = self.state["cache"].get(artifact_id)
        logging.info(f"[CACHE RETRIEVE] {artifact_id} with {len(str(artifact)) if artifact else 0} characters")
        return artifact

    def get_state(self) -> Dict:
        return self.state


# ---- Initialize LLMs ----
manager_llm = OllamaLLM(model="deepseek-r1:14b")
planner_llm = OllamaLLM(model="deepseek-r1:14b", temperature=0.3)
cold_planner_llm = OllamaLLM(model="deepseek-r1:14b", temperature=0)
web_llm = OllamaLLM(model="deepseek-r1:14b", temperature=0)
keywords_llm = OllamaLLM(model="deepseek-r1:14b", temperature=0)


# ---- Manager Agent ----
def manager_agent(state: GraphState) -> GraphState:
    if not state.get("plan"):
        prompt = f"Instruction: {state['user_request']}\nSummarize the goal of this request as a data scraping mission."
        logging.info(f"[LLM INPUT] manager_llm: {prompt}")
        msg = HumanMessage(content=prompt)
        llm_response = manager_llm.invoke([msg])
        instruction = clean_deepseek_output(llm_response)
        logging.info(f"[LLM OUTPUT] manager_llm: {instruction}")
        return {**state, "instruction": instruction}
    return state


# ---- Planner Agent ----
def planner_agent(state: GraphState) -> GraphState:
    cache = OHCacheManager(state)
    new_msgs = cache.receive("planner")

    # Process feedback messages first
    for msg in new_msgs:
        if "Search query" in msg["content"] or "Search complete" in msg["content"]:
            logging.info(f"[LLM INPUT] cold_planner_llm: {msg['content']}")
            search_results = f"Search query developed:{state['search_query']}\nSearch results: {state['web_results']}\nSearch error? {state['search_error']}"
            prompt_analysis = f"Analyze if this search query is good enough based on {state['plan']}\n{search_results}\nAnswer with ONLY 'Yes' if you think it's good enough and 'No' if you think it's not. Don't be too harsh because your team has just a few search API calls."
            HumanMessage(content=prompt_analysis)
            decision = cold_planner_llm.invoke([prompt_analysis])
            answer = clean_deepseek_output(decision)
            logging.info(f"[LLM OUTPUT] planner_llm: {answer}")
            if answer.lower() == 'yes':
                if "Stored as" in msg["content"]:
                    artifact_id = msg["content"].split("Stored as")[-1].strip()
                    artifact_data = cache.get_artifact(artifact_id)
                    artifact_text = "\n\n".join(
                        f"Title: {item['title']}\nSnippet: {item['snippet']}" for item in artifact_data)
                else:
                    artifact_text = ""

                feedback_prompt = f"""
                The search query may be suboptimal. Improve it based on:

                PLAN:
                {state['plan']}

                QUERY:
                {state['search_query']}

                RESULTS:
                {artifact_text}

                Your task is to return advice for the web search agent on how to improve the query, considering the mission goal and these results.
                Output specific, useful feedback to refine the query.
                """

                feedback_msg = HumanMessage(content=feedback_prompt)
                logging.info(f"[LLM INPUT] planner_llm: {feedback_msg.content}")
                feedback = clean_deepseek_output(planner_llm.invoke([feedback_msg]))
                logging.info(f"[LLM OUTPUT] planner_llm: {feedback}")
                cache.send("planner", ["web_agent"], f"Feedback: {feedback}")
                return {**cache.get_state(), "search_feedback": feedback, "web_feedback_required": True, "web_agent_eval": False}
            else:
                return {**cache.get_state(), "web_feedback_required": False, "web_agent_eval": True}

    # Only generate initial plan if it's missing
    if not state.get("plan"):
        prompt = f"Instruction: {state['instruction']}\nSummarize briefly (no more than a paragraph) the goal of this request as a data scraping mission."
        logging.info(f"[LLM INPUT] planner_llm: {prompt}")
        msg = HumanMessage(content=prompt)
        llm_response = planner_llm.invoke([msg])
        plan = clean_deepseek_output(llm_response)
        logging.info(f"[LLM OUTPUT] planner_llm: {plan}")
        return {**cache.get_state(), "plan": plan}

    if not state.get('web_agent_eval'):
        prompt = f"""
        You are a planning agent for a data scraping team. Your task is to analyze the instruction and develop a very BRIEF overall plan of action to generate a well-curated dataset from Reddit posts.
        Instruction:
        {state['plan']}
        Create a one-sentence plan based on the instruction for your web agent to develop relevant search queries.
        """
        logging.info(f"[LLM INPUT] planner_llm: {prompt.strip()}")
        msg = HumanMessage(content=prompt)
        llm_response = planner_llm.invoke([msg])
        plan_text = clean_deepseek_output(llm_response)
        logging.info(f"[LLM OUTPUT] planner_llm: {plan_text}")
        cache.send("planner", ["web_agent", "manager"], f"Generated plan: {plan_text}")
        return {**cache.get_state(), "plan": plan_text, "web_feedback_required": False, "web_agent_eval": False}

    return cache.get_state()


# ---- Web Agent ----

def web_agent(state: GraphState) -> GraphState:
    cache = OHCacheManager(state)
    messages = cache.receive("web_agent")

    feedback_text = None
    for msg in messages:
        if "Feedback:" in msg["content"]:
            feedback_text = msg["content"].split("Feedback:")[-1].strip()
            logging.info(f"[WEB_AGENT] Received feedback: {feedback_text}")

    # Construct the search prompt
    if feedback_text:
        prompt = f"""
        You are a web search agent for a data scraping team.
        The current mission is: {state['plan']}
        The planning agent provided feedback to improve the search query: {feedback_text}
        Using both the mission and the feedback, write a Google search query for Reddit posts.
        Output ONLY the search query and nothing more.
        """
    else:
        prompt = f"""
        You are a web search agent for a data scraping team.
        Generate a Google search query based on the plan:
        {state['plan']}
        Output ONLY the search query and nothing more.
        """

    logging.info(f"[LLM INPUT] web_llm: {prompt.strip()}")
    msg = HumanMessage(content=prompt)
    llm_response = web_llm.invoke([msg])
    search_query = clean_deepseek_output(llm_response)
    logging.info(f"[LLM OUTPUT] web_llm: {search_query}")


    # Perform the web search
    results = serpapi_search(search_query)
    if not results or "error" in results:
        return {
            **cache.get_state(),
            "search_error": True,
            "search_query": search_query,
            "error_message": results,
            "web_agent_eval": True
        }

    # Extract Reddit-specific results
    organic_results = results.get('organic_results', [])
    cleaned_results = [
        {"title": i.get("title"), "snippet": i.get("snippet"), "link": i.get("link")}
        for i in organic_results
    ]

    artifact_id = cache.store_artifact(cleaned_results)
    cache.send("web_agent", ["planner", "keywords_agent"], f"Search complete. Stored as {artifact_id}")

    return {
        **cache.get_state(),
        "web_results": cleaned_results,
        "search_query": search_query,
        "search_error": False,
        "web_agent_eval": True
    }

# ---- Keywords Agent ----
def keywords_agent(state: GraphState) -> GraphState:
    cache = OHCacheManager(state)
    messages = cache.receive("keywords_agent")

    artifact_candidates = []

    for msg in messages:
        if "Stored as" in msg["content"]:
            artifact_id = msg["content"].split("Stored as")[-1].strip()
            web_results = cache.get_artifact(artifact_id)
            if isinstance(web_results, list) and web_results:
                artifact_candidates.append((artifact_id, web_results))

    if not artifact_candidates:
        logging.warning("[KEYWORDS_AGENT] No valid artifacts received.")
        return cache.get_state()

    # Pick the artifact with the most entries or characters
    best_artifact_id, best_web_results = max(
        artifact_candidates,
        key=lambda item: len(item[1])  # could also use sum(len(x.get('snippet', '')) for x in item[1])
    )

    web_summary = "\n\n".join(
        f"Title: {item['title']}\nSnippet: {item['snippet']}" for item in best_web_results
    )
    prompt = f"""
    You are a data analysis agent. Extract high-quality keywords and regex patterns.
    PLAN:
    {state['plan']}
    WEB RESULTS:
    {web_summary}
    Output JSON with "keywords" and "regex".
    """
    logging.info(f"[LLM INPUT] keywords_llm: {prompt.strip()[:1000]}...")
    response = clean_deepseek_output(keywords_llm.invoke([HumanMessage(content=prompt)]))
    response_text = clean_deepseek_output(response)
    logging.info(f"[LLM OUTPUT] keywords_llm: {response_text[:1000]}...")
    try:
        json_str = strip_markdown_code_fence(response_text)
        data = json.loads(json_str)
        return {
            **cache.get_state(),
            "keywords": data.get("keywords", []),
            "regex": data.get("regex", [])
        }
    except json.JSONDecodeError:
        return {
            **cache.get_state(),
            "keywords": [],
            "regex": []
        }

    return cache.get_state()


# ---- Graph Setup ----
graph = StateGraph(GraphState)
graph.add_node("manager", manager_agent)
graph.add_node("planner", planner_agent)
graph.add_node("web_agent", web_agent)
graph.add_node("keywords_agent", keywords_agent)

graph.set_entry_point("manager")
graph.add_edge("manager", "planner")
graph.add_conditional_edges(
    source="planner",
    path=lambda state: (
        "web_agent" if not state.get("web_agent_eval") else (
            "web_agent" if state.get("web_feedback_required") else "keywords_agent"
        )
    )
)
graph.add_edge("web_agent", "planner")
graph.add_edge("keywords_agent", END)

app = graph.compile()

user_request = "Collect posts from English speaking users that self-declare a diagnosis of schizophrenia."
result = app.invoke({"user_request": user_request})
