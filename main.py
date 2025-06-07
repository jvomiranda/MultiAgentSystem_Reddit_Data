from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
import json
import re
import requests
from api_utils import get_reddit_instance, serpapi_search
from format_utils import clean_deepseek_output, strip_markdown_code_fence
from typing import TypedDict, List, Dict, Annotated


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

# ---- Initialize OpenAI and Tavily ----
manager_llm = OllamaLLM(model="deepseek-r1:14b")
planner_llm = OllamaLLM(model="deepseek-r1:14b", temperature=0.3)
cold_planner_llm = OllamaLLM(model="deepseek-r1:14b", temperature=0)
web_llm = OllamaLLM(model="deepseek-r1:14b", temperature=0)
keywords_llm = OllamaLLM(model="deepseek-r1:14b", temperature=0)


# ---- Manager Agent ----
def manager_agent(state: GraphState) -> GraphState:
    if not state.get("plan"):
        # Initial planning phase: summarize instruction into plan
        msg = HumanMessage(content=f"Instruction: {state['user_request']}\nSummarize the goal of this request as a data scraping mission.")
        llm_response = manager_llm.invoke([msg])
        instruction = clean_deepseek_output(llm_response)
        print("\nMANAGER ANALYSIS\n")
        print("INSTRUCTIONS:", instruction)
        return {**state, "instruction": instruction}


# ---- Planner Agent ----
def planner_agent(state: GraphState) -> GraphState:
    if not state.get('web_agent_eval'):
        msg = HumanMessage(content=f"""
        You are a planning agent for a data scraping team. Your task is to analyze the instruction and develop a very BRIEF overall plan of action to generate a well-curated dataset from Reddit posts.
        Your analysis should include the target population, populations to try to rule out (if any),  posssible concerns for this data scraping task (e.g., bias, sarcasm, etc.), specific instruction requests and any other observations you find relevant.
    
        Instruction:
        {state['instruction']}
        """)

        llm_response = planner_llm.invoke([msg])

        plan_text = clean_deepseek_output(llm_response)
        print("\nPLANNER ENVIRONMENT-BASED PLAN:\n", plan_text)

        return {
            **state,
            "plan": plan_text,
            "web_feedback_required": False,
            "web_agent_eval": False
        }
    else:
        msg = HumanMessage(content=f"""
         You are a planning agent coordinator for a Reddit data scraping team working on the current plan:

         {state['plan']}
         
         The web searcher of your team developed a search query that yielded the following results:
         Search query: {state['search_query']}
         Is there a search error? {state.get('search_error', None)}
         Error message: {state.get('error_message', None)}
         Desired number of results: {state.get('desired_search_results', 'Unspecified')}
         Results yielded: {state.get("web_results", None)}
         
         Based on this, does the web searcher require feedback to improve the search query? Do not be very harsh, require improvements only if extremely necessary.
         Output ONLY 'Yes' if improvement is required or 'No' if the search query is good enough.
         
         """)

        print(msg.content)
        llm_response = cold_planner_llm.invoke([msg])

        answer = clean_deepseek_output(llm_response)

        print("\nPLANNER ANSWER:\n", answer)

        if answer.lower() == 'yes':

            msg = HumanMessage(content=f"""
                     You are a planning agent coordinator for a Reddit data scraping team working on the current plan:

                     {state['plan']}

                     The web searcher of your team developed a search query that fail to yield relevant results.
                     Search query: {state['search_query']}
                     Is there a search error? {state.get('search_error', None)}
                     Error message: {state.get('error_message', None)}
                     Desired number of results: {state.get('desired_search_results', 'Unspecified')}
                     Results yielded: {state.get("web_results", None)}

                     Based on this, provide a very BRIEF and objective feedback to the web searcher pointing what's wrong in the search query and how it can be fixed.

                     """)

            llm_response2 = planner_llm.invoke([msg])
            print(msg.content)
            feedback = clean_deepseek_output(llm_response2)

            print("\nPLANNER FEEDBACK TO WEB AGENT:\n", feedback)

            return {**state,
                    "search_feedback": feedback,
                    "web_feedback_required": True,
                    "web_agent_eval": False
            }

        else:



            return {**state,
                    "web_feedback_required": False,
                    "web_agent_eval": True
            }


# ---- Web Based Agent ----
def web_agent(state: GraphState) -> GraphState:
    previous_results = state.get("web_results", [])
    desired_count = 10
    if not state.get("web_feedback_required"):
        msg = HumanMessage(content=f"""
            You are a web search agent for a data scraping team. Your task is to analyze the plan received from your manager and develop a Google-based search query for it to scrape Reddit content that best reflect the plan of action.
    
            Plan:
            {state['plan']}
            
            Output ONLY the search query string and nothing else.
            """)

        llm_response = web_llm.invoke([msg])
        search_query = clean_deepseek_output(llm_response)
        print("\nWEB AGENT SEARCH QUERY:\n", search_query)
        #query = f"site:reddit.com {search_query} subreddit"

        results = serpapi_search(search_query)

        if not results:
            return {
                **state,
                "search_error": True,
                "search_query": search_query,
                "error_message": "Empty or invalid response from SerpApi",
                "web_agent_eval": True,
                "desired_search_results": desired_count
            }


        if "error" in results:
            return {
                **state,
                "search_error": True,
                "search_query": search_query,
                "error_message": results['error'],
                "web_agent_eval": True,
                "desired_search_results": desired_count
            }

        organic_results = results.get('organic_results', [])

        # Extract useful parts of the results — titles and snippets
        cleaned_results = []
        for item in organic_results:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            if "reddit.com" in link:
                cleaned_results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": link
                })

        print(f"WEB AGENT RESULTS: {cleaned_results}")

        # Merge with previous results, removing duplicates
        existing_links = {r['link'] for r in previous_results}
        new_unique_results = [r for r in cleaned_results if r['link'] not in existing_links]
        merged_results = previous_results + new_unique_results

        return {
            **state,
            "search_error": False,
            "search_query": search_query,
            "error_message": None,
            "web_results": merged_results,
            "web_agent_eval": True,
            "desired_search_results": desired_count
        }
    else:
        msg = HumanMessage(content=f"""
                    You are a web search agent for a data scraping team. You received feedback from your manage about a search query you generated that failed to retrieve results.
                    Process the feedback and develop a new Google-based search to scrape Reddit content that is free os mistakes.

                    Plan:
                    {state['plan']}
                    Feedback:
                    Fail search query: {state['search_query']}
                    Feedback: {state['search_feedback']}

                    Output ONLY the search query string and nothing else.
                    """)

        llm_response = web_llm.invoke([msg])
        search_query = clean_deepseek_output(llm_response)
        print("\nWEB AGENT SEARCH UPDATED QUERY:\n", search_query)
        # query = f"site:reddit.com {search_query} subreddit"
        print(msg.content)
        results = serpapi_search(search_query)

        if not results:
            return {
                **state,
                "search_error": True,
                "search_query": search_query,
                "error_message": "Empty or invalid response from SerpApi",
                "web_agent_eval": True,
                "desired_search_results": desired_count
            }


        if "error" in results:
            return {
                **state,
                "search_error": True,
                "search_query": search_query,
                "error_message": results['error'],
                "web_agent_eval": True,
                "desired_search_results": desired_count
            }

        organic_results = results.get('organic_results', [])

        # Extract useful parts of the results — titles and snippets
        cleaned_results = []
        for item in organic_results:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            link = item.get('link', '')
            if "reddit.com" in link:
                cleaned_results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": link
                })

        print(f"WEB AGENT RESULTS: {cleaned_results}")

        # Merge with previous results, removing duplicates
        existing_links = {r['link'] for r in previous_results}
        new_unique_results = [r for r in cleaned_results if r['link'] not in existing_links]
        merged_results = previous_results + new_unique_results

        return {
            **state,
            "search_error": False,
            "search_query": search_query,
            "error_message": None,
            "web_results": merged_results,
            "web_agent_eval": True,
            "desired_search_results": desired_count
        }


# ---- Keyword/Regex Extraction Helper ----
def keywords_agent(state: GraphState) -> GraphState:
    plan = state["plan"]
    web_results = state.get("web_results", [])

    # Build a string representation of the web results to provide context
    web_summary = "\n\n".join(
        f"Title: {item['title']}\nSnippet: {item['snippet']}" for item in web_results
    )

    prompt = f"""
    You are a data analysis agent. Your task is to extract high-quality **keywords** and **regex patterns** based on a web search plan and the actual results retrieved from Reddit via Google.

    --- PLAN ---
    {plan}

    --- WEB RESULTS (titles + snippets) ---
    {web_summary}

    Analyze the plan and content and return:
    - Specific keywords (including multi-word terms)
    - Regex patterns to match relevant Reddit posts
    - Focus on relevance, coverage, and avoiding off-topic data

    Output format (JSON):
    {{
        "keywords": [...],
        "regex": [...]
    }}
    """

    msg = HumanMessage(content=prompt)
    response = clean_deepseek_output(keywords_llm.invoke([msg]))

    try:
        json_str = strip_markdown_code_fence(response)
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("Failed to parse keywords agent response:", e)
        print("Raw response:", response)
        return {
            **state,
            "keywords": [],
            "regex": []
        }

    keywords = data.get("keywords", [])
    regex_patterns = data.get("regex", [])

    print(f"\n📌 KEYWORDS: {keywords}")
    print(f"📎 REGEX: {regex_patterns}")

    return {
        **state,
        "keywords": keywords,
        "regex": regex_patterns
    }



graph = StateGraph(GraphState)

# Core agent
graph.add_node("manager", manager_agent)

# Research Squad
graph.add_node("planner", planner_agent)
graph.add_node("web_agent", web_agent)            # 🆕
graph.add_node("keywords_agent", keywords_agent)

# Set graph flow
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

graph.add_edge("keywords_agent", END)  # ✅ Final handoff
app = graph.compile()


# ---- Run Example ----
if __name__ == "__main__":
    user_request = "Collect English posts from adults with schizophrenia discussing medication side effects, but exclude usernames throwaway123 and baduser456"
    result = app.invoke({"user_request": user_request})

