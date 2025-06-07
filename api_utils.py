import praw
import os
import requests

SERPAPI_KEY= ""


def get_reddit_instance():
    # Replace these values with your own Reddit API credentials
    reddit = praw.Reddit(
        client_id="",
        client_secret="",
        user_agent=""
    )
    return reddit



def serpapi_search(query: str, max_results=10) -> list[dict]:

    os.environ["SERPAPI_KEY"] = SERPAPI_KEY
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        raise ValueError("SERPAPI_KEY not set in environment variables.")

    print(f"\n[WebAgent Search] Using SerpAPI for: {query}\n")

    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": max_results
    }
    response = requests.get(url, params=params)

    try:
        data = response.json()
    except Exception as e:
        print("Failed to decode JSON response:", e)
        print("Raw response:", response.text)
        return []

    print("SerpApi raw response JSON:")
    print(data)

    if "error" in data:
        print("SerpApi returned an error:", data["error"])
        return []

    if "organic_results" not in data:
        print("Warning: No organic results found in response.")
        return []

    return data