import re

def clean_deepseek_output(output: str):

    output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL).strip()

    return output

def strip_markdown_code_fence(text: str) -> str:
    """Remove Markdown-style code fences (like ```json) from LLM output."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)