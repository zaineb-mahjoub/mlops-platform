import json
from groq import Groq

AGENT_MODEL = "qwen/qwen3.6-27b"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_document",
            "description": "Analyze the currently active client requirement document: extract its structured specification, predict its project category, and estimate development effort in person-days. Use this when the user asks to analyze, classify, or estimate effort for the document.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_document",
            "description": "Answer a specific question about the currently active document's content (e.g. what database, what integrations, what deadline). Use this for any question about specific details rather than category or effort.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "The user's question about the document"}},
                "required": ["question"],
            },
        },
    },
]

def _dispatch(name: str, args: dict, analyze_fn, ask_fn, user_message: str):
    if name == "analyze_document":
        return analyze_fn()
    elif name == "ask_document":
        return ask_fn(args.get("question", user_message))
    return {"error": f"Unknown tool: {name}"}

def run_agent(user_message: str, analyze_fn, ask_fn) -> dict:
    client = Groq()
    completion = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": "You are an orchestrator for a project-analysis platform. Decide which single tool best serves the user's request, then call it."},
            {"role": "user", "content": user_message},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
        max_tokens=200,
        reasoning_effort="none",
    )
    message = completion.choices[0].message
    if not message.tool_calls:
        return {"tool_used": None, "result": message.content}
    call = message.tool_calls[0]
    name = call.function.name
    args = json.loads(call.function.arguments) if call.function.arguments else {}
    return {"tool_used": name, "result": _dispatch(name, args, analyze_fn, ask_fn, user_message)}

