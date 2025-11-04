import os
import json,operator
from typing import TypedDict, List, Dict, Any, Literal, Annotated
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

import diet_tools # Import our custom tools file

# --- 0. Set your API Key ---
load_dotenv()
farm_api_key = os.environ.get("FARM_API_KEY")
if not farm_api_key:
    raise ValueError("FARM_API_KEY not found in environment variables. Please set it in your .env file.")

deployment = "askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18"
default_headers = {
    "genaiplatform-farm-subscription-key": farm_api_key
}

# --- 1. Decorate Our Custom Functions as Tools ---
# (Your tool definitions are perfect, no changes needed)
@tool
def register_new_user(username: str, name: str, allergies: List[str], dietary_goals: List[str], disliked_ingredients: List[str]) -> str:
    """Registers a new user in the knowledge base."""
    return diet_tools.register_new_user(username, name, allergies, dietary_goals, disliked_ingredients)

@tool
def get_user_details(username: str) -> Dict[str, Any]:
    """Retrieves a user's profile (allergies, goals, dislikes)."""
    return diet_tools.get_user_details(username)

@tool
def search_recipes(food_type: str, allergies_list: List[str], dislikes_list: List[str], goals_list: List[str]) -> List[Dict[str, Any]]:
    """Searches recipes that match a food type, avoid allergies/dislikes, and match goals."""
    return diet_tools.search_recipes(food_type, allergies_list, dislikes_list, goals_list)

# --- 2. Configure LLM & Tools ---
llm = AzureChatOpenAI(
    azure_deployment=deployment,
    default_headers=default_headers,
    temperature=0.0, # Set to 0.0 for deterministic logic
    max_retries=3
)
tools = [register_new_user, get_user_details, search_recipes]

# 🧠 The Planner LLM (with tools)
planner_llm = llm.bind_tools(tools)

# 🛡️ The Critic LLM (no tools needed)
critic_llm = llm

# --- 3. Define the Agent State ---
# (Your ProjectState is perfect, no changes)
class ProjectState(TypedDict):
    user_request: str
    user_profile: Dict[str, Any]
    available_recipes: List[Dict[str, Any]]
    final_plan: Dict[str, Any]
    # messages: List[BaseMessage]
    messages: Annotated[List[BaseMessage], operator.add]
    critique: str


# --- 4. Define the Graph Nodes (The "Agents") ---

### ✅ FIXED PLANNER AGENT ###
def planner_agent_node(state: ProjectState) -> Dict[str, Any]:
    """
    This is the 'Planner' brain. It decides what to do based on the *current* state.
    """
    print("--- 🧠 (Planner) ---")
    
    # Get the current state
    user_profile = state.get('user_profile')
    available_recipes = state.get('available_recipes', [])
    critique = state.get('critique', 'None')
    
    # Build a *dynamic* prompt based on the state
    prompt_lines = [
        "You are a helpful diet planning assistant. Your job is to create a single, safe recipe plan for a user by calling tools in a specific, logical order.",
        "\n--- CURRENT STATE ---",
        f"User Request: {state['user_request']}",
        f"User Profile: {user_profile or 'Not loaded yet'}",
        f"Available Recipes: {len(available_recipes)} recipes found",
        f"Previous Critique: {critique}",
        "\n--- YOUR TASK ---"
    ]
    
    # This state-machine logic guides the LLM to the *correct* next step
    if not user_profile:
        prompt_lines.append("Your *only* task right now is to get the user's profile. Call the `get_user_details` tool with the username from the request.")
    
    elif "error" in str(user_profile):
        # State update must have failed or user not found
        prompt_lines.append("The user profile shows an error or could not be found. Respond with a human-readable error message: 'User not found. Please register first.' DO NOT call any tools.")
    
    elif not available_recipes:
        prompt_lines.append("You have the user profile, but no recipes. Your *only* task is to find recipes. Call the `search_recipes` tool. Use the user's profile and the original request to fill in all arguments (food_type, allergies_list, dislikes_list, goals_list).")
    
    elif critique != 'None' and critique != 'good':
        prompt_lines.append(f"Your last proposal was critiqued: '{critique}'. You MUST propose a *different* recipe from the list of available recipes. Respond with a human-readable message proposing the new recipe.")
    
    else:
        prompt_lines.append("You have a user profile and a list of safe recipes. Your task is to propose ONE recipe to the user. Respond with a human-readable message. DO NOT call any tools.")

    # Combine all messages
    full_prompt = "\n".join(prompt_lines)
    
    # We pass the *full history* so it has context, plus our new dynamic prompt
    messages_for_llm = state["messages"] + [HumanMessage(content=full_prompt)]
    
    ai_response = planner_llm.invoke(messages_for_llm)
    
    # We return the AI response to be *appended* to the message list
    return {"messages": [ai_response]}

### ✅ FIXED TOOL NODE ###
# This node replaces both of your previous tool_node functions
# It doesn't use ToolNode or ToolExecutor, just calls the functions directly.

# Create a dictionary to map tool names to the actual functions
tool_map = {tool.name: tool for tool in tools}

def tool_node(state: ProjectState) -> Dict[str, Any]:
    """This node executes the tools called by the Planner and updates state."""
    print("--- 🛠️ (Tool Node) ---")
    
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        print("Error: Planner did not call a tool.")
        return {}

    tool_messages = []
    new_state_updates = {} # This will hold our state changes

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name not in tool_map:
            response_content = f"Error: Tool '{tool_name}' not found."
        else:
            tool_function = tool_map[tool_name]
            try:
                # Call the tool function directly
                response_content = tool_function.invoke(tool_args)
                
                # Update state based on which tool was called
                if tool_name == "get_user_details":
                    new_state_updates["user_profile"] = response_content
                elif tool_name == "search_recipes":
                    new_state_updates["available_recipes"] = response_content
                elif tool_name == "register_new_user":
                    print(f"User registration result: {response_content}") # Just print this

            except Exception as e:
                response_content = f"Error executing tool {tool_name}: {e}"
        
        # Create the ToolMessage for the history
        tool_messages.append(ToolMessage(content=str(response_content), tool_call_id=tool_call["id"]))

    # Return the state updates *and* the tool messages
    new_state_updates["messages"] = tool_messages
    return new_state_updates


### ✅ FIXED CRITIC NODE ###
# Your critic logic was good, but this version is safer
# It pulls the proposed plan from the AI's *last message*
def critic_agent_node(state: ProjectState) -> Dict[str, Any]:
    """This is the 'Critic' brain. It checks the final plan."""
    print("--- 🛡️ (Critic) ---")
    
    # The planner's proposal is the last message in the history
    planner_proposal_message = state["messages"][-1].content
    user_profile = state.get('user_profile', {})
    
    if not user_profile:
         return {"critique": "CRITICAL_ERROR: Cannot critique without a user profile.", "final_plan": None}

    # Now, let the critic check this proposal
    critic_prompt = f"""
    You are a strict diet critic.
    User Profile: {user_profile}
    Proposed Plan (text): "{planner_proposal_message}"
    
    Review the *user profile* (allergies, dislikes) and check if the *proposed plan text* mentions any unsafe or disliked ingredients.
    - If the plan seems to mention an ALLERGY (e.g., 'Peanut' in plan, 'Peanuts' in profile), respond with: 'CRITIQUE: This plan is UNSAFE, it mentions a potential allergen.'
    - If the plan seems to mention a DISLIKED ingredient (e.g., 'Olives'), respond with: 'CRITIQUE: This plan is not ideal, it contains a disliked ingredient.'
    - If the plan text looks safe and good, respond ONLY with 'good'.
    """
    
    critique_response = critic_llm.invoke(critic_prompt).content
    print(f"Critic's Response: {critique_response}")
    
    if critique_response.lower().strip() == "good":
        # Plan is good!
        # We set `final_plan` to the *text* of the proposal
        return {"final_plan": {"plan_text": planner_proposal_message}, "critique": "good"}
    else:
        # Plan is bad. Send the critique.
        return {"final_plan": None, "critique": critique_response}


# --- 5. Define the Conditional Edges (The Flowchart Logic) ---

### ✅ FIXED ROUTER ###
def should_route_from_planner(state: ProjectState) -> Literal["call_tool", "critic", "__end__"]:
    """After the planner, decide where to go."""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        # Planner wants to use a tool
        return "call_tool"
    
    # Check if user profile indicates an error
    user_profile = state.get("user_profile")
    if user_profile and "error" in str(user_profile):
        # User not found, end the process
        return "__end__"
    
    # If we have recipes and the planner gave a text response (no tool call)
    available_recipes = state.get("available_recipes", [])
    if len(available_recipes) > 0:
        # Planner has info, time for the critic to check
        return "critic"
    
    # If the planner gave a text response (like "user not found")
    # or we have no recipes, end.
    return "__end__"

def should_route_from_critic(state: ProjectState) -> Literal["planner_agent_node", "__end__"]:
    """After the critic, decide whether to loop or end."""
    if state.get("critique", "").lower().strip() == "good":
        # Plan is approved!
        return "__end__"
    else:
        # Plan is flawed, loop back to the planner
        print("--- 🔁 (Rethink Loop) ---")
        return "planner_agent_node"


# --- 6. Assemble the Graph ---
from langgraph.graph import StateGraph, END

workflow = StateGraph(ProjectState)

# Add the nodes
workflow.add_node("planner_agent_node", planner_agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("critic_agent_node", critic_agent_node)

# Set the starting point
workflow.set_entry_point("planner_agent_node")

# Add the edges
workflow.add_edge("tool_node", "planner_agent_node") # Tools always go back to planner

workflow.add_conditional_edges(
    "planner_agent_node",
    should_route_from_planner,
    {
        "call_tool": "tool_node",
        "critic": "critic_agent_node",
        "__end__": END
    }
)

workflow.add_conditional_edges(
    "critic_agent_node",
    should_route_from_critic,
    {
        "planner_agent_node": "planner_agent_node", # The "rethink" loop
        "__end__": END
    }
)

# Compile the graph
app = workflow.compile()

# --- 7. Run the Application ---

print("🚀 **Starting the End-to-End Diet Agent (LangGraph)...**\n")
print("--- Test Case 1: 'alex_b' (should have a rethink loop) ---")

# We will stream the output, printing each step's state
request = "I'm Mahesh and I want a high-protein chicken recipe."
initial_state = {
    "user_request": request,
    "user_profile": None, # Start as None
    "available_recipes": [], # Start as empty
    "final_plan": None,
    "messages": [HumanMessage(content=request)],
    "critique": ""
}

# Use .stream() for live output
for step in app.stream(initial_state, {"recursion_limit": 10}):
    print(step)
    print("--------------------------------\n")

print("\n--- Test Case 2: 'tarak' (gluten-free user) ---")
request_2 = "I'm Tarak Rao and I want a low-carb beef recipe."
initial_state_2 = {
    "user_request": request_2,
    "user_profile": None,
    "available_recipes": [],
    "final_plan": None,
    "messages": [HumanMessage(content=request_2)],
    "critique": ""
}
# Use .invoke() for the final result
final_state = app.invoke(initial_state_2, {"recursion_limit": 10})
print("--- Final State for sam_g ---")
print(final_state)