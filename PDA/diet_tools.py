import json
from typing import List, Dict, Any

DB_PATH = "knowledge_base.json"

# --- Internal Helper Functions ---

def _load_db() -> Dict[str, Any]:
    """Helper: Loads the JSON database from the file."""
    try:
        with open(DB_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # If file doesn't exist, return empty structure
        return {"users": {}, "recipes": {}}
    except json.JSONDecodeError:
        # If file is empty or corrupt, return empty structure
        return {"users": {}, "recipes": {}}

def _save_db(data: Dict[str, Any]):
    """Helper: Saves the given dictionary back to the JSON file."""
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)

# --- Tool 1: Register New User ---

def register_new_user(username: str, name: str, allergies: List[str], dietary_goals: List[str], disliked_ingredients: List[str]) -> str:
    """
    Registers a new user in the knowledge base.
    """
    db = _load_db()
    
    # Check if user is already in the 'users' dictionary
    if username in db["users"]:
        return f"User '{username}' already exists. No action taken."
    
    # Add the new user
    db["users"][username] = {
        "name": name,
        "allergies": allergies,
        "dietary_goals": dietary_goals,
        "disliked_ingredients": disliked_ingredients
    }
    
    _save_db(db)
    return f"Successfully registered new user: {name} ({username})."

# --- Tool 2: Get User Details ---

def get_user_details(username: str) -> Dict[str, Any]:
    """
    Retrieves a user's profile (allergies, goals, dislikes).
    """
    db = _load_db()
    
    if username in db["users"]:
        return db["users"][username]
    else:
        # Return a dictionary with an error
        return {"error": f"User '{username}' not found."}

# --- Tool 3: Search Recipes ---

def search_recipes(food_type: str, allergies_list: List[str], dislikes_list: List[str], goals_list: List[str]) -> List[Dict[str, Any]]:
    """
    Searches recipes that match a food type, avoid allergies/dislikes,
    and match dietary goals.
    """
    db = _load_db()
    safe_recipes = []
    
    # Lowercase all allergies and dislikes for easy comparison
    allergies_lower = [a.lower() for a in allergies_list]
    dislikes_lower = [d.lower() for d in dislikes_list]

    for recipe_id, recipe in db["recipes"].items():
        
        # 1. Check for main food type (e.g., "chicken")
        if food_type.lower() not in recipe["main_protein"].lower():
            continue # Skip this recipe

        # 2. Check for Allergies (Critical Fail)
        is_unsafe = False
        for ingredient in recipe["ingredients"]:
            if ingredient.lower() in allergies_lower:
                is_unsafe = True
                break
        if is_unsafe:
            continue # Skip this recipe

        # 3. Check for Dislikes (Soft Fail)
        is_disliked = False
        for ingredient in recipe["ingredients"]:
            if ingredient.lower() in dislikes_lower:
                is_disliked = True
                break
        if is_disliked:
            continue # Skip this recipe

        # 4. Check for Goals (Positive Match)
        matches_goals = True
        for goal in goals_list:
            if goal not in recipe["tags"]:
                # You could add more logic here to check nutrition_per_serving
                matches_goals = False
                break
        if not matches_goals:
            continue # Skip this recipe

        # If it passed all checks, add it to the list
        safe_recipes.append(recipe)

    return safe_recipes