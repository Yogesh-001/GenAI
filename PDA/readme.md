# Diet Planning Agent 🍽️

A multi-agent system built with LangGraph that provides personalized diet planning and recipe recommendations based on user profiles, dietary restrictions, and health goals.

## 🌟 Features

- **User Profile Management**: Register new users with dietary preferences, allergies, and goals
- **Smart Recipe Search**: Find recipes that match user requirements while avoiding allergens
- **Safety Validation**: Built-in critic agent to ensure recipe safety and suitability
- **Multi-Agent Architecture**: Planner, Tool Executor, and Critic agents working together
- **Iterative Refinement**: Automatic rethinking loop when initial suggestions don't meet criteria

## 🏗️ Architecture

The system uses a **LangGraph-based multi-agent architecture** with three main components:

### 1. **Planner Agent** 🧠
- Analyzes user requests and current system state
- Makes decisions on what tools to call next
- Proposes recipe recommendations based on available data

### 2. **Tool Node** 🛠️
- Executes various tools for data operations
- Manages user registration and profile retrieval
- Performs recipe searches with filtering logic

### 3. **Critic Agent** 🛡️
- Reviews proposed meal plans for safety
- Checks for allergen conflicts and dietary restrictions
- Triggers refinement loops when plans need improvement

## 📋 Prerequisites

- Python 3.8+
- Azure OpenAI API access
- Required environment variables (see Configuration section)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Build_seggregation/Agent
   ```

2. **Install dependencies**:
   ```bash
   pip install langchain-openai langchain-core langgraph python-dotenv
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   FARM_API_KEY=your_azure_openai_api_key_here
   ```

## 🔧 Configuration

The system is configured to use Azure OpenAI with the following deployment:
- **Model**: GPT-4o-mini (2024-07-18)
- **Temperature**: 0.0 (for deterministic responses)
- **Max Retries**: 3

Update the `deployment` variable in `main_diet.py` if using a different model.

## 📊 Data Structure

The system uses a JSON-based knowledge base (`knowledge_base.json`) with two main sections:

### Users
```json
{
  "users": {
    "username": {
      "name": "User Name",
      "allergies": ["peanuts", "shellfish"],
      "dietary_goals": ["high-protein", "low-carb"],
      "disliked_ingredients": ["olives", "mushrooms"]
    }
  }
}
```

### Recipes
```json
{
  "recipes": {
    "recipe_id": {
      "name": "Recipe Name",
      "main_protein": "chicken",
      "ingredients": ["chicken breast", "broccoli", "olive oil"],
      "tags": ["high-protein", "low-carb"],
      "nutrition_per_serving": {
        "calories": 350,
        "protein": "35g",
        "carbs": "5g",
        "fat": "15g"
      }
    }
  }
}
```

## 🎯 Usage

### Running the Agent

```python
python main_diet.py
```

### Example Interactions

The system handles requests like:
- "I'm Mahesh and I want a high-protein chicken recipe."
- "I'm Tarak Rao and I want a low-carb beef recipe."

### Available Tools

1. **`register_new_user`**: Add new users to the system
2. **`get_user_details`**: Retrieve user profile information  
3. **`search_recipes`**: Find recipes matching criteria

## 🔄 Workflow

1. **User Request**: System receives a natural language diet request
2. **Profile Retrieval**: Planner fetches user profile if available
3. **Recipe Search**: System searches for matching recipes based on:
   - Food type preference
   - Allergy restrictions (hard filter)
   - Disliked ingredients (soft filter)
   - Dietary goals (positive matching)
4. **Safety Check**: Critic reviews proposed recipes
5. **Refinement**: If issues found, system loops back for alternatives
6. **Final Recommendation**: Safe, suitable recipe is presented

## 🛡️ Safety Features

- **Allergen Detection**: Automatic filtering of recipes containing known allergens
- **Ingredient Validation**: Cross-referencing with user's disliked ingredients
- **Goal Alignment**: Ensuring recommendations match dietary objectives
- **Iterative Improvement**: Critic-driven refinement for better suggestions

## 📁 Project Structure

```
Agent/
├── main_diet.py          # Main application with LangGraph workflow
├── diet_tools.py         # Core functionality for user/recipe management
├── knowledge_base.json   # User profiles and recipe database
├── README.md            # This file
└── __pycache__/         # Python cache files
```

## 🔍 Key Components

### State Management
The system uses a `ProjectState` TypedDict to track:
- User requests
- User profiles
- Available recipes
- Final plans
- Message history
- Critique feedback

### Error Handling
- Graceful handling of missing users
- Database file creation if not exists
- API retry mechanisms
- Validation of tool parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## 🆘 Troubleshooting

### Common Issues

1. **"FARM_API_KEY not found"**: Ensure your `.env` file is properly configured
2. **"User not found"**: Register the user first or check the username
3. **Empty recipe results**: Verify recipe database has matching entries
4. **Tool execution errors**: Check parameter types and API connectivity

### Debug Mode

Enable detailed logging by adding print statements or using the streaming output:

```python
for step in app.stream(initial_state, {"recursion_limit": 10}):
    print(step)
```

*Built with  using LangGraph and Azure OpenAI*
