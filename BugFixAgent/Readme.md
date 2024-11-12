# Code Analyzer and GitHub Pusher

## Overview

This is an Autonomous Agent that analyzes different coding language codes for errors, suggests fixes, and pushes the modified code to a specified GitHub repository. It utilizes advanced language models to provide code suggestions and relies on Git for version control.

## Features

- **Code Analysis**: Automatically analyzes Python code to identify syntax and runtime errors.
- **Code Suggestion**: Uses an AI model to suggest fixes for identified issues and provides corrected code.
- **GitHub Integration**: Pushes the modified code to a specified GitHub repository, making it easy to manage code changes.

## Requirements

- Python 3.x
- Required Python packages:
  - `langchain`
  - `requests`
  - `subprocess`
- Git installed on your machine
- An Azure OpenAI account with access to the API
- A GitHub account with a personal access token

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YourRepoName.git
   cd YourRepoName
