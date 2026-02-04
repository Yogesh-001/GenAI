# import os
# file_path = 'C:/Users/UYW1KOR/React/Sample.py'
# if os.path.isfile(file_path):
#     # Analyze the file if it exists
#     print("File exists, analyzing...")
#     # Use FileCodeAnalyzer or another tool to analyze the file
# else:
#     print("File does not exist.")
import subprocess
import os
import git
class GitPusher:
    def push_changes(self, repo_dir, commit_message):
        try:
            subprocess.run(["cd", repo_dir], check=True)

            subprocess.run(["git", "add", "."], check=True)

            subprocess.run(["git", "commit", "-m", commit_message], check=True)

            subprocess.run(["git", "push"], check=True)

        except subprocess.CalledProcessError as e:
            return f"Error: {e}"

        return "Changes pushed to GitHub successfully."
    


git_pusher = GitPusher()

# Update repo_dir to point to the Git repository directory (containing .git folder)
repo_dir = r"C:\Users\UYW1KOR\React\.git"

if not os.path.exists(repo_dir):
  print(f"Error: Git repository directory '{repo_dir}' not found.")
  exit()
else:
  print("Git repository path exists")

result = git_pusher.push_changes(repo_dir, commit_message="Updated code")

print(result)


