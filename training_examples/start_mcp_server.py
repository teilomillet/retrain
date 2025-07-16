#!/usr/bin/env python3
"""
A helper script to start the mcp-alchemy server with a robust, absolute path to the database.

This script resolves the absolute path to the target SQLite database and sets the DB_URL
environment variable before launching the mcp-alchemy server. This avoids "file not found"
errors caused by relative paths when running from different directories.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """
    Constructs the absolute path to the database and starts the mcp-alchemy server.
    """
    try:
        # This is the target database ID from the Spider dataset.
        # By changing this value, you can point the server to any other database.
        db_id = "school_scheduling"
        
        # --- Path Construction ---
        # This builds a reliable path to the database.
        # It assumes this script is in `training_examples` and the `database` dir is at the project root.
        project_root = Path(__file__).parent.parent
        db_path = project_root / "database" / db_id / f"{db_id}.sqlite"
        
        if not db_path.exists():
            print(f"Error: Database file not found at expected path: {db_path}", file=sys.stderr)
            print("Please ensure you have downloaded the Spider databases and they are in the 'database' directory.", file=sys.stderr)
            sys.exit(1)
            
        # --- Environment Setup ---
        # Create a copy of the current environment and set the absolute DB_URL.
        # Using an absolute path is crucial for the server to find the database file.
        env = os.environ.copy()
        env["DB_URL"] = f"sqlite:///{db_path.resolve()}"
        
        server_command = ["uvx", "mcp-alchemy"]
        
        print(f"Starting mcp-alchemy server for database: {db_id}")
        print(f"Database URL: {env['DB_URL']}")
        
        # --- Server Execution ---
        # This executes the mcp-alchemy server using the modified environment.
        # The `uvx` command ensures it runs within the correct virtual environment.
        subprocess.run(server_command, env=env, check=True)
        
    except FileNotFoundError:
        print("Error: 'uvx' command not found.", file=sys.stderr)
        print("Please ensure 'uv' is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: The mcp-alchemy server failed to start (exit code {e.returncode}).", file=sys.stderr)
        print("Please check the server logs for more details.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer shutdown requested. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main() 