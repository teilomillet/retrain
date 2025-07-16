import os
import subprocess
import sys
import importlib.metadata
import shutil
from pathlib import Path

def install_gdown():
    """Installs gdown if it's not already installed."""
    try:
        importlib.metadata.version("gdown")
        print("gdown is already installed.")
    except importlib.metadata.PackageNotFoundError:
        print("gdown not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "uv", "add", "gdown"])
        print("gdown installed successfully.")

def download_and_unzip_databases():
    """Downloads and unzips the Spider2 databases."""
    project_root = Path(__file__).parent.parent
    resource_dir = project_root / "Spider2" / "spider2-lite" / "resource"
    zip_path = resource_dir / "local_sqlite.zip"
    db_dir = resource_dir / "databases" / "spider2-localdb"
    
    # The Google Drive file ID for the Spider2 databases
    file_id = "1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG"
    
    if not resource_dir.exists():
        print(f"Error: The directory {resource_dir} does not exist.")
        print("Please make sure you have the Spider2 dataset in the root of the project.")
        sys.exit(1)

    # Download the file using gdown
    if not zip_path.exists():
        print(f"Downloading databases to {zip_path}...")
        download_command = [
            "gdown",
            f"https://drive.google.com/uc?id={file_id}",
            "-O",
            str(zip_path)
        ]
        try:
            subprocess.run(download_command, check=True)
            print("Download completed.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error downloading the file: {e}")
            print("Please ensure 'gdown' is installed and working correctly.")
            sys.exit(1)
    else:
        print("Database zip file already exists. Skipping download.")

    # Remove old database directory and recreate it
    if db_dir.exists():
        print(f"Removing existing database directory: {db_dir}")
        shutil.rmtree(db_dir)
    print(f"Creating database directory: {db_dir}")
    db_dir.mkdir(parents=True, exist_ok=True)

    # Unzip the databases
    print(f"Unzipping {zip_path} to {db_dir}...")
    unzip_command = ["unzip", str(zip_path), "-d", str(db_dir)]
    try:
        subprocess.run(unzip_command, check=True)
        print("Unzipping completed successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error unzipping the file: {e}")
        print("Please ensure 'unzip' is installed and available in your PATH.")
        sys.exit(1)


def start_mcp_server():
    """Starts the MCP server with a sample database."""
    project_root = Path(__file__).parent.parent
    db_path = project_root / "Spider2/spider2-lite/resource/databases/spider2-localdb/chinook.sqlite"

    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        print("Please run the download and unzip function first.")
        sys.exit(1)

    db_url = f"sqlite:///{db_path.resolve()}"
    
    print(f"Starting MCP server with DB_URL: {db_url}")

    env = os.environ.copy()
    env["DB_URL"] = db_url

    server_command = ["uvx", "mcp-alchemy", "start", "--port", "8008"]
    
    try:
        subprocess.run(server_command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Failed to start mcp-alchemy server: {e}")
    except FileNotFoundError:
        print("Error: 'uvx' command not found.")
        print("Please make sure you have the correct environment activated.")

if __name__ == "__main__":
    install_gdown()
    download_and_unzip_databases()
    start_mcp_server() 