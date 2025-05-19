#!/bin/bash
# Script to ensure essential development tools, uv, and git configuration are set.

echo "--- Checking and installing essential packages ---"

# Update package list
sudo apt-get update -y

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install cmake if not present
if command_exists cmake; then
    echo "cmake is already installed."
else
    echo "cmake not found. Installing cmake..."
    sudo apt-get install -y cmake
fi

# Install ninja-build if not present
if command_exists ninja; then # ninja-build usually provides 'ninja' command
    echo "ninja-build (ninja) is already installed."
else
    echo "ninja-build (ninja) not found. Installing ninja-build..."
    sudo apt-get install -y ninja-build
fi

# Install git if not present
if command_exists git; then
    echo "git is already installed."
else
    echo "git not found. Installing git..."
    sudo apt-get install -y git
fi

# Install curl and wget if not present (needed for uv installer)
if ! command_exists curl; then
    echo "curl not found. Installing curl..."
    sudo apt-get install -y curl
fi
if ! command_exists wget; then
    echo "wget not found. Installing wget..."
    sudo apt-get install -y wget
fi


echo ""
echo "--- Checking and installing uv (Python package and environment manager) ---"
if command_exists uv; then
    echo "uv is already installed."
    uv --version
else
    echo "uv not found. Installing uv using standalone installer..."
    if command_exists curl; then
        echo "Using curl to download and run uv installer..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command_exists wget; then
        echo "Using wget to download and run uv installer..."
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "ERROR: Neither curl nor wget is available. Cannot install uv automatically."
        echo "Please install curl or wget, or install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    fi

    # Source .profile or .bashrc to make uv available in the current script session if PATH was modified by installer
    # This is a common practice but might not always work depending on shell and installer specifics.
    # The installer itself usually instructs to open a new terminal or source the profile.
    echo "Attempting to source shell profile to make uv available..."
    # SHELL_PROFILE=""
    # if [ -f "$HOME/.bash_profile" ]; then
    #     SHELL_PROFILE="$HOME/.bash_profile"
    # elif [ -f "$HOME/.bashrc" ]; then
    #     SHELL_PROFILE="$HOME/.bashrc"
    # elif [ -f "$HOME/.profile" ]; then
    #     SHELL_PROFILE="$HOME/.profile"
    # elif [ -f "$HOME/.zshrc" ]; then # For zsh users
    #     SHELL_PROFILE="$HOME/.zshrc"
    # fi

    # if [ -n "$SHELL_PROFILE" ]; then
    #     echo "Sourcing $SHELL_PROFILE"
    #     # shellcheck source=/dev/null
    #     source "$SHELL_PROFILE"
    # else
    #     echo "Could not determine shell profile to source. You might need to open a new terminal or source your profile manually for uv to be in PATH."
    # fi
    
    # Re-check if uv is now available
    if command_exists uv; then
        echo "uv installed successfully."
        uv --version
    else
        echo "uv installation might have completed, but 'uv' command is not immediately available in PATH for this script."
        echo "Please open a new terminal or source your shell profile (e.g., source ~/.bashrc, source ~/.profile, or source ~/.zshrc)."
    fi
fi


echo ""
echo "--- Checking git global configuration ---"

# Check git user.email
GIT_USER_EMAIL=$(git config --global user.email)
if [ -z "$GIT_USER_EMAIL" ]; then
    echo "Git global user.email is not set."
    read -r -p "Enter your git user.email (default: teilomillet@gmail.com): " INPUT_EMAIL
    GIT_USER_EMAIL_TO_SET="${INPUT_EMAIL:-teilomillet@gmail.com}"
    git config --global user.email "$GIT_USER_EMAIL_TO_SET"
    echo "Git global user.email set to: $(git config --global user.email)"
else
    echo "Git global user.email is already set to: $GIT_USER_EMAIL"
fi

# Check git user.name
GIT_USER_NAME=$(git config --global user.name)
if [ -z "$GIT_USER_NAME" ]; then
    echo "Git global user.name is not set."
    read -r -p "Enter your git user.name (default: teilomillet): " INPUT_NAME
    GIT_USER_NAME_TO_SET="${INPUT_NAME:-teilomillet}"
    git config --global user.name "$GIT_USER_NAME_TO_SET"
    echo "Git global user.name set to: $(git config --global user.name)"
else
    echo "Git global user.name is already set to: $GIT_USER_NAME"
fi

echo ""
echo "Setup script finished."
echo "Please ensure you have the necessary permissions (e.g., sudo) if prompted for password during installations."
echo "If uv was just installed, you might need to open a new terminal or source your shell profile (e.g., source ~/.bashrc) for the 'uv' command to be available everywhere."
echo "You might need to make this script executable: chmod +x setup_dev_env.sh" 