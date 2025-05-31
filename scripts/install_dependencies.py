"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script automates the installation of required Python libraries and generates a
`requirements.txt` file for the project.
It ensures that all necessary dependencies are installed and provides feedback on the process.
"""

# Note: If you encounter issues while installing the 'surprise' library, such as a build error,
# it is likely because the Microsoft C++ Build Tools are not installed. To resolve this:
# 1. Download and install the Visual C++ Build Tools from this link:
#    https://visualstudio.microsoft.com/visual-cpp-build-tools/
# 2. During installation, ensure you select the "Desktop development with C++" workload.
# 3. After installation, restart your system and try running the script again.

# Note: If you experience issues while running the 'surprise' library due to a
# NumPy incompatibility, it may be because the current version of NumPy (2.x) is not supported.
# To fix this issue:
# 1. Downgrade NumPy to a version below 2.0 by running:
#    pip install "numpy<2"
# 2. Ensure all dependencies in your project are updated and compatible.
# This step will resolve the incompatibility and allow 'surprise' to function properly.

import subprocess
import os
import sys

# List of libraries with their purposes
LIBRARIES = [
    "pandas",  # Data manipulation and analysis
    "numpy",  # Mathematical operations
    "scikit-learn",  # Machine learning algorithms
    "scikit-surprise",  # Recommendation systems
    "matplotlib",  # Plotting and visualization
    "seaborn",  # Advanced visualizations
    "pytest",  # Automated testing
    "tabulate",
    "textblob",  # Text processing
    "nltk",  # Natural Language Toolkit,
    "googletrans==4.0.0-rc1",  # Google Translate API
    "flask",  # Web framework
]


def create_virtual_environment(env_name: str) -> None:
    """
    Creates a virtual environment for the project.

    :param env_name: The name of the virtual environment directory.
    """
    if not os.path.exists(env_name):
        try:
            subprocess.check_call([sys.executable, "-m", "venv", env_name])
            print(f"Virtual environment '{env_name}' created successfully!")
        except subprocess.CalledProcessError:
            print(f"Error creating the virtual environment '{env_name}'.")
            sys.exit(1)
    else:
        print(f"Virtual environment '{env_name}' already exists.")


def install_library(lib: str, env_name: str) -> None:
    """
    Installs a single Python library in the virtual environment using pip.

    :param lib: The name of the library to install.
    :param env_name: The name of the virtual environment directory.
    """
    pip_path = (
        os.path.join(env_name, "Scripts", "pip")
        if os.name == "nt"
        else os.path.join(env_name, "bin", "pip")
    )
    try:
        subprocess.check_call([pip_path, "install", lib])
        print(f"Library '{lib}' installed successfully!")
    except subprocess.CalledProcessError:
        print(f"Error installing the library '{lib}'.")


def create_requirements_file(env_name: str) -> None:
    """
    Creates a `requirements.txt` file with the list of libraries and their installed versions.

    :param env_name: The name of the virtual environment directory.
    """
    pip_path = (
        os.path.join(env_name, "Scripts", "pip")
        if os.name == "nt"
        else os.path.join(env_name, "bin", "pip")
    )
    try:
        installed_packages = subprocess.check_output([pip_path, "freeze"], text=True)
        with open("requirements.txt", "w", encoding="utf-8") as req_file:
            req_file.write(installed_packages)
        print("File 'requirements.txt' created successfully!")
    except Exception as e:
        print(f"Error creating the 'requirements.txt' file: {e}")


def main() -> None:
    """
    Main function to create a virtual environment, install libraries, and
    generate the requirements file.
    """
    env_name = "venv"

    print("Creating a virtual environment...")
    create_virtual_environment(env_name)

    print("Starting the installation of required libraries...")
    for lib in LIBRARIES:
        install_library(lib, env_name)

    print("Generating the 'requirements.txt' file...")
    create_requirements_file(env_name)

    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
