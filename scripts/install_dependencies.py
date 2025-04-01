"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script automates the installation of required Python libraries and generates a `requirements.txt` file
for the project. It ensures that all necessary dependencies are installed and provides feedback on the process.

Functions:
    install_library(lib: str) -> None:
        Installs a single Python library using pip.

    create_requirements_file(libraries: list) -> None:
        Creates a `requirements.txt` file with the list of libraries.

    main() -> None:
        Main function to install libraries and generate the requirements file.
"""
import subprocess

# List of libraries with their purposes
LIBRARIES = [
    "pandas",         # Data manipulation and analysis
    "numpy",          # Mathematical operations
    "scikit-learn",   # Machine learning algorithms
    "surprise",       # Recommendation systems
    "matplotlib",     # Plotting and visualization
    "seaborn",        # Advanced visualizations
    "pytest"          # Automated testing
]


def install_library(lib: str) -> None:
    """
    Installs a single Python library using pip.

    :param lib: The name of the library to install.
    """
    try:
        subprocess.check_call(["pip", "install", lib])
        print(f"Library '{lib}' installed successfully!")
    except subprocess.CalledProcessError:
        print(f"Error installing the library '{lib}'.")


def create_requirements_file(libraries: list) -> None:
    """
    Creates a `requirements.txt` file with the list of libraries.

    :param libraries: List of library names to include in the requirements file.
    """
    try:
        with open("requirements.txt", "w") as req_file:
            for lib in libraries:
                req_file.write(f"{lib}\n")
        print("File 'requirements.txt' created successfully!")
    except Exception as e:
        print(f"Error creating the 'requirements.txt' file: {e}")


def main() -> None:
    """
    Main function to install libraries and generate the requirements file.
    """
    print("Starting the installation of required libraries...")
    for lib in LIBRARIES:
        install_library(lib)

    print("Generating the 'requirements.txt' file...")
    create_requirements_file(LIBRARIES)
    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()