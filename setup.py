try:
    from setuptools import setup, find_packages
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: setuptools.\n\n"
        "Run packaging commands with this project's environment (it already has setuptools), e.g.:\n"
        "  .\\venv\\python.exe -m pip install -r requirement.txt\n"
        "  .\\venv\\python.exe -m pip install -e .\n"
    ) from e
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='Delaney-Solubility-with-Descriptors-ML',
    version='0.0.2',
    author='Justinfi.info',
    author_email='justinfi.info@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt')
)
