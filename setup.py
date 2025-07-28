from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-PROJECT-2",
    version="0.1",
    author="Shahab",
    packages=find_packages(exclude=["data", "data.*"]),
    install_requires = requirements,
)