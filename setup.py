# setup.py
from setuptools import setup, find_packages

setup(
    name="nats",                # Your project name
    version="0.1",              # Version (can be anything)
    packages=['src'],   # Automatically find all packages (like src/)
    # package_dir={"": "src"},    # Tell it your code is inside src/
)