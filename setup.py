"""Installation procedure."""

from setuptools import setup, find_packages  # type: ignore
from pathlib import Path

reqs_dir = Path("./requirements")


def read_requirements(filename: str):
    """Read a requirements file into a list of individual requirements."""
    requirements_file = reqs_dir / filename
    if requirements_file.is_file():
        requirements_list = requirements_file.read_text().splitlines()
        requirements_list_ = []
        for r in requirements_list:
            if r.strip().startswith("#") or r.strip() == "":
                continue
            package, version = r.split("==")
            major_version = version.split(".")[0]
            requirements_list_.append(f"{package}=={major_version}.*")
        return requirements_list_
    else:
        return []


requirements_base = read_requirements("base.txt")
requirements_test = read_requirements("test.txt")


setup(
    name="trigram-tokenizer",
    url="https://github.com/Aleph-Alpha/trigrams", 
    author="Samuel Weinbach",
    author_email="requests@aleph-alpha-ip.ai",
    install_requires=[requirements_base],
    tests_require=[],
    extras_require={
        "test": requirements_test,
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.1.0",
    license="Open Aleph License",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    entry_points="""""",
    package_data={
        # If any package contains *.json or *.typed
        "": ["*.json", "*.typed"],
    },
    include_package_data=True,
)
