from setuptools import setup, find_packages

setup(
    name="queue_prioritization",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "flask",
        "datetime",
        "plotly",
    ],
    entry_points={
        "console_scripts": [
            "queue-ui=queue_prioritization.web.app:main",
        ],
    },
    python_requires=">=3.6",
    author="Original: Mike; Package: Your Name",
    description="Queue prioritization simulation framework",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 