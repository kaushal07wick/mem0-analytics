from setuptools import setup, find_packages

setup(
    name="mem0-analytics",
    version="0.1.0",
    author="Kaushal",
    description="Lightweight local analytics and observability layer for Mem0-based systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kaushal07wick/mem0-analytics",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    include_package_data=True,
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.25.0",
        "psutil>=5.9.0",
        "rich>=13.0.0",
        "mem0ai>=0.1.0",
    ],
    entry_points={
        "console_scripts": [
            # launches dashboard with background aggregator
            "mem0-dashboard=mem0_analytics.dashboard:run_dashboard",
            # starts only the continuous aggregator (optional)
            "mem0-aggregate=mem0_analytics.aggregate:run_continuous",
            # manual analytics hook tester (optional)
            "mem0-analytics-track=mem0_analytics.analytics:_autopatch",
        ],
    },
)
