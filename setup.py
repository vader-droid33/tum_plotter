from setuptools import setup, find_packages

setup(
    name="tum_plotter",
    version="0.1.0",
    description="Visualise and compare SLAM trajectories from TUM files � publication-ready plots with APE statistics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="",
    url="https://github.com/vader-droid33/tum_plotter",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "tum_plotter=tum_plotter.plotter:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
