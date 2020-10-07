import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="minihex",
    version="1.0.0",
    author="Sebastian Wallkotter",
    author_email="sebastian@wallkoetter.net",
    description=("The game of Hex implemented for reinforcement learning in"
                 " the OpenAI gym framework. Optimized for rollout speed."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FirefoxMetzger/minihex.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.18.2',
        'gym>=0.17.1'
    ]
)
