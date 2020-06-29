import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fawkes",
    version="0.0.1",
    author="Shawn Shan",
    author_email="shansixiong@cs.uchicago.edu",
    description="Fawkes protect user privacy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shawn-Shan/fawkes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)