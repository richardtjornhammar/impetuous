import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="impetuous-gfa",
    version="0.1.0",
    author="Richard Tjörnhammar",
    author_email="richard.tjornhammar@gmail.com",
    description="Impetuous Group Factor Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richardtjornhammar/impetuous",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
        "Domain :: Group Factor Analysis",
    ],
)
