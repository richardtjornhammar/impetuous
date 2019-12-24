import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "impetuous-gfa",
    version = "0.12.0",
    author = "Richard Tjörnhammar",
    author_email = "richard.tjornhammar@gmail.com",
    description = "Impetuous Quantification, Enrichment and Group Variation Analysis",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/richardtjornhammar/impetuous",
    packages = setuptools.find_packages('src'),
    package_dir = {'impetuous':'src/impetuous','quantification':'src/quantification','convert':'src/convert','pathways':'src/pathways','clustering':'src/clustering','hierarchal':'src/hierarchal'},
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
