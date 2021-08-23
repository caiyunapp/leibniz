import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="leibniz",
    version="0.1.49",
    author="Mingli Yuan",
    author_email="mingli.yuan@gmail.com",
    description="Leibniz is a package providing facilities to express learnable differential equations based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/caiyunapp/leibniz",
    project_urls={
        'Documentation': 'https://github.com/caiyunapp/leibniz',
        'Source': 'https://github.com/caiyunapp/leibniz',
        'Tracker': 'https://github.com/caiyunapp/leibniz/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'cached_property',
        'torchpwl',
        'torch',
        'numpy',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)

