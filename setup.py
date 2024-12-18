from setuptools import setup


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


def read_file(file):
    with open(file) as f:
        return f.read()


long_description = read_file("README.md")
requirements = read_requirements("requirements.txt")

setup(
    name='bbg_fetch',
    version='1.0.30',
    author='Artur Sepp',
    author_email='artursepp@gmail.com',
    url='https://github.com/ArturSepp/BloombergFetch',
    description='Bloomberg fetching analytics wrapping xbbg package',
    long_description_content_type="text/x-rst",  # If this causes a warning, upgrade your setuptools package
    long_description=long_description,
    license="MIT license",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)