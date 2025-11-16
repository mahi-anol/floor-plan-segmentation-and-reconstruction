from setuptools import find_packages,setup

# Read requirement.txt
with open('requirements.txt') as file:
    required=file.read().splitlines()

# Read README.md for long description
with open('README.md','r',encoding='utf-8') as file:
    long_description=file.read()

setup(
    name="multi unit floorplan recognition and reconstruction",
    version="0.1.0",
    author="Mahi Sarwar Anol",
    author_email="anol.mahi@gmail.com",
    description="Replication of the paper Multi Unit floorplan recognition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mahi-anol/floor-plan-segmentation-and-reconstruction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Nasir Syntax Solution LTD.",
        "Programing Language :: Python >=3.12.7",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=required,
    extras_require={
        'dev':[
            'pytest>=7.1.1',
            'pytest-cov>=2.12.1',
        ],
    },
)