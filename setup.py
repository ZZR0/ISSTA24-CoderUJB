from setuptools import setup, find_packages

setup(
    name="coderujb",
    version="0.1.0",
    description="A brief description of your package",
    author="ZhengranZeng",
    author_email="zhengranzeng@gmail.com",
    packages=["code_parser", "code_ujb"],
    install_requires=[
        # 'transformers',
        # 'datasets',
        # 'openai',
        # 'fschat',
        # 'psutil',
        # 'torch',
        # 'accelerate',
        # 'chardet',
        # 'javalang'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
