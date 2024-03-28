from setuptools import find_packages,setup

setup(
    name="ATS sytem", # name of the package
    version="0.0.1",     #version
    author="karuppasamy", #autor
    author_email="karuppasamy.v@igtsolutions.com", # email
    install_requires=['openai','langchain','streamlit','python-dotenv','PyPDF2'],
    packages=find_packages()

)