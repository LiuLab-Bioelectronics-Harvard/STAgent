from setuptools import Command, find_packages, setup

__lib_name__ = "STAligner"
__lib_version__ = "1.0.0"
__description__ = "Integrating spatial transcriptomics data across different conditions, technologies, and developmental stages"
__url__ = "https://github.com/zhoux85/STAligner"
__author__ = "Xiang Zhou"
__author_email__ = "xzhou@amss.ac.cn"
__license__ = "MIT"
__keywords__ = ["spatial transcriptomics", "data integration", "Graph attention auto-encoder", "spatial domain", "three-dimensional reconstruction"]
__requires__ = ["requests",]

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ['STAligner'],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
)