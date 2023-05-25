from distutils.core import setup

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="cloudremoval",
    version="1.0",
    description="Cloud removal from ground-based imaging",
    author="Jay Paul Morgan",
    author_email="jay.morgan@univ-tln.fr",
    url="https://github.com/jaypmorgan/cloud-removal",
    packages=["cloudremoval"],
    package_dir={"cloudremoval": "src/cloudremoval"},
    install_requires=required_packages)
