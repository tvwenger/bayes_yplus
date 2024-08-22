from setuptools import find_packages, setup
import versioneer

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="bayes_yplus",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A Bayesian Model of Radio Recombination Line Emission",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.11",
    license="GNU General Public License v3 (GPLv3)",
    url="https://github.com/tvwenger/bayes_yplus",
)
