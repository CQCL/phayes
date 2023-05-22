from setuptools import setup, find_packages

exec(open("phayes/version.py").read())

setup(
    name="phayes",
    author="Sam Duffield",
    author_email="sam.duffield@quantinuum.com",
    url="https://github.com/SamDuffield/bayesian-phase-estimation",
    description="Bayesian phase and amplitude estimation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["jax", "jaxlib", "tensorflow-probability"],
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    include_package_data=True,
    platforms="any",
    version=__version__,
)
