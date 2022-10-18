import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="bias_adjustment",
    version="0.0.1",
    author="Emilio Gozo",
    author_email="emiliogozo@proton.me",
    py_modules=["bias_adjustment", "qm", "qdm", "dqm", "distributions"],
    description="Bias adjustment by quantile mapping",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/emiliogozo/bias_adjustment",
    license="MIT",
    python_requires=">=3.8",
    install_requires=["numpy", "scipy"],
)
