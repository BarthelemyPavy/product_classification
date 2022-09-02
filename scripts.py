# pylava:ignore=C0415
"""Script definitions for `poetry run <command>`."""
import os
import zipfile
import gdown


PACKAGE_NAME = "product_classification"


def test() -> None:
    """Start the project unit tests."""
    import pytest

    pytest.main()


def fmt() -> None:
    """Format the whole project with the autoformatter (black)."""
    import subprocess

    from halo import Halo

    spinner = Halo(
        text="> Running auto-format on the whole project",
        spinner="arc",
        placement="right",
    )
    spinner.start()

    subprocess.run(["black", "--config", "lintconfig.toml", PACKAGE_NAME], check=False)
    spinner.succeed()


def lint() -> None:
    """Start the linter on the module with the linter to find out if the linter is happy or not >:(."""
    import sys
    import subprocess

    from halo import Halo

    tests: Dict[Any, Any] = {  # type:ignore
        "pylava": {
            "command": ["pylava", PACKAGE_NAME],
            "succeeded": lambda result: result.returncode == 0,
        },
        "mypy": {
            "command": ["mypy", PACKAGE_NAME],
            "succeeded": lambda result: result.returncode == 0,
        },
        "black": {
            "command": ["black", "--diff", PACKAGE_NAME],
            "succeeded": lambda result: result.returncode == 0,
        },
    }

    status = 0
    print("> Linting the project..")
    for name, params in tests.items():
        spinner = Halo(text=f">> Performing check using {name}", spinner="arc", placement="right")
        spinner.start()

        result = subprocess.run(params["command"], capture_output=True, check=False)
        passed = params["succeeded"](result)

        if not passed:
            spinner.fail()

            print(result.stdout.decode("utf-8"), end="")
            print(result.stderr.decode("utf-8"), end="")

            status = result.returncode
        else:
            spinner.succeed()

    sys.exit(status)


def download_data() -> None:
    """Download and unzip project data"""

    url = "https://drive.google.com/u/0/uc?id=1CUcfl3JX8TNYABn2JRIPQozT0oqdqqOy&export=download"
    dest_path = "data"
    zip_name = "data.zip"

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    gdown.download(url=url, output=dest_path + "/" + zip_name, quiet=False, fuzzy=True)

    with zipfile.ZipFile(dest_path + "/" + zip_name, 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    os.remove(dest_path + "/" + zip_name)


def download_nltk_resources() -> None:
    """Download resources from nltk"""
    import nltk

    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("omw-1.4")
