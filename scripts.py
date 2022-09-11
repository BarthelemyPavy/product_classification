# pylava:ignore=C0415
"""Script definitions for `poetry run <command>`."""

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


def download_text_resources() -> None:
    """Download resources from nltk"""
    import nltk
    import sys
    import subprocess

    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("omw-1.4")
    command= ["python", "-m", "spacy", "download", "en"]
    result = subprocess.run(command, capture_output=True, check=False)
    sys.exit(result.returncode)

