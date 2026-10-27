"""Validate installed runtime requirements after packaging tools are removed."""

from importlib.metadata import distributions, version, PackageNotFoundError
from packaging.requirements import Requirement


def check():
    errors = []
    for distribution in distributions():
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = version(requirement.name)
            except PackageNotFoundError:
                errors.append(f"{distribution.name} requires missing {requirement}")
                continue
            if installed not in requirement.specifier:
                errors.append(f"{distribution.name} requires {requirement}, found {installed}")
    if errors:
        raise RuntimeError("Invalid packaged runtime:\n" + "\n".join(errors))


if __name__ == "__main__":
    check()
