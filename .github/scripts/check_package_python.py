# SPDX-FileCopyrightText: 2026 Karan Nisar
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

"""Check the Python compatibility boundary in built distribution metadata."""

from email.parser import BytesParser
from pathlib import Path
import sys
import tarfile
import zipfile


def check_metadata(payload, artifact):
    metadata = BytesParser().parsebytes(payload)
    if metadata["Requires-Python"] != ">=3.11":
        raise ValueError(f"{artifact}: expected Requires-Python >=3.11")
    classifiers = metadata.get_all("Classifier", [])
    if "Programming Language :: Python :: 3.10" in classifiers:
        raise ValueError(f"{artifact}: still advertises Python 3.10")
    print(f"{artifact.name}: Requires-Python {metadata['Requires-Python']}")


def main():
    directory = Path(sys.argv[1])
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Expected exactly one wheel metadata file")
        check_metadata(archive.read(names[0]), wheels[0])
    with tarfile.open(sources[0]) as archive:
        names = [member for member in archive.getmembers() if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")]
        if len(names) != 1:
            raise ValueError("Expected exactly one source distribution metadata file")
        with archive.extractfile(names[0]) as handle:
            check_metadata(handle.read(), sources[0])


if __name__ == "__main__":
    main()
