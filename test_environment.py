import sys

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
        min_minor = 6
        max_major = 9
    else:
        raise ValueError(
            "Unrecognized python interpreter: {}".format(REQUIRED_PYTHON))

    if system_major != required_major or not (system_minor >= min_minor
                                              and system_minor < max_major):
        raise TypeError(
            "This project requires Python >= 3.6, < 3.9. Found: Python {}".
            format(sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
