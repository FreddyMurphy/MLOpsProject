import sys


def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor
    required_major = 3
    min_minor = 6
    max_major = 9

    if system_major != required_major or not (system_minor >= min_minor
                                              and system_minor < max_major):
        raise TypeError(
            "This project requires Python >= 3.6, < 3.9. Found: Python {}".
            format(sys.version))
    else:
        print(">>> Development environment passes all tests!")


if __name__ == '__main__':
    main()
