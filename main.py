import sys

from cali import cali
from mlp import mlp

PROJECTS = {
    "mlp": mlp.go,
    "cali": cali.go,
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROJECTS:
        print(f"Usage: python main.py <{'|'.join(PROJECTS.keys())}>")
        sys.exit(1)

    PROJECTS[sys.argv[1]]()


if __name__ == "__main__":
    main()
