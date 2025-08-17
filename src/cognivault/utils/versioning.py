import subprocess


def get_git_version(default: str = "unknown") -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "describe", "--tags", "--dirty"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return default
