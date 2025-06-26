import re


def slugify_title(title: str) -> str:
    """
    Convert a string into a slug suitable for filenames.
    Removes special characters, lowercases, and replaces spaces with hyphens.

    Args:
        title (str): The input title string.

    Returns:
        str: A slugified version of the title.
    """
    title = title.lower().strip()
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"[\s_]+", "-", title)
    return title
