from cognivault.store.utils import slugify_title
from typing import Any


def test_slugify_title() -> None:
    assert slugify_title("What is Cognition?") == "what-is-cognition"
    assert slugify_title("   Hello, World!   ") == "hello-world"
    assert slugify_title("Clean_Title--123") == "clean-title--123"
    assert slugify_title("Symbols & More @#") == "symbols-more-"
    assert slugify_title("multiple    spaces") == "multiple-spaces"
