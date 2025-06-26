import pytest
from cognivault.store.utils import slugify_title


def test_slugify_title():
    assert slugify_title("What is Cognition?") == "what-is-cognition"
    assert slugify_title("   Hello, World!   ") == "hello-world"
    assert slugify_title("Clean_Title--123") == "clean-title--123"
    assert slugify_title("Symbols & More @#") == "symbols-more-"
    assert slugify_title("multiple    spaces") == "multiple-spaces"
