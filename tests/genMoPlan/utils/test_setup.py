import pytest
from genMoPlan.utils.setup import recursive_update

class TestRecursiveUpdate:
    def test_simple_update(self):
        """Test basic update of top-level keys."""
        base = {'a': 1, 'b': 2}
        update = {'b': 3, 'c': 4}
        expected = {'a': 1, 'b': 3, 'c': 4}
        assert recursive_update(base, update) == expected

    def test_nested_update(self):
        """Test updating a key in a nested dictionary."""
        base = {'a': 1, 'b': {'x': 10, 'y': 20}}
        update = {'b': {'y': 30, 'z': 40}}
        expected = {'a': 1, 'b': {'x': 10, 'y': 30, 'z': 40}}
        assert recursive_update(base, update) == expected

    def test_deeper_nesting(self):
        """Test with multiple levels of nesting."""
        base = {'a': {'b': {'c': 1}}, 'd': 2}
        update = {'a': {'b': {'c': 2, 'd': 3}}}
        expected = {'a': {'b': {'c': 2, 'd': 3}}, 'd': 2}
        assert recursive_update(base, update) == expected

    def test_add_new_key(self):
        """Test adding a new key at the top level."""
        base = {'a': 1}
        update = {'b': 2}
        expected = {'a': 1, 'b': 2}
        assert recursive_update(base, update) == expected

    def test_add_new_nested_key(self):
        """Test adding a new key inside a nested dictionary."""
        base = {'a': {'b': 1}}
        update = {'a': {'c': 2}}
        expected = {'a': {'b': 1, 'c': 2}}
        assert recursive_update(base, update) == expected

    def test_add_new_nested_dict(self):
        """Test adding a completely new nested dictionary."""
        base = {'a': 1}
        update = {'b': {'x': 10, 'y': 20}}
        expected = {'a': 1, 'b': {'x': 10, 'y': 20}}
        assert recursive_update(base, update) == expected

    def test_replace_non_dict_with_dict(self):
        """Test replacing a non-dictionary value with a dictionary raises ValueError."""
        base = {'a': 1}
        update = {'a': {'b': 2}}
        with pytest.raises(ValueError, match=r"cannot overwrite non-dict with dict for key 'a'"):
            recursive_update(base, update)

    def test_replace_dict_with_non_dict(self):
        """Test replacing a dictionary with a non-dictionary value."""
        base = {'a': {'b': 1}}
        update = {'a': 2}
        expected = {'a': 2}
        assert recursive_update(base, update) == expected

    def test_no_change_with_empty_update(self):
        """Test that an empty update dictionary doesn't change the base."""
        base = {'a': 1, 'b': {'x': 10}}
        update = {}
        expected = {'a': 1, 'b': {'x': 10}}
        assert recursive_update(base, update) == expected

    def test_update_with_empty_dict(self):
        """Test that updating with an empty dictionary value has no effect."""
        base = {'a': {'b': 1}}
        update = {'a': {}}
        expected = {'a': {'b': 1}}
        assert recursive_update(base, update) == expected 