"""
Unit tests for the Hugging Face collection manager script.

These tests mock all external API calls to test the logic without making real API requests.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest
from huggingface_hub.utils import HfHubHTTPError


# Import the module to test
# Navigate from tests/scripts/ up to repo root, then to scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import manage_hf_collection


class TestSetupApi:
    """Tests for API setup and authentication."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("manage_hf_collection.HfApi")
    def test_setup_api_no_token(self, mock_hf_api):
        """Test successful API setup path without HF_TOKEN (local auth flow)."""
        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "local_user"}
        mock_hf_api.return_value = mock_api

        api = manage_hf_collection.setup_api()

        assert api is not None
        mock_hf_api.assert_called_once_with()
        mock_api.whoami.assert_called_once()

    @patch.dict(os.environ, {"HF_TOKEN": "test_token"})
    @patch("manage_hf_collection.HfApi")
    def test_setup_api_success(self, mock_hf_api):
        """Test successful API setup."""
        mock_api = Mock()
        mock_api.whoami.return_value = {"name": "test_user"}
        mock_hf_api.return_value = mock_api

        api = manage_hf_collection.setup_api()

        assert api is not None
        mock_hf_api.assert_called_once_with(token="test_token")
        mock_api.whoami.assert_called_once()

    @patch.dict(os.environ, {"HF_TOKEN": "invalid_token"})
    @patch("manage_hf_collection.HfApi")
    def test_setup_api_auth_failure(self, mock_hf_api):
        """Test that setup_api exits when authentication fails."""
        mock_api = Mock()
        mock_api.whoami.side_effect = Exception("Auth failed")
        mock_hf_api.return_value = mock_api

        with pytest.raises(SystemExit) as exc_info:
            manage_hf_collection.setup_api()
        assert exc_info.value.code == 1


class TestGetCollectionSpaces:
    """Tests for fetching spaces from the collection."""

    def test_get_collection_spaces_success(self):
        """Test successfully fetching spaces from collection."""
        mock_api = Mock()
        mock_collection = Mock()

        # Create mock items
        mock_item1 = Mock()
        mock_item1.item_type = "space"
        mock_item1.item_id = "owner1/space1"

        mock_item2 = Mock()
        mock_item2.item_type = "space"
        mock_item2.item_id = "owner2/space2"

        mock_item3 = Mock()
        mock_item3.item_type = "model"  # Different type, should be ignored
        mock_item3.item_id = "owner3/model1"

        mock_collection.items = [mock_item1, mock_item2, mock_item3]
        mock_api.get_collection.return_value = mock_collection

        result = manage_hf_collection.get_collection_spaces(
            mock_api, "openenv/environment-hub-test"
        )

        assert len(result) == 2
        assert "owner1/space1" in result
        assert "owner2/space2" in result
        assert "owner3/model1" not in result

    def test_get_collection_spaces_not_found(self):
        """Test handling of collection not found error."""
        mock_api = Mock()
        mock_response = Mock()
        mock_response.status_code = 404
        error = HfHubHTTPError("Not found", response=mock_response)
        mock_api.get_collection.side_effect = error

        with pytest.raises(SystemExit) as exc_info:
            manage_hf_collection.get_collection_spaces(
                mock_api, "openenv/environment-hub-test"
            )
        assert exc_info.value.code == 1

    def test_get_collection_spaces_other_error(self):
        """Test handling of other HTTP errors."""
        mock_api = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        error = HfHubHTTPError("Server error", response=mock_response)
        mock_api.get_collection.side_effect = error

        with pytest.raises(SystemExit) as exc_info:
            manage_hf_collection.get_collection_spaces(
                mock_api, "openenv/environment-hub-test"
            )
        assert exc_info.value.code == 1


class TestDiscoverOpenenvSpaces:
    """Tests for discovering spaces with openenv tag."""

    @patch("manage_hf_collection.list_spaces")
    def test_discover_openenv_spaces_success(self, mock_list_spaces):
        """Test successfully discovering openenv spaces."""
        mock_api = Mock()

        # Create mock space objects
        mock_space1 = Mock()
        mock_space1.id = "owner1/openenv-space1"

        mock_space2 = Mock()
        mock_space2.id = "owner2/openenv-space2"

        mock_list_spaces.return_value = [mock_space1, mock_space2]

        # Mock space_info to return proper SpaceInfo objects
        def mock_space_info(space_id):
            space_info = Mock()
            space_info.sdk = "docker"
            space_info.tags = ["openenv", "environment"]
            return space_info

        mock_api.space_info.side_effect = mock_space_info

        result = manage_hf_collection.discover_openenv_spaces(mock_api, "openenv")

        assert len(result) == 2
        assert "owner1/openenv-space1" in result
        assert "owner2/openenv-space2" in result

        # Verify list_spaces was called with correct parameters
        mock_list_spaces.assert_called_once_with(
            search="openenv", full=False, sort="trending_score", direction=-1
        )

    @patch("manage_hf_collection.list_spaces")
    def test_discover_openenv_spaces_filters_non_docker(self, mock_list_spaces):
        """Test that non-Docker spaces are filtered out."""
        mock_api = Mock()

        # Create mock space objects
        mock_space1 = Mock()
        mock_space1.id = "owner1/openenv-space1"

        mock_space2 = Mock()
        mock_space2.id = "owner2/openenv-space2"

        mock_list_spaces.return_value = [mock_space1, mock_space2]

        # First space is Docker with openenv tag, second is Gradio
        def mock_space_info(space_id):
            space_info = Mock()
            if space_id == "owner1/openenv-space1":
                space_info.sdk = "docker"
                space_info.tags = ["openenv"]
            else:
                space_info.sdk = "gradio"
                space_info.tags = ["openenv"]
            return space_info

        mock_api.space_info.side_effect = mock_space_info

        result = manage_hf_collection.discover_openenv_spaces(mock_api, "openenv")

        # Only Docker space should be returned
        assert len(result) == 1
        assert "owner1/openenv-space1" in result
        assert "owner2/openenv-space2" not in result

    @patch("manage_hf_collection.list_spaces")
    def test_discover_openenv_spaces_filters_missing_tag(self, mock_list_spaces):
        """Test that spaces without openenv tag are filtered out."""
        mock_api = Mock()

        mock_space = Mock()
        mock_space.id = "owner1/some-space"

        mock_list_spaces.return_value = [mock_space]

        # Space is Docker but doesn't have openenv tag
        def mock_space_info(space_id):
            space_info = Mock()
            space_info.sdk = "docker"
            space_info.tags = ["other-tag"]
            return space_info

        mock_api.space_info.side_effect = mock_space_info

        result = manage_hf_collection.discover_openenv_spaces(mock_api, "openenv")

        assert len(result) == 0

    @patch("manage_hf_collection.list_spaces")
    def test_discover_openenv_spaces_empty(self, mock_list_spaces):
        """Test discovering spaces when none exist."""
        mock_api = Mock()
        mock_list_spaces.return_value = []

        result = manage_hf_collection.discover_openenv_spaces(mock_api, "openenv")

        assert len(result) == 0
        assert result == []

    @patch("manage_hf_collection.list_spaces")
    def test_discover_openenv_spaces_handles_space_info_error(self, mock_list_spaces):
        """Test handling of errors when fetching individual space info."""
        mock_api = Mock()

        mock_space1 = Mock()
        mock_space1.id = "owner1/space1"
        mock_space2 = Mock()
        mock_space2.id = "owner2/space2"

        mock_list_spaces.return_value = [mock_space1, mock_space2]

        # First space fails, second succeeds
        def mock_space_info(space_id):
            if space_id == "owner1/space1":
                raise Exception("Space not found")
            space_info = Mock()
            space_info.sdk = "docker"
            space_info.tags = ["openenv"]
            return space_info

        mock_api.space_info.side_effect = mock_space_info

        result = manage_hf_collection.discover_openenv_spaces(mock_api, "openenv")

        # Should continue and return second space
        assert len(result) == 1
        assert "owner2/space2" in result

    @patch("manage_hf_collection.list_spaces")
    def test_discover_openenv_spaces_error(self, mock_list_spaces):
        """Test handling of errors during space discovery."""
        mock_api = Mock()
        mock_list_spaces.side_effect = Exception("API error")

        with pytest.raises(SystemExit) as exc_info:
            manage_hf_collection.discover_openenv_spaces(mock_api, "openenv")
        assert exc_info.value.code == 1


class TestAddSpacesToCollection:
    """Tests for adding spaces to the collection."""

    def test_add_spaces_empty_list(self):
        """Test adding empty list of spaces."""
        mock_api = Mock()

        result = manage_hf_collection.add_spaces_to_collection(
            mock_api,
            "openenv/environment-hub-test",
            [],
            "v2.1.0",
            dry_run=False,
        )

        assert result == 0
        mock_api.add_collection_item.assert_not_called()

    def test_add_spaces_dry_run(self):
        """Test adding spaces in dry-run mode."""
        mock_api = Mock()
        space_ids = ["owner1/space1", "owner2/space2"]

        result = manage_hf_collection.add_spaces_to_collection(
            mock_api,
            "openenv/environment-hub-test",
            space_ids,
            "v2.1.0",
            dry_run=True,
        )

        assert result == 2
        mock_api.add_collection_item.assert_not_called()

    def test_add_spaces_success(self):
        """Test successfully adding spaces."""
        mock_api = Mock()
        space_ids = ["owner1/space1", "owner2/space2"]

        result = manage_hf_collection.add_spaces_to_collection(
            mock_api,
            "openenv/environment-hub-test",
            space_ids,
            "v2.1.0",
            dry_run=False,
        )

        assert result == 2
        assert mock_api.add_collection_item.call_count == 2

        # Verify calls were made with correct parameters
        calls = mock_api.add_collection_item.call_args_list
        assert calls[0][1]["collection_slug"] == "openenv/environment-hub-test"
        assert calls[0][1]["item_id"] == "owner1/space1"
        assert calls[0][1]["item_type"] == "space"
        assert calls[0][1]["note"] == "OpenEnv release 2.1.0"

    def test_add_spaces_duplicate_conflict(self):
        """Test handling of duplicate space (409 conflict)."""
        mock_api = Mock()
        mock_response = Mock()
        mock_response.status_code = 409
        error = HfHubHTTPError("Conflict", response=mock_response)
        mock_api.add_collection_item.side_effect = error

        space_ids = ["owner1/space1"]

        result = manage_hf_collection.add_spaces_to_collection(
            mock_api,
            "openenv/environment-hub-test",
            space_ids,
            "v2.1.0",
            dry_run=False,
        )

        # Should not count as success, but should not crash
        assert result == 0

    def test_add_spaces_partial_failure(self):
        """Test adding spaces with some failures."""
        mock_api = Mock()
        mock_response = Mock()
        mock_response.status_code = 500
        error = HfHubHTTPError("Server error", response=mock_response)

        # First call succeeds, second fails
        mock_api.add_collection_item.side_effect = [None, error]

        space_ids = ["owner1/space1", "owner2/space2"]

        result = manage_hf_collection.add_spaces_to_collection(
            mock_api,
            "openenv/environment-hub-test",
            space_ids,
            "v2.1.0",
            dry_run=False,
        )

        assert result == 1  # Only first one succeeded


class TestRemoveSpacesFromCollection:
    """Tests for collection reconciliation removals."""

    def test_remove_spaces_dry_run(self):
        """Dry-run reconcile should report removals without mutating the API."""
        mock_api = Mock()
        current_items = []

        keep_item = Mock()
        keep_item.item_id = "openenv/repl"
        keep_item.item_object_id = "obj-keep"
        current_items.append(keep_item)

        stale_item = Mock()
        stale_item.item_id = "third-party/example"
        stale_item.item_object_id = "obj-stale"
        current_items.append(stale_item)

        result = manage_hf_collection.remove_spaces_from_collection(
            mock_api,
            "openenv/environment-hub-test",
            current_items=current_items,
            target_space_ids=["openenv/repl"],
            dry_run=True,
        )

        assert result == 1
        mock_api.delete_collection_item.assert_not_called()

    def test_remove_spaces_success(self):
        """Reconcile should delete collection entries that are not in the target set."""
        mock_api = Mock()

        keep_item = Mock()
        keep_item.item_id = "openenv/repl"
        keep_item.item_object_id = "obj-keep"

        stale_item = Mock()
        stale_item.item_id = "third-party/example"
        stale_item.item_object_id = "obj-stale"

        result = manage_hf_collection.remove_spaces_from_collection(
            mock_api,
            "openenv/environment-hub-test",
            current_items=[keep_item, stale_item],
            target_space_ids=["openenv/repl"],
            dry_run=False,
        )

        assert result == 1
        mock_api.delete_collection_item.assert_called_once_with(
            collection_slug="openenv/environment-hub-test",
            item_object_id="obj-stale",
            missing_ok=True,
        )


class TestResolveCollectionSlug:
    """Tests for resolving explicit collection slugs."""

    def test_resolve_collection_slug_canonicalizes_explicit_alias(self):
        """Explicit user-facing slugs should resolve to the canonical Hub slug."""
        mock_api = Mock()
        mock_collection = Mock()
        mock_collection.slug = (
            "openenv/openenv-environment-hub-69a154b16387f8b33a978dd7"
        )
        mock_api.get_collection.return_value = mock_collection

        result = manage_hf_collection.resolve_collection_slug(
            mock_api,
            namespace="openenv",
            title="OpenEnv Environment Hub",
            description="desc",
            explicit_slug="openenv/openenv-environment-hub",
            private=False,
            dry_run=False,
        )

        assert result == "openenv/openenv-environment-hub-69a154b16387f8b33a978dd7"
        mock_api.get_collection.assert_called_once_with(
            "openenv/openenv-environment-hub"
        )
        mock_api.update_collection_metadata.assert_called_once_with(
            collection_slug="openenv/openenv-environment-hub-69a154b16387f8b33a978dd7",
            title="OpenEnv Environment Hub",
            description="desc",
            private=False,
        )

    def test_resolve_collection_slug_falls_back_to_explicit_alias_on_lookup_failure(
        self,
    ):
        """Lookup failures should fall back to the user-provided explicit slug."""
        mock_api = Mock()
        mock_api.get_collection.side_effect = Exception("lookup failed")

        result = manage_hf_collection.resolve_collection_slug(
            mock_api,
            namespace="openenv",
            title="OpenEnv Environment Hub",
            description="desc",
            explicit_slug="openenv/openenv-environment-hub",
            private=False,
            dry_run=False,
        )

        assert result == "openenv/openenv-environment-hub"
        mock_api.update_collection_metadata.assert_called_once_with(
            collection_slug="openenv/openenv-environment-hub",
            title="OpenEnv Environment Hub",
            description="desc",
            private=False,
        )

    def test_resolve_collection_slug_updates_existing_collection_metadata(self):
        """Existing collections should have description/title refreshed on reuse."""
        mock_api = Mock()
        mock_collection = Mock()
        mock_collection.slug = "openenv/openenv-environment-hub-123"
        mock_collection.title = "OpenEnv Environment Hub"
        mock_api.list_collections.return_value = [mock_collection]

        result = manage_hf_collection.resolve_collection_slug(
            mock_api,
            namespace="openenv",
            title="OpenEnv Environment Hub",
            description="validated canonicals",
            explicit_slug=None,
            private=False,
            dry_run=False,
        )

        assert result == "openenv/openenv-environment-hub-123"
        mock_api.update_collection_metadata.assert_called_once_with(
            collection_slug="openenv/openenv-environment-hub-123",
            title="OpenEnv Environment Hub",
            description="validated canonicals",
            private=False,
        )


class TestMain:
    """Tests for the main function."""

    @patch("manage_hf_collection.setup_api")
    @patch("manage_hf_collection.resolve_collection_slug")
    @patch("manage_hf_collection.get_collection_items")
    @patch("manage_hf_collection.discover_canonical_openenv_spaces")
    @patch("manage_hf_collection.add_spaces_to_collection")
    @patch("sys.argv", ["manage_hf_collection.py", "--dry-run"])
    def test_main_dry_run(
        self,
        mock_add_spaces,
        mock_discover,
        mock_get_collection,
        mock_resolve_slug,
        mock_setup_api,
    ):
        """Test main function in dry-run mode."""
        mock_api = Mock()
        mock_setup_api.return_value = mock_api
        mock_resolve_slug.return_value = "openenv/environment-hub-test"
        mock_item = Mock()
        mock_item.item_id = "owner1/space1"
        mock_get_collection.return_value = [mock_item]
        mock_discover.return_value = ["owner1/space1", "owner2/space2"]
        mock_add_spaces.return_value = 1

        manage_hf_collection.main()

        # Verify dry_run=True was passed
        mock_add_spaces.assert_called_once()
        args, kwargs = mock_add_spaces.call_args
        assert kwargs["dry_run"] is True

    @patch("manage_hf_collection.setup_api")
    @patch("manage_hf_collection.resolve_collection_slug")
    @patch("manage_hf_collection.get_collection_items")
    @patch("manage_hf_collection.discover_canonical_openenv_spaces")
    @patch("manage_hf_collection.remove_spaces_from_collection")
    @patch("manage_hf_collection.add_spaces_to_collection")
    @patch("sys.argv", ["manage_hf_collection.py", "--reconcile"])
    def test_main_reconcile_removes_stale_spaces(
        self,
        mock_add_spaces,
        mock_remove_spaces,
        mock_discover,
        mock_get_collection_items,
        mock_resolve_slug,
        mock_setup_api,
    ):
        """Reconcile mode should remove spaces outside the resolved target set."""
        mock_api = Mock()
        mock_setup_api.return_value = mock_api
        mock_resolve_slug.return_value = "openenv/environment-hub-test"

        keep_item = Mock()
        keep_item.item_id = "owner1/space1"
        keep_item.item_object_id = "obj-keep"

        stale_item = Mock()
        stale_item.item_id = "owner2/space2"
        stale_item.item_object_id = "obj-stale"

        mock_get_collection_items.return_value = [keep_item, stale_item]
        mock_discover.return_value = ["owner1/space1"]
        mock_add_spaces.return_value = 0
        mock_remove_spaces.return_value = 1

        manage_hf_collection.main()

        mock_remove_spaces.assert_called_once()
        _, kwargs = mock_remove_spaces.call_args
        assert kwargs["collection_slug"] == "openenv/environment-hub-test"
        assert kwargs["target_space_ids"] == ["owner1/space1"]
        assert kwargs["current_items"] == [keep_item, stale_item]

    @patch("manage_hf_collection.setup_api")
    @patch("manage_hf_collection.resolve_collection_slug")
    @patch("manage_hf_collection.get_collection_items")
    @patch("manage_hf_collection.discover_canonical_openenv_spaces")
    @patch("manage_hf_collection.add_spaces_to_collection")
    @patch("sys.argv", ["manage_hf_collection.py"])
    def test_main_finds_new_spaces(
        self,
        mock_add_spaces,
        mock_discover,
        mock_get_collection,
        mock_resolve_slug,
        mock_setup_api,
    ):
        """Test main function correctly identifies new spaces."""
        mock_api = Mock()
        mock_setup_api.return_value = mock_api
        mock_resolve_slug.return_value = "openenv/environment-hub-test"
        item1 = Mock()
        item1.item_id = "owner1/space1"
        item2 = Mock()
        item2.item_id = "owner2/space2"
        mock_get_collection.return_value = [item1, item2]
        mock_discover.return_value = ["owner1/space1", "owner2/space2", "owner3/space3"]
        mock_add_spaces.return_value = 1

        manage_hf_collection.main()

        # Verify only new space is added
        mock_add_spaces.assert_called_once()
        _, kwargs = mock_add_spaces.call_args
        assert kwargs["space_ids"] == ["owner3/space3"]  # Only the new space
        assert kwargs["collection_slug"] == "openenv/environment-hub-test"

    @patch("manage_hf_collection.setup_api")
    @patch("manage_hf_collection.resolve_collection_slug")
    @patch("manage_hf_collection.get_collection_items")
    @patch("manage_hf_collection.discover_canonical_openenv_spaces")
    @patch("manage_hf_collection.add_spaces_to_collection")
    @patch("sys.argv", ["manage_hf_collection.py", "--verbose"])
    def test_main_verbose(
        self,
        mock_add_spaces,
        mock_discover,
        mock_get_collection,
        mock_resolve_slug,
        mock_setup_api,
    ):
        """Test main function with verbose logging."""
        mock_api = Mock()
        mock_setup_api.return_value = mock_api
        mock_resolve_slug.return_value = "openenv/environment-hub-test"
        mock_get_collection.return_value = []
        mock_discover.return_value = []
        mock_add_spaces.return_value = 0

        # Should not raise any exceptions
        manage_hf_collection.main()

        mock_setup_api.assert_called_once()

    @patch("manage_hf_collection.setup_api")
    @patch("manage_hf_collection.resolve_collection_slug")
    @patch("manage_hf_collection.get_collection_items")
    @patch("manage_hf_collection.discover_openenv_spaces")
    @patch("manage_hf_collection.discover_canonical_openenv_spaces")
    @patch("manage_hf_collection.add_spaces_to_collection")
    @patch("sys.argv", ["manage_hf_collection.py", "--global-scope", "tagged"])
    def test_main_tagged_scope_uses_tag_discovery(
        self,
        mock_add_spaces,
        mock_discover_canonical,
        mock_discover_tagged,
        mock_get_collection,
        mock_resolve_slug,
        mock_setup_api,
    ):
        """Tagged scope should keep the old broad-discovery behavior when requested."""
        mock_api = Mock()
        mock_setup_api.return_value = mock_api
        mock_resolve_slug.return_value = "openenv/environment-hub-test"
        mock_get_collection.return_value = []
        mock_discover_tagged.return_value = ["owner1/space1"]
        mock_discover_canonical.return_value = ["openenv/repl"]
        mock_add_spaces.return_value = 1

        manage_hf_collection.main()

        mock_discover_tagged.assert_called_once_with(mock_api, "openenv")
        mock_discover_canonical.assert_not_called()


class TestIdempotency:
    """Tests to verify idempotent behavior."""

    @patch("manage_hf_collection.setup_api")
    @patch("manage_hf_collection.resolve_collection_slug")
    @patch("manage_hf_collection.get_collection_items")
    @patch("manage_hf_collection.discover_canonical_openenv_spaces")
    @patch("manage_hf_collection.add_spaces_to_collection")
    @patch("sys.argv", ["manage_hf_collection.py"])
    def test_no_new_spaces_does_nothing(
        self,
        mock_add_spaces,
        mock_discover,
        mock_get_collection,
        mock_resolve_slug,
        mock_setup_api,
    ):
        """Test that running with no new spaces makes no changes."""
        mock_api = Mock()
        mock_setup_api.return_value = mock_api
        mock_resolve_slug.return_value = "openenv/environment-hub-test"
        item1 = Mock()
        item1.item_id = "owner1/space1"
        item2 = Mock()
        item2.item_id = "owner2/space2"
        mock_get_collection.return_value = [item1, item2]
        mock_discover.return_value = ["owner1/space1", "owner2/space2"]
        mock_add_spaces.return_value = 0

        manage_hf_collection.main()

        # Verify add_spaces was called with empty list
        mock_add_spaces.assert_called_once()
        _, kwargs = mock_add_spaces.call_args
        assert kwargs["space_ids"] == []  # No new spaces


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
