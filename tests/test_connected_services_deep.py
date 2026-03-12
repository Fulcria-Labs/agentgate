"""Deep tests for connected services CRUD — add, remove, list, upsert,
user isolation, ordering, and edge cases."""

import time

import pytest

from src.database import (
    add_connected_service,
    get_connected_services,
    init_db,
    remove_connected_service,
)


class TestAddConnectedService:
    """Service connection creation."""

    @pytest.mark.asyncio
    async def test_add_single_service(self, db):
        await add_connected_service("u1", "github")
        services = await get_connected_services("u1")
        assert len(services) == 1
        assert services[0]["service"] == "github"

    @pytest.mark.asyncio
    async def test_add_multiple_services(self, db):
        for svc in ["github", "slack", "google", "linear", "notion"]:
            await add_connected_service("u1", svc)
        services = await get_connected_services("u1")
        assert len(services) == 5

    @pytest.mark.asyncio
    async def test_connection_id_stored(self, db):
        await add_connected_service("u1", "github", "conn-abc-123")
        services = await get_connected_services("u1")
        assert services[0]["connection_id"] == "conn-abc-123"

    @pytest.mark.asyncio
    async def test_default_connection_id_empty(self, db):
        await add_connected_service("u1", "github")
        services = await get_connected_services("u1")
        assert services[0]["connection_id"] == ""

    @pytest.mark.asyncio
    async def test_connected_at_timestamp_set(self, db):
        before = time.time()
        await add_connected_service("u1", "github")
        services = await get_connected_services("u1")
        assert services[0]["connected_at"] >= before


class TestUpsertBehavior:
    """Re-connecting an existing service updates rather than duplicates."""

    @pytest.mark.asyncio
    async def test_upsert_updates_connection_id(self, db):
        await add_connected_service("u1", "github", "old-conn")
        await add_connected_service("u1", "github", "new-conn")
        services = await get_connected_services("u1")
        assert len(services) == 1
        assert services[0]["connection_id"] == "new-conn"

    @pytest.mark.asyncio
    async def test_upsert_updates_timestamp(self, db):
        await add_connected_service("u1", "github", "conn1")
        first = (await get_connected_services("u1"))[0]["connected_at"]
        # Re-connect
        await add_connected_service("u1", "github", "conn2")
        second = (await get_connected_services("u1"))[0]["connected_at"]
        assert second >= first

    @pytest.mark.asyncio
    async def test_upsert_same_connection_id(self, db):
        await add_connected_service("u1", "github", "same")
        await add_connected_service("u1", "github", "same")
        services = await get_connected_services("u1")
        assert len(services) == 1

    @pytest.mark.asyncio
    async def test_upsert_one_service_doesnt_affect_others(self, db):
        await add_connected_service("u1", "github", "gh-conn")
        await add_connected_service("u1", "slack", "sl-conn")
        # Re-connect github only
        await add_connected_service("u1", "github", "gh-new")
        services = await get_connected_services("u1")
        assert len(services) == 2
        svc_map = {s["service"]: s for s in services}
        assert svc_map["github"]["connection_id"] == "gh-new"
        assert svc_map["slack"]["connection_id"] == "sl-conn"


class TestRemoveConnectedService:
    """Service disconnection."""

    @pytest.mark.asyncio
    async def test_remove_existing_service(self, db):
        await add_connected_service("u1", "github")
        await remove_connected_service("u1", "github")
        services = await get_connected_services("u1")
        assert len(services) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_service_no_error(self, db):
        # Should not raise
        await remove_connected_service("u1", "nonexistent")

    @pytest.mark.asyncio
    async def test_remove_one_preserves_others(self, db):
        await add_connected_service("u1", "github")
        await add_connected_service("u1", "slack")
        await add_connected_service("u1", "google")
        await remove_connected_service("u1", "slack")
        services = await get_connected_services("u1")
        assert len(services) == 2
        svc_names = {s["service"] for s in services}
        assert "slack" not in svc_names
        assert "github" in svc_names
        assert "google" in svc_names

    @pytest.mark.asyncio
    async def test_remove_then_re_add(self, db):
        await add_connected_service("u1", "github", "conn-old")
        await remove_connected_service("u1", "github")
        await add_connected_service("u1", "github", "conn-new")
        services = await get_connected_services("u1")
        assert len(services) == 1
        assert services[0]["connection_id"] == "conn-new"


class TestGetConnectedServices:
    """Service listing behavior."""

    @pytest.mark.asyncio
    async def test_empty_list_for_new_user(self, db):
        services = await get_connected_services("new-user")
        assert services == []

    @pytest.mark.asyncio
    async def test_ordered_by_connected_at_desc(self, db):
        await add_connected_service("u1", "github")
        await add_connected_service("u1", "slack")
        await add_connected_service("u1", "google")
        services = await get_connected_services("u1")
        for i in range(len(services) - 1):
            assert services[i]["connected_at"] >= services[i + 1]["connected_at"]

    @pytest.mark.asyncio
    async def test_service_dict_keys(self, db):
        await add_connected_service("u1", "github", "conn-123")
        services = await get_connected_services("u1")
        s = services[0]
        assert "service" in s
        assert "connected_at" in s
        assert "connection_id" in s


class TestConnectedServicesUserIsolation:
    """Service connections are isolated between users."""

    @pytest.mark.asyncio
    async def test_users_see_only_own_connections(self, db):
        await add_connected_service("u1", "github")
        await add_connected_service("u2", "slack")
        s1 = await get_connected_services("u1")
        s2 = await get_connected_services("u2")
        assert len(s1) == 1
        assert s1[0]["service"] == "github"
        assert len(s2) == 1
        assert s2[0]["service"] == "slack"

    @pytest.mark.asyncio
    async def test_same_service_different_users(self, db):
        await add_connected_service("u1", "github", "u1-conn")
        await add_connected_service("u2", "github", "u2-conn")
        s1 = await get_connected_services("u1")
        s2 = await get_connected_services("u2")
        assert s1[0]["connection_id"] == "u1-conn"
        assert s2[0]["connection_id"] == "u2-conn"

    @pytest.mark.asyncio
    async def test_removing_one_users_service_doesnt_affect_other(self, db):
        await add_connected_service("u1", "github")
        await add_connected_service("u2", "github")
        await remove_connected_service("u1", "github")
        assert len(await get_connected_services("u1")) == 0
        assert len(await get_connected_services("u2")) == 1
