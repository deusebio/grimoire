"""Tests for the GitHub service layer."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from grimoire.config import (
    GitHubConfig,
    GrimoireConfig,
    StalenessConfig,
    StaticRepoSource,
    TeamRepoSource,
)
from grimoire.database import (
    CachedIssue,
    CachedPullRequest,
    CachedRepository,
    CachedWorkflowStatus,
    CheckResultRecord,
    create_tables,
    get_engine,
)
from grimoire.github.client import GitHubClient
from grimoire.github.service import (
    WORKFLOW_ACTIVE_DAYS,
    fetch_repository_stats,
    load_stats_from_db,
    prune_stale_data,
    refresh_all_stats,
    resolve_repositories,
    save_stats_to_db,
)
from grimoire.models import RepositoryStats, TrackedRepository, WorkflowStatus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def engine(tmp_path) -> AsyncEngine:
    db_path = str(tmp_path / "test.db")
    eng = await get_engine(db_path)
    await create_tables(eng)
    return eng


@pytest.fixture
async def client(engine: AsyncEngine) -> GitHubClient:
    c = GitHubClient(token="ghp_test", engine=engine, backoff_factors=(0.0, 0.0, 0.0))
    yield c  # type: ignore[misc]
    await c.close()


def _make_config(
    repos: list[StaticRepoSource | TeamRepoSource] | None = None,
    staleness: StalenessConfig | None = None,
) -> GrimoireConfig:
    return GrimoireConfig(
        github=GitHubConfig(token="ghp_test"),
        repositories=repos or [StaticRepoSource(repo="owner/repo1")],
        staleness=staleness or StalenessConfig(),
    )


# ---------------------------------------------------------------------------
# resolve_repositories — static
# ---------------------------------------------------------------------------


@respx.mock
async def test_resolve_static_repos(client: GitHubClient) -> None:
    respx.get("https://api.github.com/repos/owner/repo1").mock(
        return_value=httpx.Response(
            200,
            json={
                "full_name": "owner/repo1",
                "default_branch": "main",
                "archived": False,
            },
            headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"},
        )
    )
    config = _make_config(
        repos=[StaticRepoSource(repo="owner/repo1", branches=["main", "develop"])]
    )
    repos = await resolve_repositories(config, client)
    assert len(repos) == 1
    assert repos[0].full_name == "owner/repo1"
    assert repos[0].branches == ["main", "develop"]
    assert repos[0].source == "static"


@respx.mock
async def test_resolve_static_uses_default_branch(client: GitHubClient) -> None:
    """When no branches specified, use the repo's default_branch."""
    respx.get("https://api.github.com/repos/owner/repo1").mock(
        return_value=httpx.Response(
            200,
            json={
                "full_name": "owner/repo1",
                "default_branch": "develop",
                "archived": False,
            },
            headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"},
        )
    )
    config = _make_config(repos=[StaticRepoSource(repo="owner/repo1")])
    repos = await resolve_repositories(config, client)
    assert len(repos) == 1
    assert repos[0].branches == ["develop"]


@respx.mock
async def test_resolve_static_keeps_default_branch_on_repeat(client: GitHubClient) -> None:
    """Repeated resolution must not degrade default_branch to a placeholder via ETag 304s."""

    def _side_effect(request: httpx.Request) -> httpx.Response:
        if "If-None-Match" in request.headers:
            return httpx.Response(304, headers=_rl_headers())
        return httpx.Response(
            200,
            json={"full_name": "owner/repo1", "default_branch": "3/edge", "archived": False},
            headers={**_rl_headers(), "ETag": '"abc"'},
        )

    respx.get("https://api.github.com/repos/owner/repo1").mock(side_effect=_side_effect)
    config = _make_config(
        repos=[StaticRepoSource(repo="owner/repo1", branches=["3/edge", "3/edge"])]
    )
    await resolve_repositories(config, client)
    repos = await resolve_repositories(config, client)
    assert repos[0].default_branch == "3/edge"
    assert repos[0].branches == ["3/edge"]


@respx.mock
async def test_resolve_skips_archived(client: GitHubClient) -> None:
    respx.get("https://api.github.com/repos/owner/archived").mock(
        return_value=httpx.Response(
            200,
            json={
                "full_name": "owner/archived",
                "default_branch": "main",
                "archived": True,
            },
            headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"},
        )
    )
    config = _make_config(repos=[StaticRepoSource(repo="owner/archived")])
    repos = await resolve_repositories(config, client)
    assert len(repos) == 0


# ---------------------------------------------------------------------------
# resolve_repositories — team
# ---------------------------------------------------------------------------


@respx.mock
async def test_resolve_team_repos(client: GitHubClient) -> None:
    respx.get("https://api.github.com/orgs/myorg/teams/backend/repos").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "full_name": "myorg/service-a",
                    "default_branch": "main",
                    "archived": False,
                },
                {
                    "full_name": "myorg/service-b",
                    "default_branch": "main",
                    "archived": False,
                },
                {
                    "full_name": "myorg/old-repo",
                    "default_branch": "main",
                    "archived": True,
                },
            ],
            headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"},
        )
    )
    config = _make_config(
        repos=[TeamRepoSource(team="myorg/backend", exclude=["myorg/service-b"])]
    )
    repos = await resolve_repositories(config, client)
    assert len(repos) == 1
    assert repos[0].full_name == "myorg/service-a"
    assert repos[0].source == "team:myorg/backend"


# ---------------------------------------------------------------------------
# fetch_repository_stats — staleness
# ---------------------------------------------------------------------------


@respx.mock
async def test_fetch_stats_stale_detection(client: GitHubClient) -> None:
    now = datetime.now(UTC)
    old_date = (now - timedelta(days=400)).isoformat()
    fresh_date = (now - timedelta(days=10)).isoformat()

    respx.get("https://api.github.com/repos/owner/repo1/issues").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "number": 1,
                    "title": "Old issue",
                    "updated_at": old_date,
                    "created_at": old_date,
                },
                {
                    "number": 2,
                    "title": "Fresh issue",
                    "updated_at": fresh_date,
                    "created_at": fresh_date,
                },
            ],
            headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"},
        )
    )
    respx.get("https://api.github.com/repos/owner/repo1/pulls").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"number": 10, "title": "Old PR", "updated_at": old_date, "created_at": old_date},
                {
                    "number": 11,
                    "title": "Fresh PR",
                    "updated_at": fresh_date,
                    "created_at": fresh_date,
                },
            ],
            headers={"X-RateLimit-Remaining": "4998", "X-RateLimit-Limit": "5000"},
        )
    )
    respx.get("https://api.github.com/repos/owner/repo1/actions/workflows").mock(
        return_value=httpx.Response(
            200,
            json={"total_count": 0, "workflows": []},
            headers={"X-RateLimit-Remaining": "4997", "X-RateLimit-Limit": "5000"},
        )
    )
    respx.get("https://api.github.com/repos/owner/repo1/branches/main").mock(
        return_value=httpx.Response(
            200,
            json={
                "name": "main",
                "commit": {
                    "sha": "abc",
                    "commit": {"committer": {"date": fresh_date}},
                },
            },
            headers={"X-RateLimit-Remaining": "4996", "X-RateLimit-Limit": "5000"},
        )
    )
    respx.get("https://api.github.com/repos/owner/repo1/branches").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "name": "main",
                    "commit": {"sha": "abc"},
                },
                {
                    "name": "old-feature",
                    "commit": {"sha": "def"},
                },
            ],
            headers={"X-RateLimit-Remaining": "4995", "X-RateLimit-Limit": "5000"},
        )
    )
    respx.get("https://api.github.com/repos/owner/repo1/git/commits/abc").mock(
        return_value=httpx.Response(
            200,
            json={"sha": "abc", "committer": {"date": fresh_date}},
            headers={"X-RateLimit-Remaining": "4994", "X-RateLimit-Limit": "5000"},
        )
    )
    respx.get("https://api.github.com/repos/owner/repo1/git/commits/def").mock(
        return_value=httpx.Response(
            200,
            json={"sha": "def", "committer": {"date": old_date}},
            headers={"X-RateLimit-Remaining": "4993", "X-RateLimit-Limit": "5000"},
        )
    )

    repo = TrackedRepository(full_name="owner/repo1", default_branch="main", branches=["main"])
    staleness = StalenessConfig(issues_days=365, pull_requests_days=30)
    stats = await fetch_repository_stats(repo, client, staleness)

    assert stats.open_issues == 2
    assert stats.stale_issues == 1  # only the 400-day-old one
    assert stats.open_pull_requests == 2
    assert stats.stale_pull_requests == 1  # only the 400-day-old one
    assert stats.last_commit_at is not None
    assert stats.total_branches == 2


# ---------------------------------------------------------------------------
# DB round-trip: save + load
# ---------------------------------------------------------------------------


async def test_save_and_load_stats(engine: AsyncEngine) -> None:
    repos = [
        TrackedRepository(
            full_name="owner/repo1",
            default_branch="main",
            branches=["main", "develop"],
            source="static",
        )
    ]
    now = datetime.now(UTC)
    stats_list = [
        RepositoryStats(
            full_name="owner/repo1",
            default_branch="main",
            open_issues=5,
            stale_issues=2,
            open_pull_requests=3,
            stale_pull_requests=1,
            workflows=[
                WorkflowStatus(
                    name="CI",
                    branch="main",
                    status="success",
                    url="https://github.com/owner/repo1/actions/workflows/ci.yml",
                    run_url="https://github.com/owner/repo1/actions/runs/123",
                    kind="scheduled",
                )
            ],
            fetched_at=now,
            last_commit_at=now,
            total_branches=5,
        )
    ]

    await save_stats_to_db(engine, stats_list, repos)

    # Verify records in DB
    async with AsyncSession(engine) as session:
        cached = (await session.exec(select(CachedRepository))).all()
        assert len(cached) == 1
        assert cached[0].full_name == "owner/repo1"
        assert json.loads(cached[0].branches_json) == ["main", "develop"]
        assert cached[0].total_branches == 5
        assert cached[0].last_commit_at is not None

        wf_rows = (await session.exec(select(CachedWorkflowStatus))).all()
        assert len(wf_rows) == 1
        assert wf_rows[0].workflow_name == "CI"
        assert wf_rows[0].status == "success"

    # Load back
    loaded_repos, loaded_stats = await load_stats_from_db(engine)
    assert len(loaded_repos) == 1
    assert loaded_repos[0].full_name == "owner/repo1"
    assert loaded_repos[0].branches == ["main", "develop"]
    assert len(loaded_stats) == 1
    assert loaded_stats[0].workflows[0].name == "CI"
    assert loaded_stats[0].workflows[0].kind == "scheduled"
    assert loaded_stats[0].total_branches == 5
    assert loaded_stats[0].last_commit_at is not None


async def test_save_replaces_old_data(engine: AsyncEngine) -> None:
    """Saving twice should replace, not duplicate, cached records."""
    repos = [TrackedRepository(full_name="owner/repo1", default_branch="main", branches=["main"])]
    now = datetime.now(UTC)
    stats_v1 = [RepositoryStats(full_name="owner/repo1", default_branch="main", fetched_at=now)]
    stats_v2 = [
        RepositoryStats(
            full_name="owner/repo1",
            default_branch="main",
            open_issues=10,
            fetched_at=now,
        )
    ]

    await save_stats_to_db(engine, stats_v1, repos)
    await save_stats_to_db(engine, stats_v2, repos)

    async with AsyncSession(engine) as session:
        cached = (await session.exec(select(CachedRepository))).all()
        assert len(cached) == 1


# ---------------------------------------------------------------------------
# refresh_all_stats — 304 fallback
# ---------------------------------------------------------------------------


@respx.mock
async def test_refresh_falls_back_to_cache_on_304(
    client: GitHubClient, engine: AsyncEngine
) -> None:
    """When repo resolution returns nothing (all 304s), cached repos are used."""
    # Pre-populate the DB with cached data
    repos = [TrackedRepository(full_name="myorg/svc", default_branch="main", branches=["main"])]
    now = datetime.now(UTC)
    stats_list = [
        RepositoryStats(
            full_name="myorg/svc",
            default_branch="main",
            open_issues=3,
            fetched_at=now,
        )
    ]
    await save_stats_to_db(engine, stats_list, repos)

    # Mock team endpoint returning 304 (ETag hit)
    respx.get("https://api.github.com/orgs/myorg/teams/backend/repos").mock(
        return_value=httpx.Response(
            304,
            headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"},
        )
    )

    # Mock the per-repo endpoints (also 304)
    for pattern in [
        "/repos/myorg/svc/issues",
        "/repos/myorg/svc/pulls",
        "/repos/myorg/svc/actions/workflows",
        "/repos/myorg/svc/branches/main",
        "/repos/myorg/svc/branches",
    ]:
        respx.get(f"https://api.github.com{pattern}").mock(
            return_value=httpx.Response(
                304,
                headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"},
            )
        )

    config = _make_config(repos=[TeamRepoSource(team="myorg/backend")])
    result_repos, result_stats = await refresh_all_stats(config, client)

    # Should fall back to cached repos rather than returning empty
    assert len(result_repos) == 1
    assert result_repos[0].full_name == "myorg/svc"
    assert len(result_stats) == 1


@respx.mock
async def test_workflow_runs_304_preserves_previous_status(client: GitHubClient) -> None:
    """When get_workflow_runs returns 304, preserve previous workflow status."""
    # Mock endpoints for a repo with one workflow
    respx.get("https://api.github.com/repos/owner/repo1/issues").mock(
        return_value=httpx.Response(200, json=[], headers=_rl_headers())
    )
    respx.get("https://api.github.com/repos/owner/repo1/pulls").mock(
        return_value=httpx.Response(200, json=[], headers=_rl_headers())
    )
    respx.get("https://api.github.com/repos/owner/repo1/actions/workflows").mock(
        return_value=httpx.Response(
            200,
            json={
                "total_count": 1,
                "workflows": [{"id": 42, "name": "CI", "html_url": "https://github.com/wf"}],
            },
            headers=_rl_headers(),
        )
    )
    # Workflow runs returns 304 (cache hit)
    respx.get("https://api.github.com/repos/owner/repo1/actions/workflows/42/runs").mock(
        return_value=httpx.Response(304, headers=_rl_headers())
    )
    respx.get("https://api.github.com/repos/owner/repo1/branches/main").mock(
        return_value=httpx.Response(
            200,
            json={
                "name": "main",
                "commit": {
                    "sha": "abc",
                    "commit": {"committer": {"date": datetime.now(UTC).isoformat()}},
                },
            },
            headers=_rl_headers(),
        )
    )
    respx.get("https://api.github.com/repos/owner/repo1/branches").mock(
        return_value=httpx.Response(200, json=[], headers=_rl_headers())
    )

    repo = TrackedRepository(full_name="owner/repo1", default_branch="main", branches=["main"])
    previous = RepositoryStats(
        full_name="owner/repo1",
        default_branch="main",
        workflows=[
            WorkflowStatus(
                name="CI",
                branch="main",
                status="success",
                url="https://github.com/wf",
                run_url="https://github.com/run/1",
            )
        ],
    )
    staleness = StalenessConfig()
    stats = await fetch_repository_stats(repo, client, staleness, previous=previous)

    # The previous workflow status should be preserved (not dropped)
    assert len(stats.workflows) == 1
    assert stats.workflows[0].name == "CI"
    assert stats.workflows[0].status == "success"


def _rl_headers() -> dict[str, str]:
    return {"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"}


def _mock_repo_endpoints(branches: list[str]) -> None:
    """Mock issues/PRs/workflows/branches for owner/repo1 with one workflow (id 42)."""
    respx.get("https://api.github.com/repos/owner/repo1/issues").mock(
        return_value=httpx.Response(200, json=[], headers=_rl_headers())
    )
    respx.get("https://api.github.com/repos/owner/repo1/pulls").mock(
        return_value=httpx.Response(200, json=[], headers=_rl_headers())
    )
    respx.get("https://api.github.com/repos/owner/repo1/actions/workflows").mock(
        return_value=httpx.Response(
            200,
            json={
                "total_count": 1,
                "workflows": [
                    {
                        "id": 42,
                        "name": "CI",
                        "path": ".github/workflows/ci.yaml",
                        "html_url": "https://github.com/wf",
                    }
                ],
            },
            headers=_rl_headers(),
        )
    )
    for branch in branches:
        respx.get(f"https://api.github.com/repos/owner/repo1/branches/{branch}").mock(
            return_value=httpx.Response(200, json={"name": branch}, headers=_rl_headers())
        )
    respx.get("https://api.github.com/repos/owner/repo1/branches").mock(
        return_value=httpx.Response(200, json=[], headers=_rl_headers())
    )


def _runs_response(runs: list[dict[str, object]]) -> httpx.Response:
    return httpx.Response(
        200, json={"total_count": len(runs), "workflow_runs": runs}, headers=_rl_headers()
    )


def _run(run_id: int, conclusion: str, age_days: float) -> dict[str, object]:
    created = datetime.now(UTC) - timedelta(days=age_days)
    return {
        "id": run_id,
        "conclusion": conclusion,
        "html_url": f"https://github.com/run/{run_id}",
        "created_at": created.isoformat(),
    }


@respx.mock
async def test_workflow_runs_split_by_event(client: GitHubClient) -> None:
    """Schedule-triggered runs become 'scheduled', push-triggered runs 'release'."""
    _mock_repo_endpoints(["main"])

    def _side_effect(request: httpx.Request) -> httpx.Response:
        event = request.url.params.get("event")
        if event == "schedule":
            return _runs_response([_run(1, "success", 1)])
        if event == "push":
            return _runs_response([_run(2, "failure", 2)])
        return _runs_response([])

    route = respx.get("https://api.github.com/repos/owner/repo1/actions/workflows/42/runs")
    route.side_effect = _side_effect

    repo = TrackedRepository(full_name="owner/repo1", default_branch="main", branches=["main"])
    stats = await fetch_repository_stats(repo, client, StalenessConfig())

    by_kind = {wf.kind: wf for wf in stats.workflows}
    assert len(stats.workflows) == 2
    assert by_kind["scheduled"].status == "success"
    assert by_kind["scheduled"].run_url == "https://github.com/run/1"
    assert by_kind["release"].status == "failure"
    assert by_kind["release"].run_url == "https://github.com/run/2"
    for call in route.calls:
        assert call.request.url.params["branch"] == "main"


@respx.mock
async def test_scheduled_success_rate_and_runs_url(client: GitHubClient) -> None:
    """Scheduled workflows report succeeded/completed runs and link to the scheduled runs list."""
    _mock_repo_endpoints(["main"])

    def _side_effect(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("event") != "schedule":
            return _runs_response([])
        if params.get("status") == "completed":
            assert params["created"].startswith(">=")
            return httpx.Response(200, json={"total_count": 4, "workflow_runs": []})
        if params.get("status") == "success":
            return httpx.Response(200, json={"total_count": 3, "workflow_runs": []})
        return _runs_response([_run(1, "failure", 1)])

    respx.get("https://api.github.com/repos/owner/repo1/actions/workflows/42/runs").mock(
        side_effect=_side_effect
    )

    repo = TrackedRepository(full_name="owner/repo1", default_branch="main", branches=["main"])
    stats = await fetch_repository_stats(repo, client, StalenessConfig())

    assert len(stats.workflows) == 1
    wf = stats.workflows[0]
    assert wf.kind == "scheduled"
    assert wf.success_rate == 0.75
    assert wf.url == (
        "https://github.com/owner/repo1/actions/workflows/ci.yaml?query=event%3Aschedule"
    )


@respx.mock
async def test_workflow_runs_inactive_are_excluded(client: GitHubClient) -> None:
    """Run queries are bounded to the activity window; no runs in it hides the workflow."""
    _mock_repo_endpoints(["main"])
    cutoff = (datetime.now(UTC) - timedelta(days=WORKFLOW_ACTIVE_DAYS)).date().isoformat()

    def _side_effect(request: httpx.Request) -> httpx.Response:
        # Emulate GitHub's server-side `created` filter.
        if request.url.params.get("created") == f">={cutoff}":
            return _runs_response([])
        return _runs_response([_run(1, "failure", WORKFLOW_ACTIVE_DAYS + 1)])

    route = respx.get("https://api.github.com/repos/owner/repo1/actions/workflows/42/runs")
    route.side_effect = _side_effect

    repo = TrackedRepository(full_name="owner/repo1", default_branch="main", branches=["main"])
    stats = await fetch_repository_stats(repo, client, StalenessConfig())

    assert stats.workflows == []
    assert all(c.request.url.params.get("created") == f">={cutoff}" for c in route.calls)


@respx.mock
async def test_workflow_runs_without_matching_event_are_omitted(client: GitHubClient) -> None:
    """A workflow never triggered by schedule/push (e.g. PR-only CI) is not shown."""
    _mock_repo_endpoints(["main"])
    respx.get("https://api.github.com/repos/owner/repo1/actions/workflows/42/runs").mock(
        return_value=_runs_response([])
    )

    repo = TrackedRepository(full_name="owner/repo1", default_branch="main", branches=["main"])
    stats = await fetch_repository_stats(repo, client, StalenessConfig())

    assert stats.workflows == []


@respx.mock
async def test_workflow_runs_scheduled_only_on_default_branch(client: GitHubClient) -> None:
    """Scheduled runs are queried on the default branch only; release runs on
    every tracked branch."""
    _mock_repo_endpoints(["main", "track/3.0"])

    def _side_effect(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("event") == "push" and params.get("branch") == "track/3.0":
            return _runs_response([_run(3, "failure", 1)])
        return _runs_response([_run(4, "success", 1)])

    route = respx.get("https://api.github.com/repos/owner/repo1/actions/workflows/42/runs")
    route.side_effect = _side_effect

    repo = TrackedRepository(
        full_name="owner/repo1", default_branch="main", branches=["main", "track/3.0"]
    )
    stats = await fetch_repository_stats(repo, client, StalenessConfig())

    queried = {
        (c.request.url.params["event"], c.request.url.params["branch"]) for c in route.calls
    }
    assert queried == {("schedule", "main"), ("push", "main"), ("push", "track/3.0")}

    keyed = {(wf.kind, wf.branch): wf.status for wf in stats.workflows}
    assert keyed == {
        ("scheduled", "main"): "success",
        ("release", "main"): "success",
        ("release", "track/3.0"): "failure",
    }


# ---------------------------------------------------------------------------
# prune_removed_repos
# ---------------------------------------------------------------------------


async def test_prune_removes_repos_not_in_config(engine: AsyncEngine) -> None:
    """Repos in DB but not in config should be deleted."""
    from grimoire.github.service import prune_removed_repos

    # Seed DB with two repos
    repo_a = TrackedRepository(full_name="org/keep", default_branch="main", branches=["main"])
    repo_b = TrackedRepository(full_name="org/remove", default_branch="main", branches=["main"])
    stats = [
        RepositoryStats(full_name="org/keep", default_branch="main"),
        RepositoryStats(full_name="org/remove", default_branch="main"),
    ]
    await save_stats_to_db(engine, stats, [repo_a, repo_b])

    # Config only has org/keep
    config = _make_config(repos=[StaticRepoSource(repo="org/keep")])
    pruned = await prune_removed_repos(engine, config)
    assert pruned == 1

    # Verify only org/keep remains
    repos, _ = await load_stats_from_db(engine)
    assert [r.full_name for r in repos] == ["org/keep"]


async def test_prune_skipped_with_team_sources(engine: AsyncEngine) -> None:
    """Pruning should be skipped when team sources are present."""
    from grimoire.github.service import prune_removed_repos

    # Seed DB with a repo
    repo = TrackedRepository(full_name="org/repo", default_branch="main", branches=["main"])
    await save_stats_to_db(
        engine, [RepositoryStats(full_name="org/repo", default_branch="main")], [repo]
    )

    # Config has a team source — can't know full set, so skip pruning
    config = _make_config(repos=[TeamRepoSource(team="org/myteam")])
    pruned = await prune_removed_repos(engine, config)
    assert pruned == 0

    # Repo should still be there
    repos, _ = await load_stats_from_db(engine)
    assert len(repos) == 1


# ---------------------------------------------------------------------------
# prune_stale_data
# ---------------------------------------------------------------------------


async def test_prune_stale_data_removes_repos(engine: AsyncEngine) -> None:
    """Repos not in the active set are fully removed from the DB."""
    # Seed two repos
    for name in ("org/keep", "org/stale"):
        repo = TrackedRepository(full_name=name, default_branch="main", branches=["main"])
        await save_stats_to_db(
            engine,
            [RepositoryStats(full_name=name, default_branch="main")],
            [repo],
        )

    # Only org/keep is active
    active = [TrackedRepository(full_name="org/keep", default_branch="main", branches=["main"])]
    pruned = await prune_stale_data(engine, active)
    assert pruned == 1

    repos, _ = await load_stats_from_db(engine)
    assert [r.full_name for r in repos] == ["org/keep"]


async def test_prune_stale_data_cleans_related_rows(engine: AsyncEngine) -> None:
    """Issues, PRs, workflows, and check results for stale repos are cleaned."""
    repo_name = "org/stale"
    repo = TrackedRepository(full_name=repo_name, default_branch="main", branches=["main"])
    await save_stats_to_db(
        engine,
        [RepositoryStats(full_name=repo_name, default_branch="main")],
        [repo],
    )

    # Insert related rows
    now = datetime(2025, 1, 1, tzinfo=UTC)
    async with AsyncSession(engine) as session:
        session.add(
            CachedIssue(
                repo_full_name=repo_name,
                number=1,
                title="Issue",
                url="https://example.com/1",
                created_at=now,
            )
        )
        session.add(
            CachedPullRequest(
                repo_full_name=repo_name,
                number=2,
                title="PR",
                url="https://example.com/2",
                created_at=now,
            )
        )
        session.add(
            CachedWorkflowStatus(
                repo_full_name=repo_name,
                branch="main",
                workflow_name="CI",
                status="success",
            )
        )
        session.add(
            CheckResultRecord(
                repo_full_name=repo_name,
                check_slug="test-check",
                check_name="Test Check",
                branch="main",
                passed=True,
            )
        )
        await session.commit()

    # Prune with empty active list
    pruned = await prune_stale_data(engine, [])
    assert pruned == 1

    # All related rows should be gone
    async with AsyncSession(engine) as session:
        assert (await session.exec(select(CachedIssue))).all() == []
        assert (await session.exec(select(CachedPullRequest))).all() == []
        assert (await session.exec(select(CachedWorkflowStatus))).all() == []
        assert (await session.exec(select(CheckResultRecord))).all() == []
        assert (await session.exec(select(CachedRepository))).all() == []


async def test_prune_stale_data_removes_excluded_workflows(engine: AsyncEngine) -> None:
    """Workflow statuses excluded by filters are pruned even for active repos."""
    repo_name = "org/repo"
    repo = TrackedRepository(
        full_name=repo_name,
        default_branch="main",
        branches=["main"],
        workflow_exclude=["Nightly *"],
    )
    await save_stats_to_db(
        engine,
        [RepositoryStats(full_name=repo_name, default_branch="main")],
        [repo],
    )

    # Insert workflow rows — one matching exclude, one not
    async with AsyncSession(engine) as session:
        session.add(
            CachedWorkflowStatus(
                repo_full_name=repo_name,
                branch="main",
                workflow_name="CI",
                status="success",
            )
        )
        session.add(
            CachedWorkflowStatus(
                repo_full_name=repo_name,
                branch="main",
                workflow_name="Nightly Tests",
                status="failure",
            )
        )
        await session.commit()

    await prune_stale_data(engine, [repo])

    async with AsyncSession(engine) as session:
        wfs = (await session.exec(select(CachedWorkflowStatus))).all()
        assert len(wfs) == 1
        assert wfs[0].workflow_name == "CI"


async def test_prune_workspace_dirs(engine: AsyncEngine, tmp_path) -> None:
    """Workspace directories for pruned repos are removed from disk."""
    workspace = tmp_path / "workspace"
    # Create dirs for two repos
    (workspace / "org" / "keep").mkdir(parents=True)
    (workspace / "org" / "stale").mkdir(parents=True)
    (workspace / "other" / "gone").mkdir(parents=True)

    active = [TrackedRepository(full_name="org/keep", default_branch="main", branches=["main"])]

    # Seed the stale repos in DB so prune counts them
    for name in ("org/stale", "other/gone"):
        repo = TrackedRepository(full_name=name, default_branch="main", branches=["main"])
        await save_stats_to_db(
            engine,
            [RepositoryStats(full_name=name, default_branch="main")],
            [repo],
        )

    pruned = await prune_stale_data(engine, active, workspace_dir=workspace)
    assert pruned == 2

    # Stale dirs should be gone
    assert (workspace / "org" / "keep").is_dir()
    assert not (workspace / "org" / "stale").exists()
    assert not (workspace / "other").exists()  # empty owner dir removed


# ---------------------------------------------------------------------------
# RefreshProgress tracking
# ---------------------------------------------------------------------------


def test_refresh_progress_initial_state() -> None:
    """Before any refresh, progress should be None and not running."""
    from grimoire.github.service import get_refresh_progress, is_refresh_running

    assert not is_refresh_running()
    assert get_refresh_progress() is None


@respx.mock
async def test_refresh_tracks_progress(client: GitHubClient, engine: AsyncEngine) -> None:
    """Progress should be set during refresh and cleared after."""
    from grimoire.github.service import (
        get_refresh_progress,
        is_refresh_running,
    )

    # Pre-populate cached data
    repos = [TrackedRepository(full_name="myorg/svc", default_branch="main", branches=["main"])]
    now = datetime.now(UTC)
    stats_list = [
        RepositoryStats(
            full_name="myorg/svc", default_branch="main", open_issues=0, fetched_at=now
        )
    ]
    await save_stats_to_db(engine, stats_list, repos)

    # Mock team endpoint returning 304
    respx.get("https://api.github.com/orgs/myorg/teams/backend/repos").mock(
        return_value=httpx.Response(
            304, headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"}
        )
    )
    for pattern in [
        "/repos/myorg/svc/issues",
        "/repos/myorg/svc/pulls",
        "/repos/myorg/svc/actions/workflows",
        "/repos/myorg/svc/branches/main",
        "/repos/myorg/svc/branches",
    ]:
        respx.get(f"https://api.github.com{pattern}").mock(
            return_value=httpx.Response(
                304, headers={"X-RateLimit-Remaining": "4999", "X-RateLimit-Limit": "5000"}
            )
        )

    config = _make_config(repos=[TeamRepoSource(team="myorg/backend")])
    result_repos, result_stats = await refresh_all_stats(config, client)

    # After refresh completes, progress should be cleared
    assert not is_refresh_running()
    assert get_refresh_progress() is None
    assert len(result_repos) == 1
