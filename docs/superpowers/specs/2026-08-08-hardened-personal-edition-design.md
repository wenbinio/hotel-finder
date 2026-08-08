# Hotel Finder: Hardened Personal Edition

**Date:** 2026-08-08  
**Status:** Approved  
**Target:** A fast, reliable, private single-instance hotel search tool

## Context

The current `master` application works against live Google Hotels data, but it has four structural problems:

1. A default cheapest-date sweep can issue 198 mostly serial upstream requests and run into the 120-second deployment timeout.
2. Progress and cache state are process-local while the deployment runs two Gunicorn processes, so polling can read the wrong state.
3. Network errors are commonly converted into empty results, making upstream failures indistinguishable from no availability.
4. The deploy repository contains generated frontend assets but not their source, tests, or a reproducible source-to-bundle workflow.

The operational-fix branch also identifies a broken Xotelo call and a retired `primp` browser profile. The globalization branch contains useful parser tests, caching, and parallel sweep work, but is not safe to merge wholesale because it regresses operational fixes and removes current UI workflows.

## Goals

- Preserve single-destination, all-destination, cheapest-date, and provider-comparison workflows.
- Make the interface responsive while a long sweep runs.
- Bound upstream work, latency, and memory use.
- Return partial results with intelligible warnings instead of silently hiding failures.
- Generate valid future dates every day and prevent invalid or accidentally exhaustive submissions.
- Close the arbitrary-URL fetch vulnerability in provider comparison.
- Track frontend source, backend tests, frontend tests, deployment configuration, and generated assets in one repository.
- Keep local setup and deployment simple: one Python service, one build step, no Redis, database, account system, or external queue.

## Non-goals

- Public multi-user scaling.
- Live flight-price integration; existing flight amounts will be labeled as estimates.
- Supabase, Vercel, Cloud Run, localization, custom destinations, or mobile applications.
- CAPTCHA bypassing or aggressive scraping concurrency.
- Guaranteed prices or availability; all displayed upstream data remains advisory and must be confirmed with the provider.

## Architecture

### Runtime model

Run one Gunicorn process with eight request threads. This keeps the private in-memory job registry and caches coherent while allowing progress polling and ordinary searches during a sweep.

The service owns:

- a bounded search executor;
- a thread-safe TTL cache with request coalescing;
- a single-active-sweep job registry keyed by opaque job IDs;
- a bounded upstream concurrency gate;
- structured application logging.

All state is intentionally ephemeral. Restarting the private service clears jobs and caches. This is acceptable for the target deployment and will be stated in the README.

### Source layout

- `app.py`: Flask application factory, routes, and response shaping.
- `hotel_finder/parsing.py`: pure Google/Xotelo parsing helpers.
- `hotel_finder/cache.py`: bounded TTL cache and single-flight behavior.
- `hotel_finder/jobs.py`: sweep lifecycle, cancellation, progress, result retention, and cleanup.
- `hotel_finder/validation.py`: request schemas, date rules, limits, and URL validation.
- `frontend/`: tracked React source, tests, lint configuration, lockfile, and build scripts.
- `static/`: generated deploy assets committed for the Python-only deployment.
- `tests/`: backend unit, route, cache, job, validation, and parsing tests.

## Request and job flow

### Ordinary search

`POST /api/search` and `POST /api/search-all` remain synchronous because their bounded fan-out normally completes quickly. Each response contains:

- results;
- query metadata;
- source status;
- warnings and failed destinations, when applicable;
- cache-hit information suitable for diagnostics.

`search-all` uses controlled destination concurrency and returns partial results if individual destinations fail.

### Cheapest-date sweep

`POST /api/cheapest-dates` validates and normalizes the query, then returns HTTP 202 with a job ID immediately. Only one sweep may run at a time. A new sweep may explicitly replace the current sweep; otherwise the API returns a conflict response.

`GET /api/sweeps/<job_id>` returns queued/running/completed/failed/cancelled state, progress counters, partial date summaries, warnings, and the final result.

`DELETE /api/sweeps/<job_id>` requests cooperative cancellation. Workers check cancellation between upstream calls and before publishing results.

Completed jobs are retained for 30 minutes, with at most eight retained records. Cleanup is automatic.

## Performance and upstream behavior

- Search identical normalized queries through a bounded TTL cache.
- Coalesce concurrent identical cache misses so only one upstream request is made.
- Cache successful hotel searches for 10 minutes, provider comparisons for 15 minutes, and genuine empty results briefly.
- Do not cache transport failures as empty inventory.
- Search at most four upstream destinations concurrently by default; allow configuration up to six for a private machine.
- Remove arbitrary post-request sleeps. Use the concurrency gate and one bounded retry with jitter for transient 429/5xx failures.
- Set explicit connect, read, and total timeouts on Google and Xotelo calls.
- Use a pinned, supported `primp` profile with TLS verification enabled.
- Reuse one client per worker thread because `primp` does not document a thread-safety guarantee for shared clients.
- Include every output-affecting value in cache keys.

The default sweep will sample six dates over the next 90 days. Larger ranges and more samples remain available up to a hard budget of 200 logical hotel searches per sweep; a bounded retry does not consume another logical-search slot.

## Validation and security

Central validation will enforce:

- JSON object request bodies;
- known destination names;
- dates in ISO format, no past check-in, and checkout after check-in;
- `1 <= nights <= 30`;
- `3 <= minStars <= 5`;
- bounded flight budget, date range, sample count, hotel count, and total upstream-call budget;
- well-formed hotel objects for provider comparison.

Provider comparison may fetch only HTTPS Google Hotels entity URLs on an explicit hostname allowlist. Redirect targets must remain allowed. The server will not fetch client-supplied arbitrary hosts or private addresses.

CORS defaults to same-origin. An optional environment variable may list additional private origins.

## Frontend behavior

- Generate check-in as tomorrow and checkout five nights later.
- Treat the two ordinary-search dates as authoritative and derive the displayed night count from them so totals cannot disagree with the dates.
- Default sweep window to tomorrow through 90 days later with six samples and one-night stays.
- Add HTML date minima and client validation that mirrors backend rules.
- Prevent blank Single Location mode from submitting as All Destinations.
- Use `/api` relative URLs in every environment.
- Use one request controller and run ID per operation. Superseded requests and progress polls are aborted; late results cannot overwrite newer results.
- Disable conflicting operations while a sweep is active, while retaining sweep cancellation.
- Centralize checked JSON fetching and show operation-specific, accessible errors and partial-result warnings.
- Keep query metadata attached to results so dates, stars, and totals cannot become stale.
- Remove the non-functional Room Quality filter.
- Derive star labels from the actual threshold and omit missing timings instead of rendering `nulls`.
- Sort and display total-trip prices through one shared best-rate calculation.
- Merge compared hotels by a stable identity, not name alone.
- Label flight amounts as static round-trip estimates.
- Add accessible selection state, live status/progress, table captions, visible keyboard focus, and text alternatives for charts.

## Error model

Validation failures return HTTP 400 with a stable code and field-level details. A busy sweep returns 409. Upstream timeouts or blocking are recorded per source. Single-source failure with no usable data returns 502 or 504; multi-source and multi-destination operations return successful partial results plus warnings when usable data exists.

Every executor future is isolated so one malformed hotel or provider cannot fail a batch. Job state is finalized in `finally`, and server logs include job IDs and request IDs without exposing sensitive data.

## Tests and quality gates

### Backend

- Pure parser tests ported from the globalization branch.
- Validation boundary and malformed-payload tests.
- SSRF/redirect allowlist tests.
- Cache expiry, key completeness, negative-cache, and single-flight tests.
- Sweep lifecycle, cancellation, cleanup, concurrency, and call-budget tests.
- Route tests with mocked upstream responses for success, partial failure, timeout, and malformed content.
- A regression test for the Xotelo argument mismatch and per-hotel failure isolation.

### Frontend

- Fake-clock tests for dynamic future defaults and stay-length synchronization.
- Submission guards for invalid dates and blank single-destination mode.
- Deferred-fetch tests proving stale operations cannot overwrite current results.
- Error tests for non-2xx JSON and HTML responses.
- Sweep progress and cancellation tests with fake timers.
- Rendering tests for dynamic star labels, missing timing, stable hotel identities, and total sorting.
- Automated accessibility smoke tests plus keyboard-focused component checks.

### Repository and deployment

- Pin Python runtime dependencies and Node engine expectations.
- Add a development requirements file for test and quality tools.
- Add lint/format checks appropriate to Python and React.
- Build frontend assets directly into `static/` through one documented command.
- CI performs clean Python install, backend tests, frontend clean install, lint, frontend tests, production build, and a generated-asset diff check.
- Add `/api/health` with application and dependency-profile metadata but no live upstream call.
- Add `.dockerignore` and deployment smoke instructions.

## Acceptance criteria

- The repository is clean after regenerating committed frontend assets.
- Backend tests, frontend tests, linters, and production build all pass.
- Tomorrow-based defaults pass under a fake clock and render correctly in a browser.
- Sweep creation returns a job ID without waiting for upstream completion; progress and cancellation work in the deployed one-process/threaded model.
- A simulated 198-call sweep never exceeds the configured concurrency or call budget and cannot be corrupted by a second request.
- Repeated identical mocked searches make one upstream call within the cache TTL.
- A failing destination/provider produces partial results and a visible warning rather than a false empty-success response or batch 500.
- Arbitrary and redirected non-Google URLs are rejected before any network request.
- A live smoke test can search a future stay, compare providers, run a bounded sweep, and render results without browser console errors.

## Upgrade paths

1. **Persistent personal service:** replace in-memory caches/jobs with SQLite while keeping the same interfaces.
2. **Shared/private group service:** add Redis-backed jobs/cache and authentication without changing frontend job semantics.
3. **Public service:** add an external worker queue, distributed rate limiting, durable observability, quotas, and provider contracts.
4. **Product expansion:** separately integrate real flight pricing, localization, custom destinations, and alternate deployment targets after their incomplete feature-branch implementations are corrected and tested.

## Branch strategy

The integration branch starts from `claude/operational-fix-a4rj5u` (`c81f3864`). It retains the supported browser profile, fixed Xotelo call, per-hotel failure isolation, and one-process/eight-thread deployment. Parser tests, cache concepts, and per-date destination parallelism will be selectively ported from `claude/add-globalization-support-Emx4x`; that branch will not be merged wholesale.
