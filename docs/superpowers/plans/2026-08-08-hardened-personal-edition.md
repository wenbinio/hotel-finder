# Hardened Personal Hotel Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing hotel finder into a fast, bounded, test-covered private service with dynamic defaults, background date sweeps, partial-failure reporting, and a reproducible frontend/deployment workflow.

**Architecture:** Keep a single Gunicorn process with eight threads so bounded in-memory jobs and caches stay coherent. Ordinary searches remain synchronous with controlled fan-out; cheapest-date sweeps become cancellable background jobs. Pure parsing, validation, cache, and job behavior live in focused Python modules, while tracked React source uses a checked API layer and latest-request ownership.

**Tech Stack:** Python 3.13, Flask 3.1.3, `primp` 1.3.1, `selectolax` 0.4.11, `requests` 2.34.2, Gunicorn 23.0.0, React 19.2.4, Vite 8.0.3, Vitest 4.1.10, Testing Library 16.3.2, ESLint 9.39.4, pytest 9.1.1, Ruff 0.16.2.

## Global Constraints

- Preserve single-destination, all-destination, cheapest-date, and provider-comparison workflows.
- Target one private service instance; do not add Redis, a database, authentication, Supabase, Vercel, Cloud Run, localization, or custom destinations.
- Run one Gunicorn process with eight request threads.
- Permit only one active date sweep and retain at most eight job records for 30 minutes.
- Default upstream destination concurrency is four and must never exceed configurable maximum six.
- Default sweep is six sampled dates over the next 90 days; reject more than 200 logical hotel searches.
- Google and provider transport failures must never be represented as genuine empty inventory.
- Provider comparison may fetch only validated HTTPS Google Hotels entity URLs.
- Flight amounts remain static and must be labeled as estimates.
- Use TDD for every behavior change: observe the regression test fail, implement the minimum behavior, then rerun it.
- Do not merge `claude/add-globalization-support-Emx4x`; port only reviewed parser-test, cache, and sweep-concurrency ideas.
- Commit generated `static/` assets and require them to be reproducible from tracked `frontend/` source.

## File Map

- `app.py`: Flask application factory, routes, upstream orchestration, partial-result shaping.
- `hotel_finder/__init__.py`: package marker and public version.
- `hotel_finder/parsing.py`: pure star, hotel-card, and provider-price parsing.
- `hotel_finder/cache.py`: bounded TTL cache and same-key request coalescing.
- `hotel_finder/jobs.py`: single-active-sweep lifecycle and snapshots.
- `hotel_finder/validation.py`: request dataclasses, field validation, date sampling, Google URL allowlist.
- `tests/`: Python regression, unit, route, and repository-contract tests.
- `frontend/src/lib/dates.js`: local-date defaults and date arithmetic.
- `frontend/src/lib/api.js`: checked JSON requests and sweep API.
- `frontend/src/lib/pricing.js`: stable identities and consistent rate/total calculations.
- `frontend/src/hooks/useLatestRequest.js`: abort and stale-completion ownership.
- `frontend/src/components/`: search, sweep, progress, chart, and result components.
- `frontend/src/App.jsx`: page-level state and orchestration only.
- `frontend/src/*.test.*`: Vitest/Testing Library tests.
- `scripts/verify_static.py`: checks generated asset references and source/bundle parity.
- `.github/workflows/ci.yml`: clean backend/frontend quality gates.
- `README.md`: local run, test, build, private deployment, and operational caveats.

## Parallel Execution Waves

- **Wave 1:** Task 1 on the integration branch.
- **Wave 2, parallel worktrees:** Task 2 (parsing/validation), Task 3 (cache/jobs), and Task 4 (frontend pure modules).
- **Wave 3:** Cherry-pick Wave 2, then Task 5 (backend integration) and Task 6 (frontend components) in parallel worktrees.
- **Wave 4:** Cherry-pick Wave 3, then Task 7 (full frontend integration) and Task 8 (deployment/CI/docs) in parallel.
- **Wave 5:** Task 9 full verification, browser smoke, and cleanup on the integration branch.

---

### Task 1: Track Source and Establish Quality Tooling

**Files:**
- Create: `frontend/**` by mechanically importing the existing source from `../frontend/`, excluding `node_modules/` and `dist/`
- Create: `requirements-dev.txt`
- Create: `pyproject.toml`
- Create: `tests/test_repo_contract.py`
- Create: `scripts/verify_static.py`
- Modify: `.gitignore`
- Modify: `frontend/package.json`
- Modify: `frontend/vite.config.js`

**Interfaces:**
- Consumes: existing generated files under `static/` and the byte-identical source workspace at `../frontend/`.
- Produces: `python scripts/verify_static.py --check`, `npm run build:hostable`, backend test/lint commands, and tracked frontend source for later tasks.

- [ ] **Step 1: Write the failing repository-contract test**

```python
# tests/test_repo_contract.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_source_and_hostable_build_are_tracked():
    package = json.loads((ROOT / "frontend" / "package.json").read_text("utf-8"))
    assert package["scripts"]["build:hostable"] == "vite build --outDir ../static --emptyOutDir"
    assert (ROOT / "frontend" / "src" / "App.jsx").is_file()
    assert (ROOT / "frontend" / "package-lock.json").is_file()


def test_static_verifier_exists():
    assert (ROOT / "scripts" / "verify_static.py").is_file()
```

- [ ] **Step 2: Run the test to verify the tracked-source contract is absent**

Run: `python -m pytest tests/test_repo_contract.py -q`  
Expected: FAIL because `frontend/package.json` and `scripts/verify_static.py` are not in the deploy repository.

- [ ] **Step 3: Import source and add exact quality dependencies**

Mechanically copy `../frontend` into `frontend`, excluding `node_modules` and `dist`. Add:

```text
# requirements-dev.txt
pytest==9.1.1
pytest-cov==7.1.0
ruff==0.16.2
```

Add `.worktrees/`, `.venv/`, `frontend/node_modules/`, `frontend/dist/`, `.pytest_cache/`, `.ruff_cache/`, and coverage files to `.gitignore`.

- [ ] **Step 4: Configure Python quality gates**

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["E501"]
```

- [ ] **Step 5: Add reproducible frontend scripts and test dependencies**

Set `frontend/package.json` scripts and engine to:

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "build:hostable": "vite build --outDir ../static --emptyOutDir",
    "lint": "eslint .",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "engines": { "node": ">=22.12" }
}
```

Add exact dev dependencies: `vitest@4.1.10`, `jsdom@30.0.1`, `@testing-library/react@16.3.2`, `@testing-library/jest-dom@7.0.0`, `@testing-library/user-event@14.6.3`, and `eslint-plugin-jsx-a11y@6.10.2`. Configure Vitest for `jsdom` in `vite.config.js`.

- [ ] **Step 6: Implement the static verifier**

```python
# scripts/verify_static.py
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"


def referenced_assets(index_text: str) -> set[str]:
    return set(re.findall(r'(?:src|href)="/([^"?#]+)', index_text))


def main() -> int:
    index = STATIC / "index.html"
    if not index.is_file():
        print("static/index.html is missing", file=sys.stderr)
        return 1
    missing = [name for name in referenced_assets(index.read_text("utf-8")) if not (STATIC / name).is_file()]
    if missing:
        print("missing static assets: " + ", ".join(sorted(missing)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Install cleanly and verify the foundation**

Run:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/test_repo_contract.py -q
cd frontend
npm ci --no-audit --no-fund
npm run build:hostable
cd ..
python scripts/verify_static.py
```

Expected: contract tests PASS, production build succeeds, verifier exits 0.

- [ ] **Step 8: Commit**

```bash
git add .gitignore frontend requirements-dev.txt pyproject.toml tests/test_repo_contract.py scripts/verify_static.py static
git commit -m "build: track frontend source and quality tooling"
```

---

### Task 2: Pure Parsing and Request Validation

**Files:**
- Create: `hotel_finder/__init__.py`
- Create: `hotel_finder/parsing.py`
- Create: `hotel_finder/validation.py`
- Create: `tests/test_parsing.py`
- Create: `tests/test_validation.py`

**Interfaces:**
- Consumes: destination names supplied later by `app.py`.
- Produces: `parse_hotel_cards(html, context)`, `parse_provider_prices(html)`, `parse_search_request`, `parse_search_all_request`, `parse_sweep_request`, `parse_compare_request`, `sample_stay_dates`, and `validate_google_hotel_url`.

- [ ] **Step 1: Write parsing regression tests before moving parsing code**

```python
from hotel_finder.parsing import ParseContext, parse_hotel_cards, parse_provider_prices


def test_hotel_card_extracts_star_price_and_metadata():
    html = '''<div class="uaTTDe"><h2 class="BgYkof">Test Grand Hotel</h2>
    <span class="KFi5wf lA0BZ">4.7</span><span class="ne5qie Ih19Ad">5-star hotel</span>
    <span class="LtjZ2d">Pool</span><span>$220</span><a href="/travel/hotels/entity/abc">open</a></div>'''
    hotels = parse_hotel_cards(html, ParseContext("Bangkok", "2026-08-09", "2026-08-10", 5, "non_beachfront", 126))
    assert hotels[0]["name"] == "Test Grand Hotel"
    assert hotels[0]["price"] == 220.0
    assert hotels[0]["star_class"] == 5
    assert hotels[0]["url"] == "https://www.google.com/travel/hotels/entity/abc"


def test_provider_parser_reads_escaped_dollar_prices():
    html = r'Agoda data \u0024220 Booking.com data \x24230'
    assert parse_provider_prices(html) == {"agoda": 220.0, "booking.com": 230.0}
```

- [ ] **Step 2: Verify parsing tests fail**

Run: `python -m pytest tests/test_parsing.py -q`  
Expected: FAIL with `ModuleNotFoundError: hotel_finder`.

- [ ] **Step 3: Implement pure parsing**

Define the pure parser with the current selectors and no network access:

```python
@dataclass(frozen=True)
class ParseContext:
    location: str
    checkin: str
    checkout: str
    min_stars: int
    category: str
    flight_cost: float


def parse_hotel_cards(html: str, context: ParseContext) -> list[dict]:
    parser = LexborHTMLParser(html)
    hotels = []
    for card in parser.css("div.uaTTDe"):
        name_node = card.css_first("h2.BgYkof") or card.css_first("h2.Cx32Ud")
        if name_node is None:
            continue
        name = name_node.text(strip=True)
        rating_node = card.css_first("span.KFi5wf.lA0BZ")
        try:
            rating = float(rating_node.text(strip=True)) if rating_node else None
        except ValueError:
            rating = None
        html_star = next((int(match.group(1)) for node in card.css("span.ne5qie.Ih19Ad")
                          if (match := re.match(r"([1-5])-star", node.text(strip=True)))), None)
        star = html_star or brand_star_class(name)
        if star is None or star < context.min_stars:
            continue
        price_match = re.search(r"\$([0-9,]+)", card.text())
        if price_match is None:
            continue
        price = float(price_match.group(1).replace(",", ""))
        if price > 1500:
            continue
        link = card.css_first("a[href]")
        href = link.attributes.get("href", "") if link else ""
        url = "https://www.google.com" + href if href.startswith("/travel/") else href or None
        amenities = [node.text(strip=True) for node in card.css("span.LtjZ2d")
                     if len(node.text(strip=True)) > 2]
        hotels.append({
            "name": name, "price": price, "rating": rating, "star_class": star,
            "confirmation": "html" if html_star else "brand", "amenities": amenities, "url": url,
            "location": context.location, "checkin": context.checkin,
            "checkout": context.checkout, "category": context.category,
            "flight_cost": context.flight_cost,
        })
    return sorted(hotels, key=lambda hotel: hotel["price"])


def parse_provider_prices(html: str) -> dict[str, float]:
    providers = {}
    for display, key in PROVIDER_NAMES:
        for match in re.finditer(re.escape(display), html, re.IGNORECASE):
            chunk = html[max(0, match.start() - 300):match.end() + 300]
            prices = re.findall(r"(?:\\x24|\\u0024|\$)(\d+)", chunk)
            if prices and 10 < (price := float(prices[0])) < 2000:
                providers[key] = min(price, providers.get(key, price))
                break
    return providers
```

Move current selectors, brand rules, amenity fallback, price bounds, and URL normalization into these pure functions. Do not perform network calls in this module.

- [ ] **Step 4: Write validation and security tests**

```python
from datetime import date
import pytest
from hotel_finder.validation import ValidationProblem, parse_sweep_request, validate_google_hotel_url


def test_sweep_rejects_past_and_excessive_work():
    with pytest.raises(ValidationProblem, match="startDate"):
        parse_sweep_request({"locations": ["Bangkok"], "startDate": "2026-08-07", "endDate": "2026-08-20"}, {"Bangkok"}, today=date(2026, 8, 8))
    known = {f"City {index}" for index in range(21)}
    with pytest.raises(ValidationProblem, match="200"):
        parse_sweep_request({"locations": sorted(known), "startDate": "2026-08-09", "endDate": "2026-11-07", "sampleCount": 10}, known, today=date(2026, 8, 8))


@pytest.mark.parametrize("url", [
    "http://www.google.com/travel/hotels/entity/abc",
    "https://evil.example/travel/hotels/entity/abc",
    "https://127.0.0.1/latest/meta-data",
])
def test_provider_url_allowlist_rejects_non_google_targets(url):
    with pytest.raises(ValidationProblem):
        validate_google_hotel_url(url)
```

- [ ] **Step 5: Verify validation tests fail**

Run: `python -m pytest tests/test_validation.py -q`  
Expected: FAIL because the validation API does not exist.

- [ ] **Step 6: Implement typed validation**

Create frozen dataclasses `SearchRequest`, `SearchAllRequest`, `SweepRequest`, `HotelInput`, and `CompareRequest`. `SweepRequest.logical_call_count` must be `len(locations) * len(sample_stay_dates(request.start_date, request.end_date, request.nights, request.sample_count))` and must not exceed 200. `validate_google_hotel_url` must accept only `https`, hostname `www.google.com`, and paths beginning `/travel/hotels/entity/`.

- [ ] **Step 7: Run and commit**

Run: `python -m pytest tests/test_parsing.py tests/test_validation.py -q`  
Expected: PASS.

```bash
git add hotel_finder tests/test_parsing.py tests/test_validation.py
git commit -m "feat: add pure parsing and bounded validation"
```

---

### Task 3: Bounded Cache and Sweep Job Runtime

**Files:**
- Create: `hotel_finder/cache.py`
- Create: `hotel_finder/jobs.py`
- Create: `tests/test_cache.py`
- Create: `tests/test_jobs.py`

**Interfaces:**
- Consumes: monotonic clock callables and a `runner(job, payload)` callback.
- Produces: `TTLCache.get_or_load`, `CacheResult`, `SweepJobManager.start/get/cancel/cleanup`, and immutable job snapshots.

- [ ] **Step 1: Write cache expiry and single-flight tests**

```python
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from hotel_finder.cache import TTLCache


def test_get_or_load_coalesces_same_key():
    cache = TTLCache(max_entries=8, ttl_seconds=60)
    started = Event()
    release = Event()
    calls = 0

    def loader():
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(2)
        return ["hotel"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(cache.get_or_load, "same", loader) for _ in range(4)]
        assert started.wait(1)
        release.set()
        results = [future.result(timeout=2).value for future in futures]
    assert calls == 1
    assert results == [["hotel"]] * 4
```

- [ ] **Step 2: Verify cache tests fail**

Run: `python -m pytest tests/test_cache.py -q`  
Expected: FAIL because `hotel_finder.cache` does not exist.

- [ ] **Step 3: Implement cache without caching exceptions**

```python
@dataclass(frozen=True)
class CacheResult(Generic[V]):
    value: V
    hit: bool


@dataclass
class _Entry(Generic[V]):
    value: V
    expires_at: float
```

Implement `TTLCache.__init__(max_entries, ttl_seconds, clock=time.monotonic)`, `get(key)`, `set(key, value, ttl_seconds=None)`, `get_or_load(key, loader, ttl_seconds=None)`, and `clear()`. Protect entries and an `_inflight: dict[K, threading.Event]` with one lock. The first miss creates the event and becomes owner; waiters release the lock and wait. The owner runs `loader` outside the lock, stores only a successful value, removes the event, and wakes all waiters in `finally`. Evict expired entries first and then the entry with the earliest expiry when capacity is reached.

- [ ] **Step 4: Write job lifecycle tests**

```python
from threading import Event
import pytest
from hotel_finder.jobs import JobConflict, SweepJobManager


def test_only_one_active_job_and_cancel_is_cooperative():
    entered = Event()
    release = Event()

    def runner(job, payload):
        entered.set()
        release.wait(2)
        if job.cancel_requested:
            return None
        return {"ok": True}

    manager = SweepJobManager(max_retained=8, retention_seconds=1800)
    first = manager.start({"query": 1}, runner)
    assert entered.wait(1)
    with pytest.raises(JobConflict):
        manager.start({"query": 2}, runner)
    manager.cancel(first.id)
    release.set()
    assert manager.wait(first.id, timeout=2).status == "cancelled"
```

- [ ] **Step 5: Verify job tests fail, then implement**

Run: `python -m pytest tests/test_jobs.py -q`  
Expected: FAIL because `hotel_finder.jobs` does not exist.

Implement `SweepJob` with `id`, `status`, timestamps, `progress`, `partial`, `result`, `warnings`, `error`, and a private cancellation event. Implement `SweepJobManager` with a one-worker executor, lock-protected registry, `replace=True` cancellation, immutable JSON-safe snapshots, `wait(job_id, timeout)`, and cleanup by retention/max count.

- [ ] **Step 6: Run and commit**

Run: `python -m pytest tests/test_cache.py tests/test_jobs.py -q`  
Expected: PASS with no hanging threads.

```bash
git add hotel_finder/cache.py hotel_finder/jobs.py tests/test_cache.py tests/test_jobs.py
git commit -m "feat: add coalescing cache and sweep jobs"
```

---

### Task 4: Frontend Pure Utilities and Latest-Request Ownership

**Files:**
- Create: `frontend/src/lib/dates.js`
- Create: `frontend/src/lib/dates.test.js`
- Create: `frontend/src/lib/api.js`
- Create: `frontend/src/lib/api.test.js`
- Create: `frontend/src/lib/pricing.js`
- Create: `frontend/src/lib/pricing.test.js`
- Create: `frontend/src/hooks/useLatestRequest.js`
- Create: `frontend/src/hooks/useLatestRequest.test.jsx`
- Create: `frontend/src/test/setup.js`

**Interfaces:**
- Consumes: standard `fetch`, `AbortController`, and local `Date`.
- Produces: `defaultSearchDates`, `defaultSweepDates`, `nightsBetween`, `fetchJson`, sweep API functions, `stableHotelKey`, `bestNightlyRate`, `sortHotels`, and `useLatestRequest`.

- [ ] **Step 1: Write fake-clock date tests**

```javascript
import { describe, expect, it } from 'vitest'
import { defaultSearchDates, defaultSweepDates, nightsBetween } from './dates'

describe('date defaults', () => {
  it('uses local tomorrow and a five-night stay', () => {
    expect(defaultSearchDates(new Date(2026, 7, 8, 12))).toEqual({
      checkin: '2026-08-09', checkout: '2026-08-14', nights: 5,
    })
  })
  it('uses a 90-day sweep window and six samples', () => {
    expect(defaultSweepDates(new Date(2026, 7, 8, 12))).toEqual({
      startDate: '2026-08-09', endDate: '2026-11-07', nights: 1, sampleCount: 6,
    })
  })
  it('derives nights from authoritative dates', () => {
    expect(nightsBetween('2026-08-09', '2026-08-14')).toBe(5)
  })
})
```

- [ ] **Step 2: Verify date tests fail, then implement local-date arithmetic**

Run: `npm test -- src/lib/dates.test.js` from `frontend/`  
Expected: FAIL because `dates.js` is missing.

Use local calendar construction (`new Date(year, month, day)`) rather than slicing `toISOString()`, which can shift dates across time zones.

- [ ] **Step 3: Write checked-fetch tests**

```javascript
it('turns an HTML 500 into a friendly ApiError', async () => {
  global.fetch = vi.fn().mockResolvedValue(new Response('<h1>error</h1>', {
    status: 500, headers: { 'content-type': 'text/html' },
  }))
  await expect(fetchJson('/api/search')).rejects.toMatchObject({
    name: 'ApiError', status: 500, code: 'http_error',
  })
})
```

- [ ] **Step 4: Implement the API layer**

```javascript
export class ApiError extends Error {
  constructor(message, { status = 0, code = 'network_error', details = null } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
  }
}

export async function fetchJson(path, { body, headers, ...options } = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...(body === undefined ? {} : { 'content-type': 'application/json' }), ...headers },
    body: body === undefined || typeof body === 'string' ? body : JSON.stringify(body),
  })
  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json') ? await response.json() : null
  if (!response.ok) {
    const error = payload?.error
    throw new ApiError(error?.message || `Request failed (${response.status})`, {
      status: response.status, code: error?.code || 'http_error', details: error?.fields || null,
    })
  }
  if (payload === null) {
    throw new ApiError('Server returned a non-JSON response', { status: response.status, code: 'invalid_response' })
  }
  return payload
}
export const createSweep = (payload, signal) => fetchJson('/api/cheapest-dates', { method: 'POST', body: payload, signal })
export const getSweep = (id, signal) => fetchJson(`/api/sweeps/${encodeURIComponent(id)}`, { signal })
export const cancelSweep = (id, signal) => fetchJson(`/api/sweeps/${encodeURIComponent(id)}`, { method: 'DELETE', signal })
```

Always use relative `/api` paths, check `response.ok`, inspect content type before parsing, and preserve structured backend error codes.

- [ ] **Step 5: Test and implement pricing identity**

```javascript
it('does not merge same-named hotels in different places', () => {
  expect(stableHotelKey({ name: 'Grand Hotel', location: 'Bangkok' }))
    .not.toBe(stableHotelKey({ name: 'Grand Hotel', location: 'Phuket' }))
})

it('sorts total trip by the same best rate used for display', () => {
  const sorted = sortHotels([{ name: 'A', price: 100, providers: { x: { rate: 80 } }, flight_cost: 20 }, { name: 'B', price: 90, flight_cost: 30 }], 'total', 2)
  expect(sorted.map(h => h.name)).toEqual(['A', 'B'])
})
```

- [ ] **Step 6: Test and implement latest-request ownership**

`useLatestRequest` must abort the prior controller on `begin()`, return `{ signal, isCurrent, finish }`, and abort on unmount. A deferred-promise hook test must resolve request 1 after request 2 and prove `isCurrent()` is false for request 1.

- [ ] **Step 7: Run and commit**

Run: `npm test -- src/lib src/hooks`  
Expected: PASS.

```bash
git add frontend/src/lib frontend/src/hooks frontend/src/test frontend/vite.config.js frontend/package.json frontend/package-lock.json
git commit -m "feat: add frontend request and pricing primitives"
```

---

### Task 5: Integrate Hardened Backend APIs

**Files:**
- Modify: `app.py`
- Modify: `requirements.txt`
- Create: `tests/test_api.py`
- Create: `tests/test_upstream.py`
- Create: `tests/test_sweep_api.py`

**Interfaces:**
- Consumes: all Task 2 and Task 3 interfaces.
- Produces: `create_app(test_config=None)`, hardened existing routes, `GET /api/health`, and job-based sweep GET/DELETE routes.

- [ ] **Step 1: Write route tests with injected upstream functions**

```python
def test_search_all_returns_partial_results_and_warning(client, monkeypatch):
    def fake_search(location, *_args, **_kwargs):
        if location == "Phuket":
            raise UpstreamError("timeout", source="google", retryable=True)
        return [{"name": "Bangkok Test", "location": location, "price": 100.0}]
    monkeypatch.setattr("app.search_hotels", fake_search)
    response = client.post("/api/search-all", json={
        "checkin": "2026-08-09", "checkout": "2026-08-14",
        "destinations": ["Bangkok", "Phuket"], "minStars": 5, "maxFlight": 300,
    })
    assert response.status_code == 200
    assert response.json["totalNonBeachfront"] == 1
    assert response.json["failedDestinations"] == [{"location": "Phuket", "code": "timeout"}]
```

Add tests for malformed JSON, past dates, sampleCount zero, >200 logical calls, unsupported Google URL, non-200 Google response, and one failed comparison future.

- [ ] **Step 2: Verify route tests fail for structured validation and partial results**

Run: `python -m pytest tests/test_api.py tests/test_upstream.py tests/test_sweep_api.py -q`  
Expected: FAIL because current routes return 500/empty lists and sweeps are synchronous.

- [ ] **Step 3: Pin runtime dependencies and build the client factory**

Set exact requirements:

```text
flask==3.1.3
flask-cors==6.0.5
fast-hotels==0.2.1
selectolax==0.4.11
primp==1.3.1
requests==2.34.2
gunicorn==23.0.0
```

Use thread-local clients:

```python
_client_local = threading.local()


def make_client() -> Client:
    if not hasattr(_client_local, "client"):
        _client_local.client = Client(
            impersonate="chrome_146", verify=True,
            connect_timeout=5, read_timeout=12, timeout=15,
            follow_redirects=False,
        )
    return _client_local.client
```

- [ ] **Step 4: Refactor search and provider calls through pure parsing and caches**

`search_hotels` must normalize its full key `(location, checkin, checkout, min_stars)` and call `hotel_finder.parsing.parse_hotel_cards`. Raise `UpstreamError` for transport errors, 429, 5xx, and unexpected content instead of returning `[]`. Cache successful results for 600 seconds and true empty results for 60 seconds.

`fetch_provider_prices` must validate its URL before creating a request, reject redirects to any target not independently allowlisted, parse through `parse_provider_prices`, and cache for 900 seconds.

Wrap Google calls in one process-wide semaphore whose default capacity is four and whose environment-configured capacity is clamped to `1..6`. Implement `call_with_retry(operation)` with at most two attempts: retry only transport failures, 429, and 5xx; wait `0.25 + random.uniform(0, 0.25)` seconds before the second attempt; never retry validation failures or other 4xx responses.

- [ ] **Step 5: Implement bounded partial-result fan-out**

Create `run_bounded(items, fn, max_workers)` that returns ordered successes and per-item `UpstreamError` failures. Use it in search-all, sweep dates, and comparison. Do not sleep after requests.

- [ ] **Step 6: Replace synchronous sweep with jobs**

`POST /api/cheapest-dates` returns:

```json
{ "jobId": "opaque-id", "status": "queued", "statusUrl": "/api/sweeps/opaque-id" }
```

with status 202. `GET /api/sweeps/<id>` returns the manager snapshot. `DELETE` returns 202 after setting cancellation. The runner parallelizes destinations four at a time for each sampled date, updates progress after each completion, checks cancellation between dates and futures, and always finalizes state.

- [ ] **Step 7: Add health, CORS boundaries, request IDs, and structured errors**

`GET /api/health` returns status, version, `primp` profile, cache sizes, and active job state without making an upstream call. Register handlers for `ValidationProblem`, `JobConflict`, unknown jobs, and `UpstreamError` with stable JSON `{error: {code, message, fields}}`.

Generate or accept a valid `X-Request-ID`, include it in response headers and structured log records, and include the job ID in sweep logs. Do not log complete upstream response bodies. Do not enable CORS when `HOTEL_FINDER_ALLOWED_ORIGINS` is unset; when set, split its comma-separated exact origins and apply them only to `/api/*`. Add route tests proving an unlisted origin receives no allow-origin header and a configured private origin does.

- [ ] **Step 8: Run backend gates and commit**

Run:

```powershell
python -m pytest -q
python -m ruff check app.py hotel_finder tests scripts
python -m compileall -q app.py hotel_finder
```

Expected: all tests PASS, Ruff exits 0, compileall exits 0.

```bash
git add app.py requirements.txt hotel_finder tests
git commit -m "feat: harden hotel APIs and background sweeps"
```

---

### Task 6: Build Accessible Frontend Components

**Files:**
- Create: `frontend/src/components/SearchPanel.jsx`
- Create: `frontend/src/components/SearchPanel.test.jsx`
- Create: `frontend/src/components/DateSweep.jsx`
- Create: `frontend/src/components/DateSweep.test.jsx`
- Create: `frontend/src/components/SweepProgress.jsx`
- Create: `frontend/src/components/SweepProgress.test.jsx`
- Create: `frontend/src/components/HotelResults.jsx`
- Create: `frontend/src/components/HotelResults.test.jsx`
- Create: `frontend/src/components/DateChart.jsx`
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: Task 4 date/pricing utilities and normalized result/job objects from Task 5.
- Produces: controlled presentational components; no component performs raw `fetch`.

- [ ] **Step 1: Write default and submission-guard component tests**

```javascript
it('renders future defaults and derives five nights', () => {
  render(<SearchPanel now={new Date(2026, 7, 8, 12)} loading={false} onSubmit={vi.fn()} />)
  expect(screen.getByLabelText('Check-in')).toHaveValue('2026-08-09')
  expect(screen.getByLabelText('Check-out')).toHaveValue('2026-08-14')
  expect(screen.getByText('5 nights')).toBeInTheDocument()
})

it('does not submit blank single-location sweep', async () => {
  const onSubmit = vi.fn()
  render(<DateSweep destinations={DESTINATIONS} onSubmit={onSubmit} />)
  await user.click(screen.getByRole('button', { name: 'Single location' }))
  expect(screen.getByRole('button', { name: 'Find cheapest dates' })).toBeDisabled()
})
```

- [ ] **Step 2: Verify tests fail, then implement controlled components**

Run: `npm test -- src/components/SearchPanel.test.jsx src/components/DateSweep.test.jsx`  
Expected: FAIL because components do not exist.

Use semantic `fieldset`/`legend`, `aria-pressed`, date `min`, inline validation, and derived nights. Default sweep uses six samples over 90 days.

- [ ] **Step 3: Write job-progress and cancellation tests**

Render `SweepProgress` with a running snapshot and assert a semantic `<progress>` value, live text, partial date summaries, warnings, and an enabled Cancel button that calls `onCancel(jobId)` once.

- [ ] **Step 4: Write result correctness tests**

Assert dynamic `4-star` copy for a threshold of four, absence of `nulls` when timing is missing, stable same-name rows in different locations, identical best-rate use for sorting/display, `Estimated flight RT` headers, a table caption, and no Room Quality control.

- [ ] **Step 5: Implement results and styles**

`HotelResults` accepts `{results, query, elapsedSeconds, warnings, nights}` and delegates pricing to Task 4. External Google links retain `target="_blank" rel="noreferrer"`. Add visible `:focus-visible`, live errors/status, accessible toggle states, table caption, and a textual date-chart summary.

- [ ] **Step 6: Run and commit**

Run: `npm test -- src/components`  
Expected: PASS.

```bash
git add frontend/src/components frontend/src/App.css
git commit -m "feat: add robust accessible hotel controls and results"
```

---

### Task 7: Integrate Frontend Operations and Background Sweeps

**Files:**
- Rewrite: `frontend/src/App.jsx`
- Create: `frontend/src/App.test.jsx`
- Modify: `frontend/src/main.jsx`
- Modify: `frontend/eslint.config.js`
- Modify: `frontend/src/test/setup.js`
- Regenerate: `static/**`

**Interfaces:**
- Consumes: Task 4 API/hooks and Task 6 components.
- Produces: the complete browser application and reproducible committed bundle.

- [ ] **Step 1: Write out-of-order response regression tests**

Use deferred promises for two searches. Submit search A, then B; resolve B first and A last. Assert only B's hotel and query dates render. Repeat for provider comparison against a changed result set.

- [ ] **Step 2: Write sweep polling tests**

Fake timers and API functions. Assert: POST 202 starts polling immediately; no overlapping poll is issued; running snapshots update progress; a completed snapshot stops polling and displays results; unmount aborts; Cancel calls DELETE and stops after a cancelled snapshot.

- [ ] **Step 3: Verify integration tests fail**

Run: `npm test -- src/App.test.jsx`  
Expected: FAIL because the current monolith has no checked job API, cancellation, or stale-response ownership.

- [ ] **Step 4: Rewrite App as an orchestrator**

Keep only destinations, current query/result, operation flags, error/warning, sweep job, tabs, and sort state in `App`. All network calls go through `api.js`; all overlapping operations use `useLatestRequest`. Starting an ordinary search clears sweep-only data. Starting a sweep disables conflicting search/compare controls until complete or cancelled. Comparison merges by `stableHotelKey`.

- [ ] **Step 5: Make polling condition-based**

After each `getSweep` completes, schedule the next poll with `setTimeout(750)` only if the job is queued/running. Never use async `setInterval`. Abort and clear the timer in cleanup. Surface polling failure through an accessible operation-specific error.

- [ ] **Step 6: Clean lint and build**

Run:

```powershell
cd frontend
npm run lint
npm test
npm run build:hostable
cd ..
python scripts/verify_static.py
```

Expected: zero lint errors, all frontend tests PASS, build succeeds, verifier exits 0.

- [ ] **Step 7: Commit**

```bash
git add frontend static scripts/verify_static.py
git commit -m "feat: integrate cancellable hotel search interface"
```

---

### Task 8: Deployment, CI, and Operator Documentation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.dockerignore`
- Create: `README.md`
- Modify: `Dockerfile`
- Modify: `render.yaml`
- Modify: `.gitignore`
- Modify: `scripts/verify_static.py`
- Test: `tests/test_repo_contract.py`

**Interfaces:**
- Consumes: all backend/frontend commands from prior tasks.
- Produces: repeatable CI, a small container context, health checking, and complete private-operation documentation.

- [ ] **Step 1: Extend repository-contract tests**

```python
def test_deployment_uses_one_threaded_worker():
    dockerfile = (ROOT / "Dockerfile").read_text("utf-8")
    render = (ROOT / "render.yaml").read_text("utf-8")
    for text in (dockerfile, render):
        assert "--workers" in text and '1' in text
        assert "--threads" in text and '8' in text


def test_ci_runs_every_local_gate():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    for command in ("pytest", "ruff check", "npm ci", "npm run lint", "npm test", "npm run build:hostable", "verify_static.py"):
        assert command in ci
```

- [ ] **Step 2: Verify contract tests fail for missing CI/docs**

Run: `python -m pytest tests/test_repo_contract.py -q`  
Expected: FAIL because CI and README are missing.

- [ ] **Step 3: Add CI**

Use Ubuntu, Python 3.13, and Node 22. Cache pip/npm, install runtime plus dev requirements, run Python tests/Ruff/compileall, run `npm ci`, lint/tests/build, run static verifier, and fail on `git diff --exit-code -- static`.

- [ ] **Step 4: Harden container/deployment files**

Keep `--workers 1 --threads 8 --timeout 120`; add access/error log output to stdout/stderr. Add Docker health check against `/api/health`. `.dockerignore` must exclude Git metadata, worktrees, Python/Node caches, test caches, local environments, and documentation artifacts not needed at runtime.

- [ ] **Step 5: Write operator README**

Document exact Windows setup, local run, test/lint/build commands, dynamic defaults, cache TTLs, one-active-sweep behavior, cancellation, ephemeral restart semantics, source limitations, static flight estimates, price confirmation, deployment commands, and upgrade paths from the design.

- [ ] **Step 6: Run and commit**

Run:

```powershell
python -m pytest tests/test_repo_contract.py -q
python -m ruff check .
cd frontend
npm run lint
npm test
npm run build:hostable
cd ..
python scripts/verify_static.py
git diff --exit-code -- static
```

Expected: all gates PASS and generated assets match committed files.

```bash
git add .github .dockerignore .gitignore Dockerfile render.yaml README.md scripts tests/test_repo_contract.py
git commit -m "chore: add reproducible private deployment gates"
```

---

### Task 9: Full Verification and Live Smoke

**Files:**
- Modify only if a failing regression test proves a defect in an earlier task.
- Verify: entire repository and browser UI.

**Interfaces:**
- Consumes: the complete hardened edition.
- Produces: fresh evidence for handoff and a clean integration branch.

- [ ] **Step 1: Run the complete static suite**

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest --cov=app --cov=hotel_finder --cov-report=term-missing
python -m ruff check .
python -m compileall -q app.py hotel_finder
cd frontend
npm ci --no-audit --no-fund
npm run lint
npm test
npm run build:hostable
cd ..
python scripts/verify_static.py
git diff --exit-code -- static
```

Expected: zero failures/errors and no generated diff.

- [ ] **Step 2: Run deterministic API integration checks**

Use Flask's test client with mocked upstream calls to verify health, dynamic backend defaults, search, partial search-all, 202 sweep creation, progress, cancellation, completed result, comparison, structured validation, and URL rejection.

- [ ] **Step 3: Run a bounded live smoke dated from the current day**

Search Bangkok for tomorrow through five nights later at five stars; require at least one well-formed result but do not assert a specific hotel or price. Compare one returned Google entity and record provider count/warnings. Run a one-location/one-sample sweep and wait conditionally for completion with a 60-second test deadline.

- [ ] **Step 4: Verify the actual browser UI**

Launch the Flask app without debug/reloader, open it in the in-app browser, and verify:

- future check-in/check-out defaults;
- five derived nights;
- no console errors;
- live search results;
- partial warnings area;
- background sweep progress and cancellation control;
- provider comparison;
- keyboard-visible focus and dynamic star labels.

- [ ] **Step 5: Inspect final state and commit any proven correction**

Run:

```powershell
git diff --check
git status --short
git log --oneline --decorate -12
```

If verification required a correction, its regression test must have failed first and the correction gets a focused commit. Otherwise do not create an empty commit.

- [ ] **Step 6: Final handoff evidence**

Report exact test counts, live-smoke date/query, measured cold and cached timings, warnings/limitations, changed commits, branch name, and any external deployment check that remains unexecuted.
