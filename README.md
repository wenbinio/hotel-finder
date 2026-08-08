# Hotel Finder: private personal edition

Hotel Finder is a single-instance, private tool for comparing future hotel stays. It searches Google Hotels and optional provider pages, then displays hotel prices alongside **static round-trip flight estimates**. Treat every price and availability result as advisory: open the provider and confirm the final price, taxes, room terms, and availability before booking.

This application is deliberately not a public, multi-user service. It keeps jobs and caches in one process's memory so it must run as one Gunicorn worker with eight request threads.

An ordinary Render web service URL is publicly reachable. The included deployment configuration does not add authentication, and CORS is not authentication. Before any remote exposure, this private edition requires an authentication-capable reverse proxy or access gateway whose protection cannot be bypassed through the origin, or a service reachable only on a private network. Otherwise, keep it local or VPN-only.

## Windows setup

Install Python 3.13 and Node.js 22.12 or newer, then run these commands from the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
Set-Location frontend
npm ci --no-audit --no-fund
Set-Location ..
```

If PowerShell blocks activation, use the current-process-only setting and try again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

## Run locally

Build the committed hostable frontend, then start the Flask service:

```powershell
Set-Location frontend
npm run build:hostable
Set-Location ..
python app.py
```

Open <http://127.0.0.1:5001/>. For frontend-only development, run `npm run dev` inside `frontend`; Vite proxies `/api` to the local Flask service on port 5001.

The ordinary search defaults are computed when the page opens: check-in is tomorrow and checkout is five nights later. A cheapest-date sweep defaults to tomorrow through 90 days later, sampling six one-night stays.

## Local quality gates

Run the same gates used in CI after changing source or generated assets:

```powershell
python -m pytest --cov=app --cov=hotel_finder
python -m ruff check .
python -m compileall -q app.py hotel_finder
Set-Location frontend
npm run lint
npm test
npm run build:hostable
Set-Location ..
python scripts/verify_static.py --check
git diff --exit-code -- static
```

`npm run build:hostable` writes the deployment bundle to `static/`. That output is committed; the final diff check prevents source and bundle from drifting.

## Operating limits and data behavior

- Successful hotel searches are cached for 10 minutes. Provider comparisons are cached for 15 minutes, while genuine empty hotel results are cached for 60 seconds. Transport and provider failures are not cached as empty results.
- At most four upstream destination searches run at once by default. The `HOTEL_FINDER_UPSTREAM_CONCURRENCY` environment variable may set a private deployment to a value from 1 through 6; values outside that range are clamped to the supported limit.
- Only one date sweep can be active. The client can request cooperative cancellation; a currently running upstream request is allowed to finish, and the worker stops before scheduling later work.
- A sweep is capped at 200 logical hotel searches. Completed, failed, and cancelled sweep records remain for 30 minutes, with no more than eight retained records.
- Jobs and caches are intentionally ephemeral. Restarting this service clears them, including active sweep progress and completed results.

Google Hotels and provider markup can change, rate-limit, or block requests. Results may therefore be partial or unavailable, and provider comparison is limited to validated Google Hotels entity pages. The flight amounts are static estimates, not live fares.

## Environment variables and CORS

Only these environment variables are used for normal deployment:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `5001` locally | HTTP port. Render supplies this automatically. |
| `HOTEL_FINDER_UPSTREAM_CONCURRENCY` | `4` | Upstream destination concurrency; supported range is 1–6. |
| `HOTEL_FINDER_ALLOWED_ORIGINS` | unset | Comma-separated, exact private origins allowed to call `/api/*`. |

Leave `HOTEL_FINDER_ALLOWED_ORIGINS` unset for the same-origin browser default. If a private frontend has a different origin, set only its exact HTTPS origin, for example `https://hotel.example.internal`, and do not use `*`. CORS controls which browser origins may read API responses; it is not access control and does not stop direct requests to the service.

## Protected remote deployment prerequisites

The included `render.yaml` describes the required single threaded Gunicorn worker, but it does not make Render private. Do not create or update a remote deployment until an authentication-capable reverse proxy or access gateway protects the origin, or the service is restricted to a private network. Verify that the ordinary service URL cannot bypass that protection. If those controls are unavailable, use the local instructions above and connect only through your VPN.

After those prerequisites are in place, commit the generated `static/` bundle and push the protected deployment branch:

```powershell
git status
git push origin main
```

Create the service from the repository blueprint, confirm that it uses `render.yaml`, and set only the environment variables above when needed. Render's health check is `/api/health`. Test that endpoint through the protected URL, then prove an unauthenticated request cannot reach the application directly before running a browser smoke test.

To test the same runtime shape with Docker:

```powershell
docker build --tag hotel-finder:local .
docker run --rm --publish 127.0.0.1:5001:5001 hotel-finder:local
```

Then request <http://127.0.0.1:5001/api/health>. The container health check uses that endpoint as well. Docker and Render deployment validate only service health; still perform a future-date search and confirm a displayed provider price before relying on it.

## Troubleshooting

- **`py -3.13` is unavailable:** install Python 3.13, reopen PowerShell, and recreate `.venv`.
- **`npm ci` fails:** use Node 22.12 or newer and delete only `frontend/node_modules` before retrying `npm ci`.
- **The page is old after a frontend edit:** run `npm run build:hostable`, then rerun `python scripts/verify_static.py --check` and inspect the `static/` diff before committing it.
- **A sweep reports busy or disappears:** only one sweep can run; a restart clears the in-memory registry. Start a new bounded sweep after the existing one completes or is cancelled.
- **Searches are empty or partial:** this can be an upstream timeout, rate limit, changed markup, or genuine lack of inventory. Retry later and verify directly with the provider; do not interpret an upstream failure as a price.
- **Browser CORS errors:** retain same-origin hosting, or set `HOTEL_FINDER_ALLOWED_ORIGINS` to the exact private frontend origin and redeploy.

## Future upgrade paths

1. For durable personal history and jobs, replace the in-memory job/cache storage with SQLite while retaining the API contracts.
2. Add Playwright browser checks and provider contract monitors to detect markup or workflow changes early.
3. Add Redis and multiple workers only when state must be shared across instances; do not do this for the current single-instance deployment.
4. Put authentication and a reverse proxy in front of the service before any remote exposure.
5. Add provider APIs, locales, currency handling, and real flight pricing as separately tested product work.
