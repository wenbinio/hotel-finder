// GET /functions/v1/destinations — port of GET /api/destinations.
// Returns the static DESTINATIONS map. Bundled rather than read from the DB
// because the data rarely changes; bumping it requires a function redeploy.

import { jsonResponse, preflight } from "../_shared/cors.ts";
import { DESTINATIONS } from "../_shared/destinations.ts";

Deno.serve((req) => {
  const pre = preflight(req);
  if (pre) return pre;
  if (req.method !== "GET") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }
  return jsonResponse(DESTINATIONS);
});
