create table if not exists public.tripadvisor_keys (
    hotel_name text primary key,
    ta_key     text not null,
    updated_at timestamptz not null default now()
);

create index if not exists tripadvisor_keys_lower_name_idx
    on public.tripadvisor_keys (lower(hotel_name));

alter table public.tripadvisor_keys enable row level security;

-- Anon role only needs read access — writes happen via the service role from
-- the seed script and (eventually) an admin tool.
create policy "anon read tripadvisor_keys"
    on public.tripadvisor_keys for select
    to anon
    using (true);
