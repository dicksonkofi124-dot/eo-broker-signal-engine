-- EO Broker Signal Engine: full Supabase schema (safe to re-run).
-- Run in Supabase > SQL Editor. Use the service_role key in .env.

create table if not exists bot_users (
  chat_id text primary key,
  username text,
  created_at timestamptz default now()
);

create table if not exists paper_trades (
  id bigint generated always as identity primary key,
  asset_pair text,
  direction text,
  entry_price numeric,
  exit_price numeric,
  pytorch_output numeric,
  status text default 'PENDING',
  executed_at timestamptz default now()
);

create table if not exists model_state (
  id text primary key,
  state jsonb,
  updated_at timestamptz
);

create table if not exists learning_samples (
  id bigint generated always as identity primary key,
  created_at timestamptz,
  horizon_seconds int,
  features jsonb,
  entry_price numeric,
  exit_price numeric,
  label int,
  predicted_edge numeric
);

create table if not exists prediction_logs (
  id bigint generated always as identity primary key,
  predicted_at timestamptz not null,
  resolved_at timestamptz,
  utc_hour int,
  direction text,
  confidence text,
  entry_price numeric,
  exit_price numeric,
  outcome text,
  correct boolean,
  lgb_proba numeric,
  nn_proba numeric,
  ensemble_proba numeric,
  edge numeric,
  volatility numeric,
  vol_regime text,
  gate_open boolean,
  features jsonb
);

create index if not exists prediction_logs_predicted_at_idx on prediction_logs (predicted_at);
create index if not exists learning_samples_created_at_idx on learning_samples (created_at);
create index if not exists paper_trades_executed_at_idx on paper_trades (executed_at);

grant usage on schema public to service_role;
grant all on all tables in schema public to service_role;
grant all on all sequences in schema public to service_role;
alter default privileges in schema public grant all on tables to service_role;
alter default privileges in schema public grant all on sequences to service_role;
