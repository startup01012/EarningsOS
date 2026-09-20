# EarningsOS earnings-event data quality checkpoint

## Current diagnosis

The repository's `EarningsEvent` model existed, but the Git repository did not contain an earnings-ingestion/event-building script or tracked earnings dataset. The processed parquet/CSV data is excluded by `.gitignore`.

The API was selecting the latest `EarningsEvent` by `event_date` and using the request/current timestamp as the intelligence boundary when no `as_of` was supplied. That made the default baseline boundary independent of the actual result announcement and allowed post-result news to be included.

The canonical model is now:
- one event per `symbol + period_ended`
- `event_key = SYMBOL_YYYY-MM-DD`
- `result_announcement_datetime` is the information boundary
- related filings are documents attached to the event
- duplicate ingestion is removed by deterministic `document_id`
- missing reporting periods are flagged instead of inferred from announcement timing
- document classification is deterministic and based on document type/text/URL
- no fixed day-window is used to assign a reporting period

## Consolidated status

Any future consolidated/non-consolidated filtering must use exact normalized matching. A substring check such as `str.contains("consolidated")` is invalid because `Non-Consolidated` contains the same substring.

## ADANIENT case

The repository does not currently contain the historical source rows needed to independently verify `ADANIENT_2024-03-31` or reconstruct its period from original NSE metadata. The new importer therefore refuses to invent a period when `period_ended` is missing. It can recover an explicit period only when the source text contains a clear phrase such as `quarter ended 31/03/2024`.

A source dataset containing the original filing metadata is required to rebuild the historical event tables and verify the ADANIENT case end-to-end.

## Remaining external step

The Neon database must be populated from the real historical source dataset. The importer supports local parquet/CSV input and PostgreSQL persistence via `--persist`. The repository does not store the source data because historical data files are gitignored.
