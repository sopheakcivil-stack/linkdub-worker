-- Keep one active job per source/language so repeated clicks cannot multiply work.

with ranked as (
  select
    id,
    row_number() over (
      partition by md5(source_url), target_language
      order by (status = 'processing') desc, created_at asc
    ) as duplicate_rank
  from public.linkdub_jobs
  where status in ('queued', 'processing')
)
update public.linkdub_jobs as jobs
set status = 'canceled',
    stage = 'Duplicate request canceled',
    error = 'An earlier request for this video and language is already processing.',
    finished_at = now()
from ranked
where jobs.id = ranked.id
  and ranked.duplicate_rank > 1;

create unique index if not exists linkdub_jobs_active_source_language_uidx
  on public.linkdub_jobs ((md5(source_url)), target_language)
  where status in ('queued', 'processing');
