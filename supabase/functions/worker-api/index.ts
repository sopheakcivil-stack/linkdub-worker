import "jsr:@supabase/functions-js@2.4.4/edge-runtime.d.ts";
import { createClient } from "jsr:@supabase/supabase-js@2.57.4";
import { createRemoteJWKSet, jwtVerify } from "npm:jose@6.1.0";

const GITHUB_ISSUER = "https://token.actions.githubusercontent.com";
const GITHUB_JWKS = createRemoteJWKSet(
  new URL("https://token.actions.githubusercontent.com/.well-known/jwks"),
);
const EXPECTED_AUDIENCE = "linkdub-worker";
const EXPECTED_REPOSITORY = Deno.env.get("LINKDUB_GITHUB_REPOSITORY") ??
  "sopheakcivil-stack/linkdub-worker";
const BUCKET = "linkdub-results";

const supabase = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
  { auth: { persistSession: false, autoRefreshToken: false } },
);

type Claims = {
  repository?: string;
  ref?: string;
  run_id?: string;
  run_attempt?: string;
  workflow_ref?: string;
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

async function authenticate(req: Request): Promise<{ claims: Claims; workerId: string }> {
  const header = req.headers.get("authorization") ?? "";
  const token = header.startsWith("Bearer ") ? header.slice(7) : "";
  if (!token) throw new Error("Missing GitHub OIDC bearer token");

  const { payload } = await jwtVerify(token, GITHUB_JWKS, {
    issuer: GITHUB_ISSUER,
    audience: EXPECTED_AUDIENCE,
  });
  const claims = payload as Claims;
  if (claims.repository !== EXPECTED_REPOSITORY) {
    throw new Error("OIDC token is not from the LinkDub worker repository");
  }
  if (claims.ref !== "refs/heads/main") {
    throw new Error("Only the main branch may process production jobs");
  }
  if (!claims.run_id || !claims.run_attempt) {
    throw new Error("OIDC token is missing workflow run identity");
  }
  return {
    claims,
    workerId: `github:${claims.run_id}:${claims.run_attempt}`,
  };
}

async function ownsJob(jobId: string, workerId: string): Promise<boolean> {
  const { data, error } = await supabase
    .from("linkdub_jobs")
    .select("id")
    .eq("id", jobId)
    .eq("worker_id", workerId)
    .maybeSingle();
  if (error) throw error;
  return Boolean(data);
}

const UPDATE_FIELDS = new Set([
  "status",
  "stage",
  "progress",
  "error",
  "duration_seconds",
  "output_url",
  "output_path",
  "translated_subtitles_url",
  "translated_subtitles_path",
  "source_subtitles_url",
  "source_subtitles_path",
  "finished_at",
]);

Deno.serve(async (req: Request) => {
  if (req.method === "GET") {
    return json({ ok: true, service: "linkdub-worker-api" });
  }
  if (req.method !== "POST") return json({ error: "Method not allowed" }, 405);

  try {
    const { workerId } = await authenticate(req);
    const body = await req.json();
    const action = String(body.action ?? "");

    if (action === "claim") {
      const { data, error } = await supabase.rpc("claim_linkdub_job", {
        p_worker_id: workerId,
      });
      if (error) throw error;
      return json({ job: data?.[0] ?? null, worker_id: workerId });
    }

    const jobId = String(body.job_id ?? "");
    if (!jobId || !(await ownsJob(jobId, workerId))) {
      return json({ error: "Job is not owned by this workflow run" }, 403);
    }

    if (action === "heartbeat") {
      const { error } = await supabase
        .from("linkdub_jobs")
        .update({ heartbeat_at: new Date().toISOString() })
        .eq("id", jobId)
        .eq("worker_id", workerId);
      if (error) throw error;
      return json({ ok: true });
    }

    if (action === "update") {
      const requested = body.fields ?? {};
      const fields: Record<string, unknown> = {
        heartbeat_at: new Date().toISOString(),
      };
      for (const [key, value] of Object.entries(requested)) {
        if (UPDATE_FIELDS.has(key)) fields[key] = value;
      }
      if (fields.status === "completed" || fields.status === "failed") {
        fields.finished_at = new Date().toISOString();
      }
      const { data, error } = await supabase
        .from("linkdub_jobs")
        .update(fields)
        .eq("id", jobId)
        .eq("worker_id", workerId)
        .select()
        .single();
      if (error) throw error;
      return json({ job: data });
    }

    if (action === "segments") {
      const segments = Array.isArray(body.segments) ? body.segments : [];
      if (segments.length > 10000) return json({ error: "Too many segments" }, 400);

      const { error: deleteError } = await supabase
        .from("linkdub_segments")
        .delete()
        .eq("job_id", jobId);
      if (deleteError) throw deleteError;

      if (segments.length) {
        const rows = segments.map((segment: Record<string, unknown>, index: number) => ({
          job_id: jobId,
          segment_index: index,
          start_seconds: segment.start,
          end_seconds: segment.end,
          source_text: String(segment.source_text ?? ""),
          translated_text: String(segment.translated_text ?? ""),
        }));
        const { error: insertError } = await supabase
          .from("linkdub_segments")
          .insert(rows);
        if (insertError) throw insertError;
      }
      return json({ ok: true, count: segments.length });
    }

    if (action === "upload-url") {
      const kind = String(body.kind ?? "");
      const files: Record<string, { extension: string; contentType: string }> = {
        video: { extension: "mp4", contentType: "video/mp4" },
        translated_subtitles: { extension: "srt", contentType: "application/x-subrip" },
        source_subtitles: { extension: "srt", contentType: "application/x-subrip" },
      };
      const file = files[kind];
      if (!file) return json({ error: "Unsupported artifact kind" }, 400);

      const path = `${jobId}/${kind}.${file.extension}`;
      const { data, error } = await supabase.storage
        .from(BUCKET)
        .createSignedUploadUrl(path, { upsert: true });
      if (error) throw error;
      const { data: publicData } = supabase.storage.from(BUCKET).getPublicUrl(path);
      return json({
        path,
        token: data.token,
        signed_url: data.signedUrl,
        public_url: publicData.publicUrl,
        content_type: file.contentType,
      });
    }

    return json({ error: "Unknown action" }, 400);
  } catch (error) {
    console.error(error);
    const message = error instanceof Error ? error.message : "Unknown worker API error";
    const authError = message.includes("OIDC") || message.includes("bearer token") ||
      message.includes("main branch");
    return json({ error: message }, authError ? 401 : 500);
  }
});

