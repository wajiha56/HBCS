"""
Honeybee x ShafiShifa ingestion API v3.1  (FastAPI + Pydantic v2 + Supabase)

Run:   uvicorn ingestion_api_v3_1:app --reload
Env:   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_JWT_SECRET
       AREA_LAT_MIN/MAX, AREA_LNG_MIN/MAX  (set a tight box per target area)
Deps:  pip install fastapi uvicorn pydantic supabase pyjwt

Security model
- Identity always comes from the verified JWT ("sub"), never from query/body fields.
- Reviewer role comes from app_metadata.role == "reviewer" (only the service role can set it).
- If your Supabase project uses asymmetric JWT signing keys, verify via JWKS instead of HS256.
- Add rate limiting at the reverse proxy (Caddy/nginx) before launch.
"""
import os
import re
import time
from collections import defaultdict, deque
from datetime import date
from typing import List, Literal, Optional
from uuid import UUID

import jwt  # PyJWT
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from postgrest.exceptions import APIError
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator
from supabase import create_client


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


SUPABASE = create_client(_env("SUPABASE_URL"), _env("SUPABASE_SERVICE_ROLE_KEY"))
JWT_SECRET = _env("SUPABASE_JWT_SECRET")
LAT_MIN, LAT_MAX = float(os.environ.get("AREA_LAT_MIN", 23.0)), float(os.environ.get("AREA_LAT_MAX", 37.5))
LNG_MIN, LNG_MAX = float(os.environ.get("AREA_LNG_MIN", 60.0)), float(os.environ.get("AREA_LNG_MAX", 78.0))

app = FastAPI(title="Honeybee x ShafiShifa Ingestion API", version="3.1")
app.add_middleware(  # set ALLOWED_ORIGINS="https://honeybeecodestudio.com" in production
    CORSMiddleware,
    allow_origins=[o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
bearer = HTTPBearer(auto_error=True)


# ------------------------------------------------------------------ auth
_HITS: dict = defaultdict(deque)
RATE_LIMIT, RATE_WINDOW = int(os.environ.get("RATE_LIMIT_PER_MIN", 60)), 60


def _rate_limit(key: str) -> None:
    """In-process sliding window per user. Fine for one worker; use Redis/proxy limits if you scale out."""
    now, q = time.monotonic(), _HITS[key]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many requests; slow down")
    q.append(now)


def current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    try:
        user = jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"], audience="authenticated")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    _rate_limit(user["sub"])
    return user


def require_reviewer(user: dict = Depends(current_user)) -> dict:
    if (user.get("app_metadata") or {}).get("role") != "reviewer":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Reviewer role required")
    return user


def require_admin(user: dict = Depends(current_user)) -> dict:
    if (user.get("app_metadata") or {}).get("role") != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return user


TERMS_VERSION = os.environ.get("TERMS_VERSION", "2026-10-v1")  # must match legal_content.json "version"


# ------------------------------------------------------------------ schemas
class Strict(BaseModel):
    """Unknown keys are rejected, matching the 'no extra key' rubric rule."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Source(Strict):
    title: str = Field(min_length=2, max_length=120)
    url: HttpUrl


def _words(text: str) -> int:
    return len(text.split())


class HealthcareNode(Strict):
    node_name: str = Field(min_length=3, max_length=150)
    node_type: Literal["hospital", "clinic", "lab", "pharmacy", "allied_health", "pharma_supplier"]
    ownership: Literal["government", "private", "charitable", "unknown"]
    district: str = Field(min_length=2, max_length=80)
    tehsil: str = Field(min_length=2, max_length=80)
    uc: Optional[str] = Field(default=None, max_length=80)
    area: str = Field(min_length=2, max_length=100)
    address: str = Field(min_length=5, max_length=250)
    lat: Optional[float] = None
    lng: Optional[float] = None
    coords_source: Optional[Literal["google_maps_pin", "official_listing"]] = None
    services: List[str] = Field(min_length=1, max_length=20)
    source_url: HttpUrl
    last_verified_date: date
    verification_status: Literal["unverified"] = "unverified"

    @model_validator(mode="after")
    def _checks(self):
        if (self.lat is None) != (self.lng is None):
            raise ValueError("lat and lng must both be set or both be null")
        if self.lat is not None:
            if self.coords_source is None:
                raise ValueError("coords_source is required when coordinates are given")
            if not (LAT_MIN <= self.lat <= LAT_MAX and LNG_MIN <= self.lng <= LNG_MAX):
                raise ValueError("coordinates fall outside the configured target-area box")
        if self.last_verified_date > date.today():
            raise ValueError("last_verified_date cannot be in the future")
        if any(not s.strip() or len(s) > 80 for s in self.services):
            raise ValueError("services must be non-empty strings of at most 80 characters")
        return self


class HeroProfile(Strict):
    name: str = Field(min_length=2, max_length=100)
    field: str = Field(min_length=2, max_length=100)
    birth_year: Optional[int] = Field(default=None, ge=1500, le=date.today().year)
    death_year: Optional[int] = Field(default=None, ge=1500, le=date.today().year)
    relation_to_area: Literal["born", "lived", "worked", "other"]
    relation_source_url: HttpUrl
    summary_short: str = Field(min_length=10, max_length=500)
    key_achievements: List[str] = Field(min_length=1, max_length=10)
    sources: List[Source] = Field(min_length=1, max_length=10)
    image_type: Literal["stylized_illustration", "archival_photo_link"]
    ai_generated: bool = True
    verification_status: Literal["unverified"] = "unverified"

    @model_validator(mode="after")
    def _checks(self):
        if _words(self.summary_short) > 60:
            raise ValueError("summary_short must be at most 60 words")
        if self.birth_year and self.death_year and self.death_year < self.birth_year:
            raise ValueError("death_year cannot be earlier than birth_year")
        return self


class SpecialtyGuide(Strict):
    specialty: str = Field(min_length=3, max_length=100)
    category: Literal["medical", "surgical"]
    plain_language_intro: str = Field(min_length=20, max_length=800)
    when_to_see: List[str] = Field(min_length=1, max_length=8)
    triage_stage: int = Field(ge=1, le=8)
    ai_generated: bool = True
    verification_status: Literal["unverified"] = "unverified"


class MediaWork(Strict):
    """Image / audio / video / portfolio work (modules 1, 3, 5, 7, 8, 10)."""
    title: str = Field(min_length=3, max_length=120)
    work_url: HttpUrl
    prompt_used: str = Field(min_length=10, max_length=2000)
    ai_label_confirmed: Literal[True]
    disclaimer_included: Literal[True]
    verification_status: Literal["unverified"] = "unverified"


class Enrollment(Strict):
    full_name: str = Field(min_length=3, max_length=100)
    district: str = Field(min_length=2, max_length=80)
    age_group: Literal["adult", "minor_with_guardian"]
    guardian_name: Optional[str] = Field(default=None, min_length=3, max_length=100)
    guardian_relationship: Optional[Literal["parent", "legal_guardian"]] = None
    terms_version: str = Field(max_length=40)
    accepts_rules: Literal[True]
    accepts_privacy: Literal[True]

    @model_validator(mode="after")
    def _checks(self):
        if self.terms_version != TERMS_VERSION:
            raise ValueError("terms_version is out of date; reload the page and accept the current terms")
        if self.age_group == "minor_with_guardian":
            if not self.guardian_name or not self.guardian_relationship:
                raise ValueError("guardian_name and guardian_relationship are required under 18")
        elif self.guardian_name or self.guardian_relationship:
            raise ValueError("guardian details apply only to students under 18")
        return self


class Batch(Strict):
    module_number: int = Field(ge=0, le=10, description="0 = Data Bounty")


class NodeBatch(Batch):
    records: List[HealthcareNode] = Field(min_length=1, max_length=10)


class HeroBatch(Batch):
    records: List[HeroProfile] = Field(min_length=1, max_length=5)


class MediaBatch(Batch):
    records: List[MediaWork] = Field(min_length=1, max_length=3)


class GuideBatch(Batch):
    records: List[SpecialtyGuide] = Field(min_length=1, max_length=5)


class ReviewDecision(Strict):
    submission_id: UUID
    action: Literal["approved", "rejected"]
    multiplier: Literal[0.0, 1.0, 1.5]
    bonus_credits: int = Field(default=0, ge=0, le=20)
    publish: bool = False
    reviewer_notes: str = Field(min_length=5, max_length=500)

    @model_validator(mode="after")
    def _consistent(self):
        if self.action == "rejected" and (self.multiplier != 0 or self.bonus_credits or self.publish):
            raise ValueError("a rejected submission must have multiplier 0, no bonus, and publish=false")
        if self.action == "approved" and self.multiplier == 0:
            raise ValueError("an approved submission needs multiplier 1.0 or 1.5")
        return self


# ------------------------------------------------------------------ helpers
# Which submission kind each module accepts (module 0 = Data Bounty accepts the three data kinds).
MODULE_KINDS = {1: "media_work", 2: "hero_profile", 3: "media_work", 4: "healthcare_node", 5: "media_work",
                6: "specialty_guide", 7: "media_work", 8: "media_work", 9: "healthcare_node", 10: "media_work"}
MIN_RECORDS = {9: 3}          # Module 9 requires a batch of 3+ nodes
MAX_PENDING_PER_STUDENT = 5   # abuse control: reviewers are the bottleneck


def _guard(kind: str, module: int, student_id: str, n_records: int) -> None:
    """Server-side enforcement of what the UI only suggests: kind match, sequential unlock, pending cap."""
    if module == 0:
        if kind == "media_work":
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Data Bounty accepts nodes, heroes or guides only")
    elif MODULE_KINDS[module] != kind:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Module {module} does not accept this submission type")
    if n_records < MIN_RECORDS.get(module, 1):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Module {module} needs at least {MIN_RECORDS[module]} records")
    if module >= 2:
        prev = (SUPABASE.table("credit_ledger").select("id").eq("student_id", student_id)
                .eq("module_number", module - 1).gt("multiplier", 0).limit(1).execute().data)
        if not prev:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Module {module - 1} must be approved first")
    pend = SUPABASE.table("staging_submissions").select("id").eq("student_id", student_id).eq("status", "pending").execute().data
    if len(pend) >= MAX_PENDING_PER_STUDENT:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many submissions awaiting review; wait for a decision")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _stage(kind: str, module: int, student_id: str, records, key_fn) -> dict:
    _guard(kind, module, student_id, len(records))
    accepted, duplicates = [], []
    for rec in records:
        row = {
            "student_id": student_id,
            "kind": kind,
            "module_number": module,
            "payload": rec.model_dump(mode="json"),
            "dedupe_key": key_fn(rec),
        }
        try:
            res = SUPABASE.table("staging_submissions").insert(row).execute()
            accepted.append(res.data[0]["id"])
        except APIError as exc:
            if getattr(exc, "code", None) == "23505":  # unique_violation
                duplicates.append(row["dedupe_key"])
            else:
                raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Database error while staging submission")
    return {"accepted_ids": accepted, "duplicates_skipped": duplicates}


# Plain `def` endpoints: FastAPI runs them in a threadpool, so the sync Supabase client won't block the loop.
@app.get("/api/v1/me")
def me(user: dict = Depends(current_user)):
    res = (SUPABASE.table("students").select("full_name,tier,verified_credits")
           .eq("id", user["sub"]).limit(1).execute())
    if not res.data:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student profile not found")
    return res.data[0]


@app.post("/api/v1/enroll", status_code=status.HTTP_201_CREATED)
def enroll(body: Enrollment, user: dict = Depends(current_user)):
    try:
        SUPABASE.rpc("enroll_student", {
            "p_id": user["sub"], "p_name": body.full_name, "p_email": user.get("email"),
            "p_district": body.district, "p_age_group": body.age_group,
            "p_guardian_name": body.guardian_name, "p_guardian_rel": body.guardian_relationship,
            "p_terms_version": body.terms_version}).execute()
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":
            raise HTTPException(status.HTTP_409_CONFLICT, "Already enrolled")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Database error while enrolling")
    return {"status": "enrolled"}


@app.get("/api/v1/terms-version")
def terms_version():
    return {"version": TERMS_VERSION}


@app.get("/api/v1/me/export")
def export_my_data(user: dict = Depends(current_user)):
    """Right of access: everything we hold about the caller, as JSON."""
    uid = user["sub"]
    t = lambda name, cols: SUPABASE.table(name).select(cols).eq("student_id" if name != "students" else "id", uid).execute().data
    return {
        "profile": t("students", "full_name,email,city,age_group,tier,verified_credits"),
        "consents": t("enrollment_consents", "terms_version,age_group,guardian_name,guardian_relationship,accepted_at"),
        "submissions": t("staging_submissions", "kind,module_number,status,payload,created_at,reviewed_at,reviewer_notes"),
        "credits": t("credit_ledger", "module_number,base_credits,multiplier,bonus_credits,total_awarded,created_at"),
    }


@app.post("/api/v1/me/deletion-request", status_code=status.HTTP_201_CREATED)
def request_deletion(user: dict = Depends(current_user)):
    try:
        SUPABASE.table("deletion_requests").insert({"student_id": user["sub"]}).execute()
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":
            raise HTTPException(status.HTTP_409_CONFLICT, "A deletion request is already open")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Database error while saving request")
    return {"status": "requested"}


@app.get("/api/v1/admin/deletion-requests")
def list_deletions(admin: dict = Depends(require_admin)):
    return SUPABASE.table("deletion_requests").select("id,student_id,status,requested_at").eq("status", "pending").execute().data


@app.post("/api/v1/admin/deletion-requests/{request_id}/process")
def process_deletion(request_id: UUID, admin: dict = Depends(require_admin)):
    """Anonymises the profile in Postgres. Then delete the auth user in the Supabase dashboard/Admin API."""
    try:
        SUPABASE.rpc("anonymise_student", {"p_request_id": str(request_id)}).execute()
    except APIError as exc:
        msg = str(exc)
        if "request_not_found" in msg:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Request not found")
        if "already_handled" in msg:
            raise HTTPException(status.HTTP_409_CONFLICT, "Request already handled")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Database error while anonymising")
    return {"status": "anonymised"}


@app.get("/api/v1/me/progress")
def progress(user: dict = Depends(current_user)):
    led = SUPABASE.table("credit_ledger").select("module_number,multiplier").eq("student_id", user["sub"]).execute().data
    pend = SUPABASE.table("staging_submissions").select("module_number").eq("student_id", user["sub"]).eq("status", "pending").execute().data
    return {
        "approved_modules": sorted({r["module_number"] for r in led if r["module_number"] >= 1 and float(r["multiplier"]) > 0}),
        "pending_modules": sorted({r["module_number"] for r in pend if r["module_number"] >= 1}),
    }


@app.post("/api/v1/submissions/media", status_code=status.HTTP_201_CREATED)
def submit_media(body: MediaBatch, user: dict = Depends(current_user)):
    return _stage("media_work", body.module_number, user["sub"], body.records, lambda r: _norm(str(r.work_url)))


@app.post("/api/v1/submissions/nodes", status_code=status.HTTP_201_CREATED)
def submit_nodes(body: NodeBatch, user: dict = Depends(current_user)):
    return _stage("healthcare_node", body.module_number, user["sub"], body.records,
                  lambda r: f"{_norm(r.node_name)}|{_norm(r.district)}|{_norm(r.tehsil)}")


@app.post("/api/v1/submissions/heroes", status_code=status.HTTP_201_CREATED)
def submit_heroes(body: HeroBatch, user: dict = Depends(current_user)):
    return _stage("hero_profile", body.module_number, user["sub"], body.records, lambda r: _norm(r.name))


@app.post("/api/v1/submissions/specialty-guides", status_code=status.HTTP_201_CREATED)
def submit_guides(body: GuideBatch, user: dict = Depends(current_user)):
    return _stage("specialty_guide", body.module_number, user["sub"], body.records, lambda r: _norm(r.specialty))


_RISKY = re.compile(r"\b(\d+\s?(mg|ml|mcg)|dosage|dose|tablet|capsule|cure[sd]?|guaranteed)\b", re.I)


def _flags(row: dict, all_pending: list) -> list:
    """Rule-based pre-screen. Advisory only: a human still decides."""
    text = str(row["payload"])
    out = []
    if "<TO_VERIFY>" in text or "YYYY-MM-DD" in text:
        out.append("placeholder text left in payload")
    if _RISKY.search(text):
        out.append("possible dosage or cure claim")
    if row["kind"] == "media_work":
        url = _norm(str(row["payload"].get("work_url", "")))
        if url and sum(1 for r in all_pending if r["kind"] == "media_work"
                       and _norm(str(r["payload"].get("work_url", ""))) == url) > 1:
            out.append("same link submitted more than once")
    if (row.get("students") or {}).get("age_group") == "minor_with_guardian" and row["kind"] == "media_work":
        out.append("student is under 18: check the work has no face photos or personal details")
    if row["kind"] == "healthcare_node" and row["payload"].get("lat") is None:
        out.append("no coordinates given")
    return out


@app.get("/api/v1/me/submissions")
def my_submissions(user: dict = Depends(current_user)):
    rows = (SUPABASE.table("staging_submissions")
            .select("id,kind,module_number,status,created_at,reviewed_at,reviewer_notes")
            .eq("student_id", user["sub"]).order("created_at", desc=True).limit(100).execute().data)
    return rows


@app.get("/api/v1/admin/submissions")
def pending_queue(limit: int = 50, status_filter: Literal["pending", "approved", "rejected"] = "pending",
                  kind: Optional[Literal["healthcare_node", "hero_profile", "specialty_guide", "media_work"]] = None,
                  module: Optional[int] = None, reviewer: dict = Depends(require_reviewer)):
    q = (SUPABASE.table("staging_submissions")
         .select("id,kind,module_number,payload,status,created_at,reviewed_at,reviewer_notes,students(full_name,age_group)")
         .eq("status", status_filter))
    if kind:
        q = q.eq("kind", kind)
    if module is not None:
        q = q.eq("module_number", module)
    rows = q.order("created_at", desc=(status_filter != "pending")).limit(min(max(limit, 1), 100)).execute().data
    for r in rows:
        r["flags"] = _flags(r, rows)
    return rows


@app.post("/api/v1/admin/reviews/decision")
def review(decision: ReviewDecision, reviewer: dict = Depends(require_reviewer)):
    """Reviewer id comes from the token. Credits, status and publish happen atomically in Postgres."""
    try:
        res = SUPABASE.rpc("review_submission", {
            "p_submission_id": str(decision.submission_id),
            "p_reviewer": reviewer["sub"],
            "p_action": decision.action,
            "p_multiplier": decision.multiplier,
            "p_bonus": decision.bonus_credits,
            "p_publish": decision.publish,
            "p_notes": decision.reviewer_notes,
        }).execute()
    except APIError as exc:
        msg = str(exc)
        if "already_reviewed" in msg:
            raise HTTPException(status.HTTP_409_CONFLICT, "Submission was already reviewed")
        if "submission_not_found" in msg:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Submission not found")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Database error while recording review")
    return {"status": "recorded", "credits_awarded": res.data}
