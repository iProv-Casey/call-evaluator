import argparse
import csv
import datetime as dt
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
import yaml
from openai import OpenAI


CALLRAIL_API_BASE = "https://api.callrail.com/v3"


@dataclass
class ClientConfig:
    name: str
    account_id: str
    company_id: str
    airtable_base_id: Optional[str] = None
    airtable_table_name: Optional[str] = None
    prompt_dir: Optional[str] = None
    system_prompt: Optional[str] = None


def load_clients(config_path: str) -> List[ClientConfig]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    clients = []
    for raw in data.get("clients", []):
        try:
            clients.append(
                ClientConfig(
                    name=str(raw["name"]),
                    account_id=str(raw["account_id"]),
                    company_id=str(raw["company_id"]),
                    airtable_base_id=raw.get("airtable_base_id"),
                    airtable_table_name=raw.get("airtable_table_name"),
                    prompt_dir=raw.get("prompt_dir"),
                    system_prompt=raw.get("system_prompt"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"Missing required client key: {exc} in {raw}") from exc
    if not clients:
        raise ValueError("No clients defined in config.")
    return clients


def select_client(clients: Iterable[ClientConfig], name: str) -> ClientConfig:
    for client in clients:
        if client.name.lower() == name.lower():
            return client
    raise ValueError(f"Client '{name}' not found. Use list-clients to see available options.")


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    headers: Dict[str, str],
    max_attempts: int = 5,
    backoff_seconds: int = 2,
    **kwargs,
) -> requests.Response:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.request(method, url, headers=headers, timeout=60, **kwargs)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(backoff_seconds * attempt)
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Request failed after {max_attempts} attempts: {last_error}")


def fetch_calls(
    session: requests.Session,
    token: str,
    account_id: str,
    company_id: str,
    start_date: str,
    end_date: str,
) -> List[Dict]:
    headers = {"Authorization": f"Token token={token}"}
    calls: List[Dict] = []
    page = 1
    while True:
        params = {
            "company_id": company_id,
            "start_date": start_date,
            "end_date": end_date,
            "per_page": 100,
            "page": page,
        }
        url = f"{CALLRAIL_API_BASE}/a/{account_id}/calls.json"
        response = request_with_retry(session, "GET", url, headers=headers, params=params)
        data = response.json()
        page_calls = data.get("calls", [])
        calls.extend(page_calls)
        total_pages = data.get("total_pages") or 1
        if page >= total_pages:
            break
        page += 1
    return calls


def _is_audio_content(content_type: str) -> bool:
    return content_type.startswith("audio/") or content_type == "application/octet-stream"


def extract_call_id_from_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]
    for part in path_parts:
        if part.startswith("CAL"):
            return part
    qs = parse_qs(parsed.query)
    for key in ("call_id", "id", "call"):
        candidates = qs.get(key, [])
        for val in candidates:
            if val.startswith("CAL"):
                return val
    return None


def fetch_recording_url(session: requests.Session, token: str, account_id: str, call_id: str) -> str:
    headers = {"Authorization": f"Token token={token}"}
    url = f"{CALLRAIL_API_BASE}/a/{account_id}/calls/{call_id}/recording.json"
    response = request_with_retry(session, "GET", url, headers=headers, stream=False)
    data = response.json()
    recording_url = data.get("url") or data.get("recording_url") or data.get("recording")
    if not recording_url:
        raise ValueError(f"Recording API did not return a usable URL: {data}")
    return recording_url


def download_recording(session: requests.Session, token: str, url: str) -> bytes:
    headers = {"Authorization": f"Token token={token}"}
    response = request_with_retry(session, "GET", url, headers=headers, stream=True, allow_redirects=True)
    content_type = response.headers.get("Content-Type", "")

    # Some CallRail recording URLs return JSON with a signed audio URL; follow it.
    if "application/json" in content_type or url.endswith(".json"):
        data = response.json()
        recording_url = data.get("url") or data.get("recording_url")
        if not recording_url:
            raise ValueError(f"Recording endpoint returned JSON without audio URL: {data}")
        audio_resp = request_with_retry(
            session,
            "GET",
            recording_url,
            headers=headers,
            stream=True,
            allow_redirects=True,
        )
        audio_type = audio_resp.headers.get("Content-Type", "")
        if not _is_audio_content(audio_type):
            sample = audio_resp.text[:200]
            raise ValueError(f"Expected audio from recording URL, got {audio_type}. Sample: {sample}")
        return audio_resp.content

    if not _is_audio_content(content_type):
        sample = response.text[:200]
        raise ValueError(f"Recording fetch did not return audio (content-type={content_type}). Sample: {sample}")

    return response.content


def download_recording_with_fallback(
    session: requests.Session,
    token: str,
    account_id: str,
    call_id: str,
    recording_url: str,
) -> bytes:
    errors: List[str] = []

    def _try_fetch_audio(url: str, headers: Optional[Dict[str, str]]) -> Optional[bytes]:
        response = session.get(url, headers=headers or {}, timeout=60, allow_redirects=True, stream=True)
        if response.status_code == 401:
            errors.append(f"401 from {url} (auth={'yes' if headers else 'no'}) body={response.text[:200]}")
            return None
        response.raise_for_status()
        ct = response.headers.get("Content-Type", "")
        if not _is_audio_content(ct):
            errors.append(f"Non-audio from {url} (content-type={ct}) sample={response.text[:200]}")
            return None
        return response.content

    def try_download(url: str, headers: Optional[Dict[str, str]]) -> Optional[bytes]:
        try:
            response = session.get(url, headers=headers or {}, timeout=60, allow_redirects=True, stream=True)
            if response.status_code == 401:
                errors.append(f"401 from {url} (auth={'yes' if headers else 'no'}) body={response.text[:200]}")
                return None
            response.raise_for_status()
            ct = response.headers.get("Content-Type", "")
            if "application/json" in ct:
                data = response.json()
                signed_url = data.get("url") or data.get("recording_url")
                if not signed_url:
                    errors.append(f"JSON without audio url from {url}: {data}")
                    return None
                audio = _try_fetch_audio(signed_url, None)
                if audio:
                    return audio
                if headers:
                    return _try_fetch_audio(signed_url, headers)
                return None
            if not _is_audio_content(ct):
                errors.append(f"Non-audio from {url} (content-type={ct}) sample={response.text[:200]}")
                return None
            return response.content
        except Exception as exc:
            errors.append(f"Error fetching {url} (auth={'yes' if headers else 'no'}): {exc}")
            return None

    # 1) Try signed URL with auth header
    audio = try_download(recording_url, {"Authorization": f"Token token={token}"})
    if audio:
        return audio
    # 2) Try signed URL without auth header
    audio = try_download(recording_url, None)
    if audio:
        return audio
    # 3) Try direct API mp3 endpoint
    api_mp3_url = f"{CALLRAIL_API_BASE}/a/{account_id}/calls/{call_id}/recording.mp3"
    audio = try_download(api_mp3_url, {"Authorization": f"Token token={token}"})
    if audio:
        return audio

    raise RuntimeError("Recording download failed. Attempts: " + " | ".join(errors))


def fetch_call_details(session: requests.Session, token: str, account_id: str, call_id: str) -> Dict:
    headers = {"Authorization": f"Token token={token}"}
    url = f"{CALLRAIL_API_BASE}/a/{account_id}/calls/{call_id}.json"
    response = request_with_retry(session, "GET", url, headers=headers, stream=False)
    return response.json()


def load_recordings_from_csv(path: str, url_column: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or url_column not in reader.fieldnames:
            raise ValueError(f"CSV must include column '{url_column}'. Found: {reader.fieldnames}")
        for row in reader:
            recording_url = (row.get(url_column) or "").strip()
            if not recording_url:
                continue
            rows.append({"recording_url": recording_url, "row": row})
    if not rows:
        raise ValueError("CSV contained no recording URLs.")
    return rows


def transcribe_audio(client: OpenAI, audio_bytes: bytes) -> str:
    with io.BytesIO(audio_bytes) as buffer:
        buffer.name = "recording.mp3"
        buffer.seek(0)
        transcription = client.audio.transcriptions.create(model="whisper-1", file=buffer)
    return transcription.text


def ensure_airtable_record(
    session: requests.Session,
    airtable_api_key: str,
    base_id: str,
    table_name: str,
    fields: Dict[str, object],
) -> str:
    url = f"https://api.airtable.com/v0/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {airtable_api_key}", "Content-Type": "application/json"}
    call_id = str(fields["Call ID"]).replace("'", "\\'")
    filter_formula = f"{{Call ID}} = '{call_id}'"
    existing = session.get(url, headers=headers, params={"filterByFormula": filter_formula})
    existing.raise_for_status()
    records = existing.json().get("records", [])

    payload = {"fields": fields}
    if records:
        record_id = records[0]["id"]
        update_url = f"{url}/{record_id}"
        resp = session.patch(update_url, headers=headers, json=payload)
    else:
        resp = session.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json().get("id", "")


def fetch_airtable_records(
    session: requests.Session,
    airtable_api_key: str,
    base_id: str,
    table_name: str,
    filter_formula: Optional[str],
    max_records: Optional[int] = None,
) -> List[Dict]:
    url = f"https://api.airtable.com/v0/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {airtable_api_key}"}
    params: Dict[str, object] = {"pageSize": 100}
    if filter_formula:
        params["filterByFormula"] = filter_formula
    records: List[Dict] = []
    while True:
        resp = session.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        records.extend(data.get("records", []))
        if max_records and len(records) >= max_records:
            return records[:max_records]
        offset = data.get("offset")
        if not offset:
            break
        params["offset"] = offset
    return records


def update_airtable_record_fields(
    session: requests.Session,
    airtable_api_key: str,
    base_id: str,
    table_name: str,
    record_id: str,
    fields: Dict[str, object],
) -> None:
    # Build the Airtable record endpoint URL.
    url = f"https://api.airtable.com/v0/{base_id}/{table_name}/{record_id}"
    # Prepare auth and JSON headers for the PATCH request.
    headers = {"Authorization": f"Bearer {airtable_api_key}", "Content-Type": "application/json"}
    # Send the partial update with the provided field payload.
    resp = session.patch(url, headers=headers, json={"fields": fields})
    # Raise an exception for non-2xx responses.
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Airtable update failed ({resp.status_code}): {resp.text}") from exc


def fetch_airtable_table_fields(
    session: requests.Session,
    airtable_api_key: str,
    base_id: str,
    table_name: str,
) -> Dict[str, str]:
    url = f"https://api.airtable.com/v0/meta/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {airtable_api_key}"}
    response = request_with_retry(session, "GET", url, headers=headers)
    data = response.json()
    tables = data.get("tables", [])
    target = None
    for table in tables:
        if table.get("name") == table_name or table.get("id") == table_name:
            target = table
            break
    if not target:
        available = ", ".join(sorted(table.get("name", "") for table in tables if table.get("name")))
        raise ValueError(f"Airtable table '{table_name}' not found. Available tables: {available}")
    fields: Dict[str, str] = {}
    for field in target.get("fields", []):
        name = field.get("name")
        if name:
            fields[name] = field.get("type", "")
    return fields


def parse_dates(start_date: str, end_date: str) -> (str, str):
    def parse_date(value: str) -> dt.date:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()

    start = parse_date(start_date)
    end = parse_date(end_date)
    if end < start:
        raise ValueError("End date must be on or after start date.")
    return start.isoformat(), end.isoformat()


def build_airtable_fields(client: ClientConfig, call: Dict, transcript: str) -> Dict[str, object]:
    started_at = call.get("started_at")
    return {
        "Client": client.name,
        "Call ID": call.get("id"),
        "Call Started At": started_at,
        "Caller Number": call.get("customer_phone_number"),
        "Tracking Number": call.get("tracking_phone_number"),
        "Duration (s)": call.get("duration"),
        "Recording URL": call.get("recording"),
        "Transcript": transcript,
    }


def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


PROMPT_FILES = {
    "gatekeeper": "gatekeeper.md",
    "evaluator": "evaluator.md",
    "parser": "parser.md",
}


def resolve_prompt_path(client: ClientConfig, config_path: str, prompt_name: str = "evaluator") -> Optional[str]:
    if not client.prompt_dir:
        return None
    config_root = os.path.dirname(os.path.abspath(config_path))
    prompt_dir = client.prompt_dir
    if not os.path.isabs(prompt_dir):
        prompt_dir = os.path.join(config_root, prompt_dir)
    filename = PROMPT_FILES.get(prompt_name)
    if not filename:
        raise ValueError(f"Unknown prompt name '{prompt_name}'. Expected one of: {', '.join(PROMPT_FILES)}.")
    prompt_path = os.path.join(prompt_dir, filename)
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path


def parse_gatekeeper_output(raw: object) -> Tuple[str, Optional[str]]:
    if raw is None:
        raise ValueError("Gatekeeper output is empty.")
    if isinstance(raw, dict):
        payload = raw
        raw_text = json.dumps(payload, ensure_ascii=False)
    elif isinstance(raw, str):
        raw_text = raw.strip()
        if not raw_text:
            raise ValueError("Gatekeeper output is empty.")
        try:
            payload = json.loads(raw_text)
        except Exception as exc:
            raise ValueError(f"Gatekeeper JSON could not be parsed: {exc}")
    else:
        raise ValueError(f"Unsupported gatekeeper payload type: {type(raw)}")

    channel = payload.get("channel")
    if isinstance(channel, str):
        channel = channel.strip() or None
    else:
        channel = None
    return raw_text, channel


def build_eval_input(transcript: str, gatekeeper_json: Optional[str], gatekeeper_channel: Optional[str]) -> str:
    parts: List[str] = []
    if gatekeeper_json:
        parts.append(f"Gatekeeper JSON:\n{gatekeeper_json.strip()}")
    elif gatekeeper_channel:
        parts.append(f"Gatekeeper channel: {gatekeeper_channel}")
    parts.append(f"Transcript:\n{transcript}")
    return "\n\n".join(parts)


def parse_agent_output(text: str) -> Tuple[str, str]:
    json_section = ""
    summary_section = ""

    if "### JSON" in text:
        _, remainder = text.split("### JSON", 1)
        if "### Executive Summary" in remainder:
            json_section, summary_section = remainder.split("### Executive Summary", 1)
        else:
            json_section = remainder
    else:
        json_section = text

    json_section = json_section.strip()
    # Remove fences if present
    if json_section.startswith("```"):
        json_section = re.sub(r"^```[a-zA-Z]*", "", json_section).strip()
        if "```" in json_section:
            json_section = json_section.split("```", 1)[0].strip()

    # Validate JSON
    try:
        parsed = json.loads(json_section)
        json_section = json.dumps(parsed, ensure_ascii=False)
    except Exception as exc:
        raise ValueError(f"Agent JSON section could not be parsed: {exc}")

    summary_section = summary_section.strip()
    if summary_section.startswith("```"):
        summary_section = summary_section.strip("` \n")

    return json_section, summary_section


def parse_eval_json_fields(
    eval_json: Dict[str, object],
    call_type_field: str,
    lead_quality_field: str,
    converted_field: str,
    primary_issue_field: str,
    overall_score_field: str,
    scheduling_efficiency_field: str,
    appointment_explicitly_offered_field: str,
    estimated_revenue_lost_range_field: str,
    estimated_revenue_lost_rationale_field: str,
    failure_mode_tags_field: str,
    strategic_quadrant_field: str,
    confidence_in_assessment_field: str,
) -> Dict[str, object]:
    def as_dict(value: object) -> Dict[str, object]:
        return value if isinstance(value, dict) else {}

    def normalize_select_value(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float, str)):
            text = str(value).strip()
            return text if text else None
        return None

    def normalize_text(value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, (int, float, bool)):
            return str(value)
        return None

    def normalize_bool(value: object) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "yes", "y", "1"):
                return True
            if text in ("false", "no", "n", "0"):
                return False
        return None

    def normalize_multi_select(value: object) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            raw = [item.strip() for item in value.split(",")]
            values = [item for item in raw if item]
            return values or None
        if isinstance(value, list):
            values: List[str] = []
            for item in value:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        values.append(stripped)
            return values or None
        return None

    fields: Dict[str, object] = {}
    call_type = normalize_select_value(eval_json.get("call_type") or eval_json.get("call_classification"))
    if call_type:
        fields[call_type_field] = call_type

    lead_quality = normalize_select_value(
        as_dict(eval_json.get("lead_quality")).get("rating") or eval_json.get("lead_quality")
    )
    if lead_quality:
        fields[lead_quality_field] = lead_quality

    outcome = as_dict(eval_json.get("outcome"))
    converted_raw = outcome.get("converted")
    if converted_raw is None:
        converted_raw = eval_json.get("converted")
    converted_value = normalize_select_value(converted_raw)
    fields[converted_field] = converted_value or "not_applicable"

    primary_issue = normalize_select_value(
        as_dict(eval_json.get("root_cause")).get("primary_issue") or eval_json.get("root_cause_issue")
    )
    if primary_issue:
        fields[primary_issue_field] = primary_issue

    scores = as_dict(eval_json.get("scores") or eval_json.get("scoring"))
    overall = scores.get("overall_score")
    if overall not in (None, ""):
        try:
            fields[overall_score_field] = float(overall)
        except Exception:
            pass

    scheduling_efficiency = scores.get("scheduling_efficiency") or scores.get("scheduling_efficiency_score")
    if scheduling_efficiency not in (None, ""):
        try:
            fields[scheduling_efficiency_field] = float(scheduling_efficiency)
        except Exception:
            pass

    appointment_offer_raw = outcome.get("appointment_explicitly_offered")
    if appointment_offer_raw is None:
        appointment_offer_raw = eval_json.get("appointment_explicitly_offered")
    if appointment_offer_raw is not None:
        appointment_explicitly_offered = normalize_bool(appointment_offer_raw)
        if appointment_explicitly_offered is not None:
            fields[appointment_explicitly_offered_field] = appointment_explicitly_offered

    estimated_revenue_lost = as_dict(eval_json.get("estimated_revenue_lost"))
    revenue_range = normalize_select_value(estimated_revenue_lost.get("range"))
    if revenue_range not in (None, ""):
        fields[estimated_revenue_lost_range_field] = revenue_range
    revenue_rationale = normalize_text(estimated_revenue_lost.get("rationale"))
    if revenue_rationale not in (None, ""):
        fields[estimated_revenue_lost_rationale_field] = revenue_rationale

    failure_mode_tags = eval_json.get("failure_mode_tags")
    if not failure_mode_tags:
        failure_mode_tags = as_dict(eval_json.get("root_cause")).get("failure_mode_tags")
    normalized_tags = normalize_multi_select(failure_mode_tags)
    if normalized_tags:
        fields[failure_mode_tags_field] = normalized_tags

    strategic_quadrant = (
        eval_json.get("strategic_quadrant")
        or eval_json.get("strategic_quadrant_classification")
        or eval_json.get("quadrant_classification")
    )
    strategic_quadrant_value = normalize_select_value(strategic_quadrant)
    if strategic_quadrant_value:
        fields[strategic_quadrant_field] = strategic_quadrant_value

    confidence_in_assessment = normalize_select_value(eval_json.get("confidence_in_assessment"))
    if confidence_in_assessment:
        fields[confidence_in_assessment_field] = confidence_in_assessment

    return fields


def flatten_json(value: object, prefix: str = "") -> Dict[str, object]:
    flattened: Dict[str, object] = {}

    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            nested_prefix = f"{prefix}.{key_text}" if prefix else key_text
            flattened.update(flatten_json(nested, nested_prefix))
        return flattened

    if isinstance(value, list):
        if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            flattened[prefix] = ", ".join("" if item is None else str(item) for item in value)
        else:
            flattened[prefix] = json.dumps(value, ensure_ascii=False)
        return flattened

    flattened[prefix] = value
    return flattened


def evaluate_transcript(
    client: OpenAI,
    model: str,
    prompt: str,
    transcript: str,
) -> Tuple[str, str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
    )
    content = response.choices[0].message.content or ""
    return parse_agent_output(content)


def run_pipeline(args: argparse.Namespace) -> None:
    clients = load_clients(args.config)
    selected_client = select_client(clients, args.client)
    start_date, end_date = parse_dates(args.start_date, args.end_date)
    max_calls = 5 if args.test else None

    callrail_token = os.getenv("CALLRAIL_API_KEY")
    if not callrail_token:
        raise EnvironmentError("CALLRAIL_API_KEY is required.")
    airtable_api_key = os.getenv("AIRTABLE_API_KEY")
    if not args.dry_run and not airtable_api_key:
        raise EnvironmentError("AIRTABLE_API_KEY is required to write to Airtable.")

    airtable_base = args.airtable_base or selected_client.airtable_base_id
    airtable_table = args.airtable_table or selected_client.airtable_table_name
    if not args.dry_run and (not airtable_base or not airtable_table):
        raise ValueError("Airtable base and table are required (set via flags or client config).")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY is required for transcription.")
    openai_client = OpenAI(api_key=openai_key)

    with requests.Session() as session:
        print(f"Fetching calls for {selected_client.name} from {start_date} to {end_date}...")
        calls = fetch_calls(
            session,
            token=callrail_token,
            account_id=selected_client.account_id,
            company_id=selected_client.company_id,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"Found {len(calls)} calls.")

        calls_with_recordings = [c for c in calls if c.get("recording")]
        print(f"{len(calls_with_recordings)} calls have recordings.")

        for index, call in enumerate(calls_with_recordings, start=1):
            if max_calls and index > max_calls:
                print(f"Test mode: processed {max_calls} calls, stopping.")
                break
            print(f"[{index}/{len(calls_with_recordings)}] Downloading recording for call {call.get('id')}...")
            # Always ask CallRail for a fresh signed recording URL to avoid stale access keys.
            recording_url = fetch_recording_url(session, callrail_token, selected_client.account_id, call["id"])
            audio_bytes = download_recording_with_fallback(
                session,
                callrail_token,
                selected_client.account_id,
                call["id"],
                recording_url,
            )

            print("Transcribing...")
            transcript = transcribe_audio(openai_client, audio_bytes)

            if args.dry_run:
                print("Dry run: skipping Airtable write.")
                continue

            fields = build_airtable_fields(selected_client, call, transcript)
            record_id = ensure_airtable_record(
                session,
                airtable_api_key=airtable_api_key,
                base_id=airtable_base,
                table_name=airtable_table,
                fields=fields,
            )
            print(f"Upserted Airtable record {record_id} for call {call.get('id')}.")


def run_csv_pipeline(args: argparse.Namespace) -> None:
    clients = load_clients(args.config)
    selected_client = select_client(clients, args.client)
    max_calls = 5 if args.test else None

    callrail_token = os.getenv("CALLRAIL_API_KEY")
    if not callrail_token:
        raise EnvironmentError("CALLRAIL_API_KEY is required.")
    airtable_api_key = os.getenv("AIRTABLE_API_KEY")
    if not args.dry_run and not airtable_api_key:
        raise EnvironmentError("AIRTABLE_API_KEY is required to write to Airtable.")

    airtable_base = args.airtable_base or selected_client.airtable_base_id
    airtable_table = args.airtable_table or selected_client.airtable_table_name
    if not args.dry_run and (not airtable_base or not airtable_table):
        raise ValueError("Airtable base and table are required (set via flags or client config).")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY is required for transcription.")
    openai_client = OpenAI(api_key=openai_key)

    rows = load_recordings_from_csv(args.csv_path, args.recording_url_column)
    total_rows = len(rows)
    seen_ids = set()

    with requests.Session() as session:
        for index, entry in enumerate(rows, start=1):
            if max_calls and index > max_calls:
                print(f"Test mode: processed {max_calls} calls, stopping.")
                break
            recording_url = entry["recording_url"]
            call_id = None
            if args.call_id_column:
                call_id = (entry["row"].get(args.call_id_column) or "").strip() or None
            if not call_id:
                call_id = extract_call_id_from_url(recording_url)
            if not call_id:
                print(f"[{index}/{total_rows}] Skipping row: could not determine CallRail ID from recording URL.")
                continue
            if call_id in seen_ids:
                print(f"[{index}/{total_rows}] Skipping duplicate call {call_id}.")
                continue
            seen_ids.add(call_id)

            print(f"[{index}/{total_rows}] Processing call {call_id}...")
            try:
                call = fetch_call_details(session, callrail_token, selected_client.account_id, call_id)
            except Exception as exc:
                print(f"Warning: could not fetch call details for {call_id}: {exc}")
                call = {"id": call_id, "recording": recording_url}

            try:
                fresh_recording_url = fetch_recording_url(session, callrail_token, selected_client.account_id, call_id)
            except Exception as exc:
                print(f"Warning: could not fetch fresh recording url for {call_id}: {exc}; using CSV URL")
                fresh_recording_url = recording_url

            try:
                audio_bytes = download_recording_with_fallback(
                    session,
                    callrail_token,
                    selected_client.account_id,
                    call_id,
                    fresh_recording_url,
                )
            except Exception as exc:
                print(f"Failed downloading recording for {call_id}: {exc}")
                continue

            print("Transcribing...")
            transcript = transcribe_audio(openai_client, audio_bytes)

            if args.dry_run:
                print("Dry run: skipping Airtable write.")
                continue

            fields = build_airtable_fields(selected_client, call, transcript)
            record_id = ensure_airtable_record(
                session,
                airtable_api_key=airtable_api_key,
                base_id=airtable_base,
                table_name=airtable_table,
                fields=fields,
            )
            print(f"Upserted Airtable record {record_id} for call {call_id}.")


def run_classify_airtable(args: argparse.Namespace) -> None:
    # Load client configurations and resolve the target client.
    clients = load_clients(args.config)
    selected_client = select_client(clients, args.client)
    # Limit record count in test mode to keep runs fast.
    max_records = 5 if args.test else None

    # Read Airtable credentials and target metadata.
    airtable_api_key = os.getenv("AIRTABLE_API_KEY")
    if not airtable_api_key:
        raise EnvironmentError("AIRTABLE_API_KEY is required.")
    airtable_base = args.airtable_base or selected_client.airtable_base_id
    airtable_table = args.airtable_table or selected_client.airtable_table_name
    if not airtable_base or not airtable_table:
        raise ValueError("Airtable base and table are required (set via flags or client config).")

    # Read OpenAI credentials and create the API client.
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY is required.")
    openai_client = OpenAI(api_key=openai_key)

    # Resolve the gatekeeper prompt from CLI override or client config.
    prompt = None
    if args.prompt_file:
        prompt = load_prompt(args.prompt_file)
    else:
        prompt_path = resolve_prompt_path(selected_client, args.config, prompt_name="gatekeeper")
        if prompt_path:
            prompt = load_prompt(prompt_path)
    if not prompt:
        raise ValueError("Gatekeeper prompt is required. Add prompt_dir to the client config or provide --prompt-file.")

    # Build an Airtable filter formula that skips already-classified rows unless forced.
    if not args.force:
        filter_formula = (
            f"AND({{{args.transcript_field}}}, LEN({{{args.transcript_field}}})>0, "
            f"NOT({{{args.json_field}}}))"
        )
    else:
        filter_formula = f"AND({{{args.transcript_field}}}, LEN({{{args.transcript_field}}})>0)"

    # Fetch candidate records and classify them one by one.
    with requests.Session() as session:
        records = fetch_airtable_records(
            session,
            airtable_api_key=airtable_api_key,
            base_id=airtable_base,
            table_name=airtable_table,
            filter_formula=filter_formula,
            max_records=max_records,
        )
        total = len(records)
        if total == 0:
            print("No Airtable records found matching filter.")
            return

        for idx, record in enumerate(records, start=1):
            fields = record.get("fields", {})
            transcript = fields.get(args.transcript_field)
            if not transcript:
                print(f"[{idx}/{total}] Skipping record {record.get('id')}: missing transcript field.")
                continue

            print(f"[{idx}/{total}] Classifying record {record.get('id')}...")
            try:
                json_text, _ = evaluate_transcript(
                    openai_client,
                    model=args.model,
                    prompt=prompt,
                    transcript=transcript,
                )
            except Exception as exc:
                print(f"Failed to classify record {record.get('id')}: {exc}")
                continue

            try:
                gatekeeper_json, gatekeeper_channel = parse_gatekeeper_output(json_text)
            except Exception as exc:
                print(f"Failed to parse gatekeeper JSON for record {record.get('id')}: {exc}")
                continue

            update_fields: Dict[str, object] = {
                args.json_field: gatekeeper_json,
            }
            if gatekeeper_channel and args.channel_field:
                update_fields[args.channel_field] = gatekeeper_channel
            if args.timestamp_field:
                update_fields[args.timestamp_field] = dt.datetime.now(dt.timezone.utc).isoformat()

            if args.dry_run:
                print(f"Dry run: would update {record.get('id')} with {update_fields}")
                continue

            try:
                update_airtable_record_fields(
                    session,
                    airtable_api_key=airtable_api_key,
                    base_id=airtable_base,
                    table_name=airtable_table,
                    record_id=record["id"],
                    fields=update_fields,
                )
                print(f"Updated record {record.get('id')} with gatekeeper classification.")
            except Exception as exc:
                print(f"Failed to update Airtable for record {record.get('id')}: {exc}")


def run_eval_airtable(args: argparse.Namespace) -> None:
    # Load client configurations and resolve the target client.
    clients = load_clients(args.config)
    selected_client = select_client(clients, args.client)
    # Limit record count in test mode to keep runs fast.
    max_records = 5 if args.test else None

    # Read Airtable credentials and target metadata.
    airtable_api_key = os.getenv("AIRTABLE_API_KEY")
    if not airtable_api_key:
        raise EnvironmentError("AIRTABLE_API_KEY is required.")
    # Allow CLI flags to override client defaults for base/table.
    airtable_base = args.airtable_base or selected_client.airtable_base_id
    airtable_table = args.airtable_table or selected_client.airtable_table_name
    if not airtable_base or not airtable_table:
        raise ValueError("Airtable base and table are required (set via flags or client config).")

    # Read OpenAI credentials and create the API client.
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY is required.")
    openai_client = OpenAI(api_key=openai_key)

    # Resolve the system prompt from CLI override or client config.
    prompt = None
    if args.prompt_file:
        prompt = load_prompt(args.prompt_file)
    elif selected_client.system_prompt:
        prompt = selected_client.system_prompt
    else:
        prompt_path = resolve_prompt_path(selected_client, args.config, prompt_name="evaluator")
        if prompt_path:
            prompt = load_prompt(prompt_path)
    if not prompt:
        raise ValueError(
            "System prompt is required. Add prompt_dir to the client config or provide --prompt-file."
        )
    # Build an Airtable filter formula that skips already-evaluated rows unless forced.
    filter_formula = None
    if not args.force:
        filter_formula = (
            f"AND({{{args.transcript_field}}}, LEN({{{args.transcript_field}}})>0, "
            f"{{{args.gatekeeper_json_field}}}, LEN({{{args.gatekeeper_json_field}}})>0, "
            f"NOT({{{args.json_field}}}))"
        )
    else:
        filter_formula = f"AND({{{args.transcript_field}}}, LEN({{{args.transcript_field}}})>0)"

    # Fetch candidate records and evaluate them one by one.
    with requests.Session() as session:
        records = fetch_airtable_records(
            session,
            airtable_api_key=airtable_api_key,
            base_id=airtable_base,
            table_name=airtable_table,
            filter_formula=filter_formula,
            max_records=max_records,
        )
        total = len(records)
        if total == 0:
            print("No Airtable records found matching filter.")
            return

        for idx, record in enumerate(records, start=1):
            # Pull the transcript field from the record payload.
            fields = record.get("fields", {})
            transcript = fields.get(args.transcript_field)
            if not transcript:
                print(f"[{idx}/{total}] Skipping record {record.get('id')}: missing transcript field.")
                continue
            gatekeeper_raw = fields.get(args.gatekeeper_json_field) if args.gatekeeper_json_field else None
            gatekeeper_channel = fields.get(args.gatekeeper_channel_field) if args.gatekeeper_channel_field else None
            if not gatekeeper_raw and not gatekeeper_channel:
                print(f"[{idx}/{total}] Skipping record {record.get('id')}: missing gatekeeper output.")
                continue

            gatekeeper_json = None
            if gatekeeper_raw:
                try:
                    gatekeeper_json, parsed_channel = parse_gatekeeper_output(gatekeeper_raw)
                    if not gatekeeper_channel:
                        gatekeeper_channel = parsed_channel
                except Exception as exc:
                    print(f"[{idx}/{total}] Skipping record {record.get('id')}: gatekeeper JSON invalid ({exc})")
                    continue

            # Send the transcript to OpenAI for evaluation.
            print(f"[{idx}/{total}] Evaluating record {record.get('id')}...")
            try:
                eval_input = build_eval_input(transcript, gatekeeper_json, gatekeeper_channel)
                json_text, summary_text = evaluate_transcript(
                    openai_client,
                    model=args.model,
                    prompt=prompt,
                    transcript=eval_input,
                )
            except Exception as exc:
                print(f"Failed to evaluate record {record.get('id')}: {exc}")
                continue

            # Skip writes in dry-run mode.
            if args.dry_run:
                print("Dry run: skipping Airtable write.")
                continue

            # Assemble fields for the Airtable update.
            update_fields: Dict[str, object] = {
                args.json_field: json_text,
                args.summary_field: summary_text,
            }
            if args.timestamp_field:
                # Stamp the update time when configured.
                update_fields[args.timestamp_field] = dt.datetime.now(dt.timezone.utc).isoformat()

            # Persist the evaluation back to Airtable.
            try:
                update_airtable_record_fields(
                    session,
                    airtable_api_key=airtable_api_key,
                    base_id=airtable_base,
                    table_name=airtable_table,
                    record_id=record["id"],
                    fields=update_fields,
                )
                print(f"Updated record {record.get('id')} with evaluation.")
            except Exception as exc:
                print(f"Failed to update Airtable for record {record.get('id')}: {exc}")


def run_parse_eval_json(args: argparse.Namespace) -> None:
    # Load client configs and resolve the target client.
    clients = load_clients(args.config)
    selected_client = select_client(clients, args.client)
    # Limit record count in test mode to keep runs fast.
    max_records = 5 if args.test else None

    # Read Airtable credentials and target metadata.
    airtable_api_key = os.getenv("AIRTABLE_API_KEY")
    if not airtable_api_key:
        raise EnvironmentError("AIRTABLE_API_KEY is required.")
    # Allow CLI flags to override client defaults for base/table.
    airtable_base = args.airtable_base or selected_client.airtable_base_id
    airtable_table = args.airtable_table or selected_client.airtable_table_name
    if not airtable_base or not airtable_table:
        raise ValueError("Airtable base and table are required (set via flags or client config).")

    output_csv = args.output_csv
    output_rows: List[Dict[str, object]] = []

    # Build a filter formula that can skip already-parsed rows unless forced.
    if args.force or output_csv:
        filter_formula = f"AND({{{args.eval_json_field}}}, LEN({{{args.eval_json_field}}})>0)"
    else:
        filter_formula = (
            f"AND({{{args.eval_json_field}}}, LEN({{{args.eval_json_field}}})>0, NOT({{{args.call_type_field}}}))"
        )

    # Fetch candidate records and parse their evaluation JSON.
    with requests.Session() as session:
        existing_fields = fetch_airtable_table_fields(
            session,
            airtable_api_key=airtable_api_key,
            base_id=airtable_base,
            table_name=airtable_table,
        )
        records = fetch_airtable_records(
            session,
            airtable_api_key=airtable_api_key,
            base_id=airtable_base,
            table_name=airtable_table,
            filter_formula=filter_formula,
            max_records=max_records,
        )
        total = len(records)
        if total == 0:
            print("No Airtable records found matching filter.")
            return

        for idx, record in enumerate(records, start=1):
            # Pull the evaluation JSON text from the record payload.
            fields = record.get("fields", {})
            eval_json_str = fields.get(args.eval_json_field)
            if not eval_json_str:
                print(f"[{idx}/{total}] Skipping record {record.get('id')}: missing eval JSON.")
                continue

            # Parse the JSON payload (handle nested JSON strings when present).
            print(f"[{idx}/{total}] Parsing evaluation for record {record.get('id')}...")
            try:
                eval_json = json.loads(eval_json_str)
            except Exception as exc:
                print(f"Failed to parse JSON for record {record.get('id')}: {exc}")
                continue
            if isinstance(eval_json, str):
                try:
                    eval_json = json.loads(eval_json)
                except Exception as exc:
                    print(f"Failed to parse nested JSON for record {record.get('id')}: {exc}")
                    continue
            if not isinstance(eval_json, dict):
                print(f"Failed to parse JSON for record {record.get('id')}: expected object, got {type(eval_json)}")
                continue

            if output_csv:
                row: Dict[str, object] = {"Airtable Record ID": record.get("id")}
                for field_name in ("Call ID", "Client", "Gatekeeper Channel"):
                    if field_name in fields:
                        row[field_name] = fields[field_name]
                row.update(flatten_json(eval_json))
                output_rows.append(row)
                continue

            # Map the evaluation JSON into Airtable field updates.
            update_fields = parse_eval_json_fields(
                eval_json,
                call_type_field=args.call_type_field,
                lead_quality_field=args.lead_quality_field,
                converted_field=args.converted_field,
                primary_issue_field=args.primary_issue_field,
                overall_score_field=args.overall_score_field,
                scheduling_efficiency_field=args.scheduling_efficiency_field,
                appointment_explicitly_offered_field=args.appointment_explicitly_offered_field,
                estimated_revenue_lost_range_field=args.estimated_revenue_lost_range_field,
                estimated_revenue_lost_rationale_field=args.estimated_revenue_lost_rationale_field,
                failure_mode_tags_field=args.failure_mode_tags_field,
                strategic_quadrant_field=args.strategic_quadrant_field,
                confidence_in_assessment_field=args.confidence_in_assessment_field,
            )
            update_fields = {key: value for key, value in update_fields.items() if key in existing_fields}
            if not update_fields:
                print(f"[{idx}/{total}] Skipping record {record.get('id')}: no matching Airtable fields found.")
                continue

            # Skip writes in dry-run mode.
            if args.dry_run:
                print(f"Dry run: would update {record.get('id')} with {update_fields}")
                continue

            # Persist parsed fields back to Airtable.
            try:
                update_airtable_record_fields(
                    session,
                    airtable_api_key=airtable_api_key,
                    base_id=airtable_base,
                    table_name=airtable_table,
                    record_id=record["id"],
                    fields=update_fields,
                )
                print(f"Updated record {record.get('id')} with parsed evaluation fields.")
            except Exception as exc:
                print(f"Failed to update Airtable for record {record.get('id')}: {exc}")

    if output_csv:
        if not output_rows:
            print("No rows available to write to CSV.")
            return
        output_keys = []
        seen = set()
        for row in output_rows:
            for key in row.keys():
                if key not in seen:
                    output_keys.append(key)
                    seen.add(key)
        with open(output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=output_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"Wrote {len(output_rows)} rows to {output_csv}.")


def run_check_airtable_schema(args: argparse.Namespace) -> None:
    clients = load_clients(args.config)
    selected_client = select_client(clients, args.client)

    airtable_api_key = os.getenv("AIRTABLE_API_KEY")
    if not airtable_api_key:
        raise EnvironmentError("AIRTABLE_API_KEY is required.")

    airtable_base = args.airtable_base or selected_client.airtable_base_id
    airtable_table = args.airtable_table or selected_client.airtable_table_name
    if not airtable_base or not airtable_table:
        raise ValueError("Airtable base and table are required (set via flags or client config).")

    classify_fields = [
        args.transcript_field,
        args.gatekeeper_json_field,
    ]
    if args.gatekeeper_channel_field:
        classify_fields.append(args.gatekeeper_channel_field)
    if args.gatekeeper_timestamp_field:
        classify_fields.append(args.gatekeeper_timestamp_field)

    eval_fields = [
        args.transcript_field,
        args.json_field,
        args.summary_field,
    ]
    if args.timestamp_field:
        eval_fields.append(args.timestamp_field)

    parse_fields = [
        args.eval_json_field,
        args.call_type_field,
        args.lead_quality_field,
        args.converted_field,
        args.primary_issue_field,
        args.overall_score_field,
        args.scheduling_efficiency_field,
        args.appointment_explicitly_offered_field,
        args.estimated_revenue_lost_range_field,
        args.estimated_revenue_lost_rationale_field,
        args.failure_mode_tags_field,
        args.strategic_quadrant_field,
        args.confidence_in_assessment_field,
    ]

    mode = args.mode.lower()
    if mode not in ("classify", "eval", "parse", "all"):
        raise ValueError("Mode must be one of: classify, eval, parse, all.")

    expected_fields = []
    if mode in ("classify", "all"):
        expected_fields.extend(classify_fields)
    if mode in ("eval", "all"):
        expected_fields.extend(eval_fields)
    if mode in ("parse", "all"):
        expected_fields.extend(parse_fields)

    expected_fields = [field for field in expected_fields if field]
    expected_unique = sorted({field.strip() for field in expected_fields if field.strip()})

    with requests.Session() as session:
        existing_fields = fetch_airtable_table_fields(
            session,
            airtable_api_key=airtable_api_key,
            base_id=airtable_base,
            table_name=airtable_table,
        )

    missing = [field for field in expected_unique if field not in existing_fields]

    print(f"Checked Airtable schema for table '{airtable_table}' ({mode}).")
    if missing:
        print("Missing fields:")
        for field in missing:
            print(f"- {field}")
    else:
        print("All expected fields are present.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CallRail recording downloader and transcriber.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-clients", help="List configured clients.")
    list_parser.add_argument("--config", default="clients.yaml", help="Path to clients config.")

    run_parser = subparsers.add_parser("run", help="Download, transcribe, and sync calls.")
    run_parser.add_argument("--client", required=True, help="Client name from config.")
    run_parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD).")
    run_parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD).")
    run_parser.add_argument("--config", default="clients.yaml", help="Path to clients config.")
    run_parser.add_argument("--airtable-base", help="Override Airtable base ID.")
    run_parser.add_argument("--airtable-table", help="Override Airtable table name.")
    run_parser.add_argument("--dry-run", action="store_true", help="Skip Airtable writes.")
    run_parser.add_argument("--test", action="store_true", help="Process only 5 calls (useful for validation).")

    run_csv_parser = subparsers.add_parser(
        "run-csv", help="Process only calls listed in a CSV (by recording URL)."
    )
    run_csv_parser.add_argument("--client", required=True, help="Client name from config.")
    run_csv_parser.add_argument("--csv", dest="csv_path", required=True, help="Path to CSV file.")
    run_csv_parser.add_argument(
        "--recording-url-column",
        default="Recording URL",
        help="Column name in the CSV containing the CallRail recording URL.",
    )
    run_csv_parser.add_argument(
        "--call-id-column",
        default=None,
        help="Optional column name containing CallRail call ID. If absent, the ID is parsed from the recording URL.",
    )
    run_csv_parser.add_argument("--config", default="clients.yaml", help="Path to clients config.")
    run_csv_parser.add_argument("--airtable-base", help="Override Airtable base ID.")
    run_csv_parser.add_argument("--airtable-table", help="Override Airtable table name.")
    run_csv_parser.add_argument("--dry-run", action="store_true", help="Skip Airtable writes.")
    run_csv_parser.add_argument("--test", action="store_true", help="Process only 5 rows (useful for validation).")

    classify_parser = subparsers.add_parser(
        "classify-airtable", help="Classify transcripts as live/voicemail using the gatekeeper prompt."
    )
    classify_parser.add_argument("--client", required=True, help="Client name from config.")
    classify_parser.add_argument("--config", default="clients.yaml", help="Path to clients config.")
    classify_parser.add_argument("--airtable-base", help="Override Airtable base ID.")
    classify_parser.add_argument("--airtable-table", help="Override Airtable table name.")
    classify_parser.add_argument(
        "--prompt-file",
        default=None,
        help="Override gatekeeper prompt from clients config by pointing to a prompt file.",
    )
    classify_parser.add_argument(
        "--transcript-field",
        default="Transcript",
        help="Airtable field containing the call transcript.",
    )
    classify_parser.add_argument(
        "--json-field",
        default="Gatekeeper JSON",
        help="Airtable field to store the gatekeeper JSON output.",
    )
    classify_parser.add_argument(
        "--channel-field",
        default="Gatekeeper Channel",
        help="Airtable field to store the gatekeeper channel (single select).",
    )
    classify_parser.add_argument(
        "--timestamp-field",
        default="Gatekeeper Timestamp",
        help="Airtable field to store the gatekeeper timestamp (set blank to skip).",
    )
    classify_parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for classification.",
    )
    classify_parser.add_argument(
        "--force",
        action="store_true",
        help="Process even if json-field already populated.",
    )
    classify_parser.add_argument("--dry-run", action="store_true", help="Skip Airtable writes.")
    classify_parser.add_argument("--test", action="store_true", help="Process only 5 records (useful for validation).")

    eval_parser = subparsers.add_parser(
        "eval-airtable",
        help="Evaluate transcripts already stored in Airtable using the client's system prompt (requires gatekeeper).",
    )
    eval_parser.add_argument("--client", required=True, help="Client name from config.")
    eval_parser.add_argument("--config", default="clients.yaml", help="Path to clients config.")
    eval_parser.add_argument("--airtable-base", help="Override Airtable base ID.")
    eval_parser.add_argument("--airtable-table", help="Override Airtable table name.")
    eval_parser.add_argument(
        "--prompt-file",
        default=None,
        help="Override system prompt from clients config by pointing to a prompt file.",
    )
    eval_parser.add_argument(
        "--transcript-field",
        default="Transcript",
        help="Airtable field containing the call transcript.",
    )
    eval_parser.add_argument(
        "--json-field",
        default="Evaluation JSON",
        help="Airtable field to store the JSON output from the agent.",
    )
    eval_parser.add_argument(
        "--summary-field",
        default="Evaluation Summary",
        help="Airtable field to store the executive summary markdown.",
    )
    eval_parser.add_argument(
        "--gatekeeper-json-field",
        default="Gatekeeper JSON",
        help="Airtable field containing gatekeeper JSON output.",
    )
    eval_parser.add_argument(
        "--gatekeeper-channel-field",
        default="Gatekeeper Channel",
        help="Airtable field containing gatekeeper channel.",
    )
    eval_parser.add_argument(
        "--timestamp-field",
        default="Evaluation Timestamp",
        help="Airtable field to store the evaluation timestamp (set blank to skip).",
    )
    eval_parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model for evaluation.",
    )
    eval_parser.add_argument(
        "--force",
        action="store_true",
        help="Process even if json-field already populated.",
    )
    eval_parser.add_argument("--dry-run", action="store_true", help="Skip Airtable writes.")
    eval_parser.add_argument("--test", action="store_true", help="Process only 5 records (useful for validation).")

    parse_parser = subparsers.add_parser(
        "parse-eval-json",
        help=(
            "Parse Evaluation JSON field into typed Airtable columns (call type, lead quality, converted, "
            "primary issue, overall score, and v2 evaluation fields)."
        ),
    )
    parse_parser.add_argument("--client", required=True, help="Client name from config.")
    parse_parser.add_argument("--config", default="clients.yaml", help="Path to clients config.")
    parse_parser.add_argument("--airtable-base", help="Override Airtable base ID.")
    parse_parser.add_argument("--airtable-table", help="Override Airtable table name.")
    parse_parser.add_argument(
        "--eval-json-field",
        default="Evaluation JSON",
        help="Airtable field containing evaluation JSON text.",
    )
    parse_parser.add_argument(
        "--output-csv",
        default=None,
        help="Write flattened evaluation JSON to a CSV file instead of updating Airtable.",
    )
    parse_parser.add_argument(
        "--call-type-field",
        default="Call Classification",
        help="Airtable field to store call classification (single select).",
    )
    parse_parser.add_argument(
        "--lead-quality-field",
        default="Lead Quality Rating",
        help="Airtable field to store lead quality rating (single select).",
    )
    parse_parser.add_argument(
        "--converted-field",
        default="Converted",
        help="Airtable field to store converted status (single select).",
    )
    parse_parser.add_argument(
        "--primary-issue-field",
        default="Primary Issue",
        help="Airtable field to store primary issue (single select).",
    )
    parse_parser.add_argument(
        "--overall-score-field",
        default="Overall Score",
        help="Airtable field to store overall score (number).",
    )
    parse_parser.add_argument(
        "--scheduling-efficiency-field",
        default="Scheduling Efficiency",
        help="Airtable field to store scheduling efficiency score (number).",
    )
    parse_parser.add_argument(
        "--appointment-explicitly-offered-field",
        default="Appointment Explicitly Offered",
        help="Airtable field to store appointment explicitly offered (checkbox).",
    )
    parse_parser.add_argument(
        "--estimated-revenue-lost-range-field",
        default="Estimated Revenue Lost Range",
        help="Airtable field to store estimated revenue lost range (single select).",
    )
    parse_parser.add_argument(
        "--estimated-revenue-lost-rationale-field",
        default="Estimated Revenue Lost Rationale",
        help="Airtable field to store estimated revenue lost rationale (long text).",
    )
    parse_parser.add_argument(
        "--failure-mode-tags-field",
        default="Failure Mode Tags",
        help="Airtable field to store failure mode tags (multi-select).",
    )
    parse_parser.add_argument(
        "--strategic-quadrant-field",
        default="Strategic Quadrant",
        help="Airtable field to store strategic quadrant classification (single select).",
    )
    parse_parser.add_argument(
        "--confidence-in-assessment-field",
        default="Confidence In Assessment",
        help="Airtable field to store confidence indicator (single select).",
    )
    parse_parser.add_argument(
        "--force",
        action="store_true",
        help="Process even if destination fields are already populated.",
    )
    parse_parser.add_argument("--dry-run", action="store_true", help="Skip Airtable writes.")
    parse_parser.add_argument("--test", action="store_true", help="Process only 5 records (useful for validation).")

    schema_parser = subparsers.add_parser(
        "check-airtable-schema",
        help="Check Airtable table for required classify/eval/parse fields.",
    )
    schema_parser.add_argument("--client", required=True, help="Client name from config.")
    schema_parser.add_argument("--config", default="clients.yaml", help="Path to clients config.")
    schema_parser.add_argument("--airtable-base", help="Override Airtable base ID.")
    schema_parser.add_argument("--airtable-table", help="Override Airtable table name.")
    schema_parser.add_argument(
        "--mode",
        default="all",
        help="Fields to validate: classify, eval, parse, or all (default).",
    )
    schema_parser.add_argument(
        "--transcript-field",
        default="Transcript",
        help="Airtable field containing the call transcript.",
    )
    schema_parser.add_argument(
        "--gatekeeper-json-field",
        default="Gatekeeper JSON",
        help="Airtable field containing gatekeeper JSON output.",
    )
    schema_parser.add_argument(
        "--gatekeeper-channel-field",
        default="Gatekeeper Channel",
        help="Airtable field to store gatekeeper channel (single select).",
    )
    schema_parser.add_argument(
        "--gatekeeper-timestamp-field",
        default="Gatekeeper Timestamp",
        help="Airtable field to store the gatekeeper timestamp (set blank to skip).",
    )
    schema_parser.add_argument(
        "--json-field",
        default="Evaluation JSON",
        help="Airtable field to store the JSON output from the agent.",
    )
    schema_parser.add_argument(
        "--summary-field",
        default="Evaluation Summary",
        help="Airtable field to store the executive summary markdown.",
    )
    schema_parser.add_argument(
        "--timestamp-field",
        default="Evaluation Timestamp",
        help="Airtable field to store the evaluation timestamp (set blank to skip).",
    )
    schema_parser.add_argument(
        "--eval-json-field",
        default="Evaluation JSON",
        help="Airtable field containing evaluation JSON text.",
    )
    schema_parser.add_argument(
        "--call-type-field",
        default="Call Classification",
        help="Airtable field to store call classification (single select).",
    )
    schema_parser.add_argument(
        "--lead-quality-field",
        default="Lead Quality Rating",
        help="Airtable field to store lead quality rating (single select).",
    )
    schema_parser.add_argument(
        "--converted-field",
        default="Converted",
        help="Airtable field to store converted status (single select).",
    )
    schema_parser.add_argument(
        "--primary-issue-field",
        default="Primary Issue",
        help="Airtable field to store primary issue (single select).",
    )
    schema_parser.add_argument(
        "--overall-score-field",
        default="Overall Score",
        help="Airtable field to store overall score (number).",
    )
    schema_parser.add_argument(
        "--scheduling-efficiency-field",
        default="Scheduling Efficiency",
        help="Airtable field to store scheduling efficiency score (number).",
    )
    schema_parser.add_argument(
        "--appointment-explicitly-offered-field",
        default="Appointment Explicitly Offered",
        help="Airtable field to store appointment explicitly offered (checkbox).",
    )
    schema_parser.add_argument(
        "--estimated-revenue-lost-range-field",
        default="Estimated Revenue Lost Range",
        help="Airtable field to store estimated revenue lost range (single select).",
    )
    schema_parser.add_argument(
        "--estimated-revenue-lost-rationale-field",
        default="Estimated Revenue Lost Rationale",
        help="Airtable field to store estimated revenue lost rationale (long text).",
    )
    schema_parser.add_argument(
        "--failure-mode-tags-field",
        default="Failure Mode Tags",
        help="Airtable field to store failure mode tags (multi-select).",
    )
    schema_parser.add_argument(
        "--strategic-quadrant-field",
        default="Strategic Quadrant",
        help="Airtable field to store strategic quadrant classification (single select).",
    )
    schema_parser.add_argument(
        "--confidence-in-assessment-field",
        default="Confidence In Assessment",
        help="Airtable field to store confidence indicator (single select).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list-clients":
        clients = load_clients(args.config)
        for client in clients:
            print(f"- {client.name} (account_id={client.account_id}, company_id={client.company_id})")
        return
    if args.command == "run":
        run_pipeline(args)
        return
    if args.command == "run-csv":
        run_csv_pipeline(args)
        return
    if args.command == "classify-airtable":
        run_classify_airtable(args)
        return
    if args.command == "eval-airtable":
        run_eval_airtable(args)
        return
    if args.command == "parse-eval-json":
        run_parse_eval_json(args)
        return
    if args.command == "check-airtable-schema":
        run_check_airtable_schema(args)
        return
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
