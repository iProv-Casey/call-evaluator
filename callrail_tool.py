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

    def try_download(url: str, headers: Optional[Dict[str, str]]) -> Optional[bytes]:
        try:
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
    resp.raise_for_status()


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
) -> Dict[str, object]:
    fields: Dict[str, object] = {}
    call_type = eval_json.get("call_type")
    if call_type:
        fields[call_type_field] = call_type

    lead_quality = (eval_json.get("lead_quality") or {}).get("rating")
    if lead_quality:
        fields[lead_quality_field] = lead_quality

    outcome = (eval_json.get("outcome") or {})
    converted_raw = outcome.get("converted")
    if converted_raw is None or converted_raw == "":
        converted_value = "not_applicable"
    else:
        converted_value = str(converted_raw)
    fields[converted_field] = converted_value

    primary_issue = (eval_json.get("root_cause") or {}).get("primary_issue")
    if primary_issue:
        fields[primary_issue_field] = primary_issue

    scores = (eval_json.get("scores") or {})
    overall = scores.get("overall_score")
    if overall not in (None, ""):
        try:
            fields[overall_score_field] = float(overall)
        except Exception:
            pass

    return fields


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

    # Resolve the system prompt from client config or a prompt file.
    prompt = selected_client.system_prompt
    if not prompt and args.prompt_file:
        prompt = load_prompt(args.prompt_file)
    if not prompt:
        raise ValueError(
            "System prompt is required. Add system_prompt to the client config or provide --prompt-file."
        )
    # Build an Airtable filter formula that skips already-evaluated rows unless forced.
    filter_formula = None
    if not args.force:
        filter_formula = f"AND({{{args.transcript_field}}}, LEN({{{args.transcript_field}}})>0, NOT({{{args.json_field}}}))"
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

            # Send the transcript to OpenAI for evaluation.
            print(f"[{idx}/{total}] Evaluating record {record.get('id')}...")
            try:
                json_text, summary_text = evaluate_transcript(
                    openai_client,
                    model=args.model,
                    prompt=prompt,
                    transcript=transcript,
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
    clients = load_clients(args.config)
    selected_client = select_client(clients, args.client)
    max_records = 5 if args.test else None

    airtable_api_key = os.getenv("AIRTABLE_API_KEY")
    if not airtable_api_key:
        raise EnvironmentError("AIRTABLE_API_KEY is required.")
    airtable_base = args.airtable_base or selected_client.airtable_base_id
    airtable_table = args.airtable_table or selected_client.airtable_table_name
    if not airtable_base or not airtable_table:
        raise ValueError("Airtable base and table are required (set via flags or client config).")

    if args.force:
        filter_formula = f"AND({{{args.eval_json_field}}}, LEN({{{args.eval_json_field}}})>0)"
    else:
        filter_formula = (
            f"AND({{{args.eval_json_field}}}, LEN({{{args.eval_json_field}}})>0, NOT({{{args.call_type_field}}}))"
        )

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
            eval_json_str = fields.get(args.eval_json_field)
            if not eval_json_str:
                print(f"[{idx}/{total}] Skipping record {record.get('id')}: missing eval JSON.")
                continue

            print(f"[{idx}/{total}] Parsing evaluation for record {record.get('id')}...")
            try:
                eval_json = json.loads(eval_json_str)
            except Exception as exc:
                print(f"Failed to parse JSON for record {record.get('id')}: {exc}")
                continue

            update_fields = parse_eval_json_fields(
                eval_json,
                call_type_field=args.call_type_field,
                lead_quality_field=args.lead_quality_field,
                converted_field=args.converted_field,
                primary_issue_field=args.primary_issue_field,
                overall_score_field=args.overall_score_field,
            )

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
                print(f"Updated record {record.get('id')} with parsed evaluation fields.")
            except Exception as exc:
                print(f"Failed to update Airtable for record {record.get('id')}: {exc}")


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

    eval_parser = subparsers.add_parser(
        "eval-airtable", help="Evaluate transcripts already stored in Airtable using the client's system prompt."
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
        help="Parse Evaluation JSON field into typed Airtable columns (call type, lead quality, converted, primary issue, overall score).",
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
        "--call-type-field",
        default="Call Type",
        help="Airtable field to store call type (single select).",
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
        "--force",
        action="store_true",
        help="Process even if destination fields are already populated.",
    )
    parse_parser.add_argument("--dry-run", action="store_true", help="Skip Airtable writes.")
    parse_parser.add_argument("--test", action="store_true", help="Process only 5 records (useful for validation).")

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
    if args.command == "eval-airtable":
        run_eval_airtable(args)
        return
    if args.command == "parse-eval-json":
        run_parse_eval_json(args)
        return
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
