#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync a client system_prompt in clients.yaml with a prompt file."
    )
    parser.add_argument("--client", required=True, help="Client name to update.")
    parser.add_argument("--prompt-file", required=True, help="Path to the prompt file.")
    parser.add_argument("--config", default="clients.yaml", help="Path to clients config.")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return value.strip().strip('"').strip("'").lower()


def extract_client_name(line: str) -> Optional[str]:
    stripped = line.lstrip()
    if not stripped.startswith("- name:"):
        return None
    _, raw = stripped.split(":", 1)
    return raw.strip()


def leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def sync_prompt(lines: List[str], client_name: str, prompt: str) -> Tuple[List[str], bool]:
    output: List[str] = []
    current_client: Optional[str] = None
    target = normalize_name(client_name)
    prompt_lines = prompt.splitlines()
    updated = False
    i = 0
    while i < len(lines):
        line = lines[i]
        name = extract_client_name(line)
        if name is not None:
            current_client = normalize_name(name)
            output.append(line)
            i += 1
            continue

        if current_client == target and line.lstrip().startswith("system_prompt: |"):
            indent = leading_spaces(line)
            output.append(line)
            i += 1
            # Skip existing block lines (more indented than the system_prompt line).
            while i < len(lines) and leading_spaces(lines[i]) > indent:
                i += 1
            block_indent = " " * (indent + 2)
            for pline in prompt_lines:
                if pline:
                    output.append(f"{block_indent}{pline}\n")
                else:
                    output.append(f"{block_indent.rstrip()}\n")
            updated = True
            continue

        output.append(line)
        i += 1

    return output, updated


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    prompt_path = Path(args.prompt_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    prompt = prompt_path.read_text(encoding="utf-8").rstrip()
    updated_lines, did_update = sync_prompt(lines, args.client, prompt)
    if not did_update:
        raise ValueError(f"system_prompt block not found for client '{args.client}'.")
    config_path.write_text("".join(updated_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
