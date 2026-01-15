#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update a client's prompt_dir in clients.yaml."
    )
    parser.add_argument("--client", required=True, help="Client name to update.")
    parser.add_argument("--prompt-dir", required=True, help="Prompt directory to store in config.")
    parser.add_argument("--config", default="clients.yaml", help="Path to clients config.")
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return value.strip().strip('"').strip("'").lower()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    prompt_dir = Path(args.prompt_dir)
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    clients = config.get("clients", [])
    target = normalize_name(args.client)
    updated = False

    for client in clients:
        if normalize_name(client.get("name", "")) == target:
            client["prompt_dir"] = str(prompt_dir)
            client.pop("system_prompt", None)
            updated = True
            break

    if not updated:
        raise ValueError(f"Client '{args.client}' not found in {config_path}.")

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
