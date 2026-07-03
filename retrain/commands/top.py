"""Top-level CLI help."""

from __future__ import annotations


def print_help(cli_name: str) -> None:
    """Print concise top-level help with strong manual discoverability."""
    print(f"{cli_name} — TOML-first RLVR trainer")
    print()
    print("Usage:")
    print(f"  {cli_name} [config.toml] [--flag value ...]")
    print(f"  {cli_name} backends [--json]")
    print(f"  {cli_name} doctor")
    print(
        f"  {cli_name} migrate-config <config.toml> "
        "[--check|--write|--output PATH] [--backup] [--stdin|--stdout] [--json]"
    )
    print(f"  {cli_name} init [--template NAME] [--list] [--interactive]")
    print(
        f"  {cli_name} init-plugin --kind KIND --name NAME "
        "[--output-dir DIR] [--with-test]"
    )
    print(f"  {cli_name} plugins [--json] [config.toml]")
    print(f"  {cli_name} status [logdir] [--json] [--all] [--watch]")
    print(f"  {cli_name} top [logdir]")
    print(f"  {cli_name} explain [config.toml] [--json]")
    print(f"  {cli_name} diff <run_a> <run_b> [--json]")
    print(
        f"  {cli_name} benchmark <config.toml|run_dir> "
        "[--repeat N] [--output-dir DIR] [--json]"
    )
    print(f"  {cli_name} trace [config.toml] [--json]")
    print(
        f"  {cli_name} tree [tree.toml] [--json]"
        "  (view | next | show | run | note | eval | reset)"
    )
    print(f"  {cli_name} man")
    print()
    print("Manual:")
    print(f"  {cli_name} man")
    print(f"  {cli_name} man --topic quickstart")
    print(f"  {cli_name} man --path")
    print(f"  {cli_name} man --sync")
    print(f"  {cli_name} man --check")
    print()
    print("Tip: read docs/configuration.md for full TOML reference.")
