# Changelog

All notable changes to the Repo-Aware AI VS Code extension are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- API keys are now stored **only** in VS Code SecretStorage and are no longer mirrored
  into `settings.json`, so they can never leak via settings exports or Settings Sync.

### Added
- Extension `README.md`, `LICENSE`, and `CHANGELOG.md` for Marketplace publishing.

## [0.1.0] — 2026-05-09

First public preview.

### Added
- End-to-end activation: bootstrap a private Python venv, install the backend,
  spawn the uvicorn sidecar, and auto-index the workspace.
- Commands: Ask Codebase, Explain Selection, Find Related Code, Explain This
  Repository, Rebuild Index, Open Chat Panel, Open Repo Map, Set API Key,
  Show Backend Logs.
- Chat sidebar with live token streaming (blinking cursor while generating).
- Repo Map panel (dependency / community graph).
- CodeLens and hover providers for inline action hints.
- Status-bar states: down / indexing / ready / error.
- "Show Backend Logs" and "Set API Key" always registered, even if activation fails.
- Citations open the referenced file at the chunk's start character.
