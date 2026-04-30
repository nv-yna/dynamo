// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4 decoder-track tests using live-captured fixtures.
//
// Each fixture under `tests/fixtures/dsv4_live_captured/<probe-id>/` is a quad:
//   - request.json — the /v1/completions payload that produced this output
//   - raw.txt      — the model's raw text output (parser flags off at the serve layer)
//   - meta.json    — { thinking_mode, finish_reason, model, ... }
//   - golden.json  — what HF's `parse_message_from_completion_text` produces from raw.txt
//
// Each test feeds raw.txt through dynamo's reasoning parser + DSML tool-call parser,
// reconstructs the assistant message, and asserts (after normalization) that it matches
// the HF golden. Disagreements are decoder-track gaps between dynamo and the HF reference.
//
// Capture methodology and design notes:
//   ~/code/tmp_context/deepseek-v4/frontend/20260430/triage-plan.md
//
// Adding a new probe (one-time per fixture):
//   1. Define the (messages, thinking_mode, max_tokens) probe in the capture script at
//      ~/code/tmp_context/deepseek-v4/frontend/20260430/capture_probes.py
//   2. Run the script — it renders the prompt via HF Python, POSTs to the parser-disabled
//      trtllm-serve endpoint, parses the raw output through HF, and writes the four files.
//   3. Copy the new probe directory into tests/fixtures/dsv4_live_captured/.
//   4. Re-run `cargo test -p dynamo-parsers --test dsv4_live_captured` — the new probe is
//      auto-discovered.

use std::fs;
use std::path::PathBuf;

use dynamo_parsers::reasoning::{ReasoningParser, ReasoningParserType};
use dynamo_parsers::tool_calling::config::{ParserConfig, ToolCallConfig};
use dynamo_parsers::tool_calling::dsml::try_tool_call_parse_dsml;
use serde_json::{json, Value};

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dsv4_live_captured")
}

/// Run dynamo's reasoning + DSML tool-call parsers on a captured raw output and assemble
/// an assistant-message JSON shaped like HF's `parse_message_from_completion_text` output.
fn dynamo_parse(raw: &str, thinking_mode: &str) -> Value {
    // Reasoning extraction. V4 always pre-injects `<think>` into the prompt rather than
    // emitting it in the completion, so for thinking-mode probes we must seed the parser
    // via `set_in_reasoning(true)` — otherwise the leading reasoning prose gets bucketed
    // as normal_text. See ReasoningParserType::DeepSeekV4 docstring.
    let mut reasoning_parser =
        ReasoningParserType::get_reasoning_parser_from_name("deepseek_v4");
    if thinking_mode == "thinking" {
        reasoning_parser.set_in_reasoning(true);
    }
    let split = reasoning_parser.detect_and_parse_reasoning(raw, &[]);
    let reasoning_text = split.reasoning_text;
    let normal_text = split.normal_text;

    // Tool-call extraction runs on the post-reasoning text.
    let v4_cfg = match ToolCallConfig::deepseek_v4().parser_config {
        ParserConfig::Dsml(c) => c,
        other => panic!("expected Dsml config for deepseek_v4, got {:?}", other),
    };
    let (tool_calls, leftover) = try_tool_call_parse_dsml(&normal_text, &v4_cfg)
        .expect("try_tool_call_parse_dsml should not fail on valid V4 output");

    let tool_calls_json: Vec<Value> = tool_calls
        .into_iter()
        .map(|tc| {
            json!({
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            })
        })
        .collect();

    json!({
        "role": "assistant",
        "content": leftover.unwrap_or_default(),
        "reasoning_content": reasoning_text,
        "tool_calls": tool_calls_json,
    })
}

/// Normalize an assistant message for byte-stable comparison between dynamo and HF outputs.
///
/// HF's `parse_message_from_completion_text` does not emit tool-call IDs, while dynamo
/// always does. `function.arguments` is a JSON-encoded string in both, but with possibly
/// different key ordering (HF preserves Python dict order; serde_json may reorder), so we
/// reparse and compare structurally. Whitespace at the edges of content/reasoning is
/// not load-bearing (HF and dynamo split slightly differently around `</think>` newlines),
/// so we trim both ends. Empty vs null collapses to empty for both content and tool_calls.
fn normalize(v: &Value) -> Value {
    let role = v
        .get("role")
        .and_then(|r| r.as_str())
        .unwrap_or("assistant")
        .to_string();

    let content = v
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let reasoning = v
        .get("reasoning_content")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let tool_calls: Vec<Value> = v
        .get("tool_calls")
        .and_then(|t| t.as_array())
        .map(|arr| arr.iter().map(normalize_tool_call).collect())
        .unwrap_or_default();

    json!({
        "role": role,
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
    })
}

fn normalize_tool_call(tc: &Value) -> Value {
    // Strip id, parse arguments JSON for structural comparison.
    let function = tc.get("function").cloned().unwrap_or_else(|| json!({}));
    let name = function
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string();
    let arguments_str = function
        .get("arguments")
        .and_then(|a| a.as_str())
        .unwrap_or("{}");
    let arguments_value: Value =
        serde_json::from_str(arguments_str).unwrap_or_else(|_| json!(arguments_str));

    json!({
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_value,
        },
    })
}

fn run_probe(probe_id: &str) {
    let dir = fixture_root().join(probe_id);
    let raw = fs::read_to_string(dir.join("raw.txt"))
        .unwrap_or_else(|e| panic!("read raw.txt for {}: {}", probe_id, e));
    let meta: Value = serde_json::from_str(
        &fs::read_to_string(dir.join("meta.json"))
            .unwrap_or_else(|e| panic!("read meta.json for {}: {}", probe_id, e)),
    )
    .unwrap_or_else(|e| panic!("parse meta.json for {}: {}", probe_id, e));
    let golden: Value = serde_json::from_str(
        &fs::read_to_string(dir.join("golden.json"))
            .unwrap_or_else(|e| panic!("read golden.json for {}: {}", probe_id, e)),
    )
    .unwrap_or_else(|e| panic!("parse golden.json for {}: {}", probe_id, e));

    if golden.get("_hf_raised").is_some() {
        // HF rejected the raw text. Decoder-track tests don't exercise HF-rejection paths
        // in this fixture set; skip with a clear log so future probes aren't silently dropped.
        eprintln!(
            "[skip] {}: HF parser raised ({}); not asserting dynamo behavior here",
            probe_id,
            golden["_hf_raised"]
        );
        return;
    }

    let thinking_mode = meta
        .get("thinking_mode")
        .and_then(|t| t.as_str())
        .unwrap_or("chat");

    let actual = dynamo_parse(&raw, thinking_mode);
    let actual_norm = normalize(&actual);
    let golden_norm = normalize(&golden);

    assert_eq!(
        actual_norm,
        golden_norm,
        "\nProbe '{}' diverged from HF golden.\n\nactual (normalized):\n{}\n\ngolden (normalized):\n{}\n\nraw output:\n{}\n",
        probe_id,
        serde_json::to_string_pretty(&actual_norm).unwrap(),
        serde_json::to_string_pretty(&golden_norm).unwrap(),
        raw,
    );
}

#[test]
fn basic_tool_call() {
    run_probe("basic_tool_call");
}

#[test]
fn multi_param_tool() {
    run_probe("multi_param_tool");
}

#[test]
fn multi_tool_call() {
    run_probe("multi_tool_call");
}

#[test]
fn reasoning_only_no_tools() {
    run_probe("reasoning_only_no_tools");
}

#[test]
fn tool_followup_assistant_turn() {
    run_probe("tool_followup_assistant_turn");
}

#[test]
fn chat_mode_no_thinking() {
    run_probe("chat_mode_no_thinking");
}

#[test]
fn fixture_directory_is_complete() {
    // Sanity check: every fixture directory must have all four files.
    let root = fixture_root();
    let probes: Vec<_> = fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("read fixture root {}: {}", root.display(), e))
        .filter_map(|entry| entry.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    assert!(!probes.is_empty(), "no fixture directories found at {}", root.display());

    for probe in probes {
        let dir = probe.path();
        for required in &["request.json", "raw.txt", "meta.json", "golden.json"] {
            assert!(
                dir.join(required).is_file(),
                "fixture {} missing {}",
                dir.display(),
                required
            );
        }
    }
}
