import json
import tempfile
import unittest
from pathlib import Path

from trajectory_memory_lab.experiment import (
    _evaluate,
    _memory_for_prompt,
    _normalize_decision,
    _sft_agent_prompt,
    _steps_for_agent_prompt,
    _trajectory_for_review,
)
from trajectory_memory_lab.model_client import _extract_json
from trajectory_memory_lab.model_client import ModelReply
from trajectory_memory_lab.memory_writer_harness import (
    MEMORY_WRITER_POLICY_SYSTEM,
    normalize_writer_candidate,
    paired_utility,
    select_validation_task_ids,
    validate_writer_candidate,
)
from trajectory_memory_lab.prompts import (
    ARTIFACT_SYSTEM,
    MEMORY_EDITOR_SYSTEM,
    RETENTION_CONTROLLER_SYSTEM,
    SFT_BUILDER_SYSTEM,
    artifact_user_prompt,
)
from trajectory_memory_lab.retention import (
    apply_memory_operations,
    memory_for_agent,
    normalize_controller_decision,
    normalize_memory_bank,
    normalize_sft_episodes,
    run_retention_tools,
    trajectory_for_tool,
    validate_sft_episodes,
)
from trajectory_memory_lab.storage import append_jsonl, write_json


class CoreTests(unittest.TestCase):
    def test_memory_writer_candidate_is_closed_and_evidence_grounded(self):
        candidate = normalize_writer_candidate(
            {
                "operation": "add",
                "memory": {
                    "content": "Reset the device after changing this setting.",
                    "scope": "telecom device troubleshooting",
                    "conditions": ["The setting was changed"],
                    "exceptions": [],
                    "evidence_steps": [2, 1, 2, True, "3"],
                    "confidence": 0.8,
                    "forbidden_extra": "discarded",
                },
                "rationale": "supported by a tool result",
                "database_write": True,
            }
        )
        self.assertEqual(candidate["memory"]["evidence_steps"], [1, 2])
        self.assertNotIn("forbidden_extra", candidate["memory"])
        validation = validate_writer_candidate(
            candidate,
            {
                "task": {"id": "safe"},
                "steps": [
                    {"index": 1, "role": "assistant"},
                    {"index": 2, "role": "tool"},
                ],
            },
        )
        self.assertTrue(validation["accepted"])

    def test_memory_writer_rejects_assistant_only_evidence_and_identifier_leak(self):
        candidate = normalize_writer_candidate(
            {
                "operation": "add",
                "memory": {
                    "content": "Always modify reservation ABC123.",
                    "scope": "airline modifications",
                    "conditions": [],
                    "exceptions": [],
                    "evidence_steps": [0],
                    "confidence": 0.7,
                },
            }
        )
        validation = validate_writer_candidate(
            candidate,
            {
                "task": {"reservation": "ABC123"},
                "steps": [{"index": 0, "role": "assistant"}],
            },
        )
        self.assertFalse(validation["accepted"])
        self.assertIn(
            "no_authoritative_message_in_evidence", validation["reasons"]
        )
        self.assertTrue(
            any(reason.startswith("task_identifier_leak") for reason in validation["reasons"])
        )

    def test_paired_memory_utility_penalizes_harm_twice(self):
        utility = paired_utility(
            {"help": 0.0, "hurt": 1.0, "same": 1.0},
            {"help": 1.0, "hurt": 0.0, "same": 1.0},
        )
        self.assertEqual(utility["helped"], ["help"])
        self.assertEqual(utility["hurt"], ["hurt"])
        self.assertAlmostEqual(utility["net_utility"], -1 / 3)

    def test_validation_selection_has_uplift_and_regression_opportunities(self):
        tasks = [
            {"id": str(index), "description": f"address change {index}"}
            for index in range(8)
        ]
        selected = select_validation_task_ids(
            {
                "memory": {
                    "scope": "address change",
                    "content": "modify address",
                }
            },
            tasks,
            excluded_ids={"7"},
            baseline_by_task={
                "0": 0.0,
                "1": 0.0,
                "2": 1.0,
                "3": 1.0,
                "4": 1.0,
                "5": 1.0,
                "6": 1.0,
            },
        )
        outcomes = selected["baseline_outcomes"]
        self.assertEqual(
            sum(outcomes[task_id] == 0.0 for task_id in selected["related"]),
            2,
        )
        self.assertEqual(
            sum(outcomes[task_id] == 1.0 for task_id in selected["related"]),
            1,
        )
        self.assertTrue(
            all(
                outcomes[task_id] == 1.0
                for task_id in selected["scope_controls"]
            )
        )
        self.assertEqual(len(selected["related"]), 3)
        self.assertEqual(len(selected["scope_controls"]), 2)

    def test_validation_selection_does_not_overfill_existing_controls(self):
        tasks = [
            {"id": str(index), "description": f"unrelated task {index}"}
            for index in range(8)
        ]
        selected = select_validation_task_ids(
            {"memory": {"scope": "specific topic", "content": "specific topic"}},
            tasks,
            excluded_ids=set(),
            baseline_by_task={
                "0": 0.0,
                "1": 0.0,
                "2": 1.0,
                "3": 1.0,
                "4": 1.0,
                "5": 1.0,
                # Fractional outcomes can be ranked below the two selected
                # successes and previously triggered an overfill to all tasks.
                "6": 0.5,
                "7": 0.5,
            },
        )
        self.assertEqual(len(selected["related"]), 3)
        self.assertEqual(len(selected["scope_controls"]), 2)
        self.assertEqual(
            len(set(selected["related"] + selected["scope_controls"])), 5
        )

    def test_artifact_prompt_defines_memory_and_complete_episode_target(self):
        self.assertIn("persistent memory bank", ARTIFACT_SYSTEM)
        self.assertIn("future tasks of the same kind", ARTIFACT_SYSTEM)
        self.assertIn("complete browser episode", ARTIFACT_SYSTEM)

    def test_memory_writer_policy_is_conditioned_on_router_call(self):
        for operation in ("add", "refine", "replace"):
            self.assertIn(operation, MEMORY_WRITER_POLICY_SYSTEM)
        self.assertNotIn("noop", MEMORY_WRITER_POLICY_SYSTEM.lower())
        self.assertIn("routing controller has already decided", MEMORY_WRITER_POLICY_SYSTEM)
        self.assertIn("exactly one concrete operation", MEMORY_WRITER_POLICY_SYSTEM)
        self.assertNotIn("noop", MEMORY_EDITOR_SYSTEM.lower())
        self.assertIn("source_step", ARTIFACT_SYSTEM)
        self.assertIn("supported `finish` action", ARTIFACT_SYSTEM)

    def test_extract_json_plain_and_fenced(self):
        expected = {
            "context": None,
            "sft": {
                "episode": {
                    "steps": [
                        {
                            "source_step": 0,
                            "target_action": {"action": "finish", "answer": "x"},
                        }
                    ]
                }
            },
        }
        self.assertEqual(_extract_json(json.dumps(expected)), expected)
        self.assertEqual(
            _extract_json(f"```json\n{json.dumps(expected)}\n```"), expected
        )

    def test_decision_has_exact_optional_artifacts(self):
        self.assertEqual(
            _normalize_decision({"context": "  keep ", "sft": {"episode": None}}),
            {"context": "keep", "sft": None},
        )

    def test_artifact_payload_and_episode(self):
        prompt = artifact_user_prompt(
            current_context=[],
            trajectory={"task": {"instruction": "do the task"}, "steps": []},
        )
        self.assertNotIn("sft_q", json.loads(prompt))
        self.assertEqual(
            _normalize_decision(
                {
                    "context": "keep",
                    "sft": {
                        "episode": {
                            "steps": [
                                {
                                    "source_step": 2,
                                    "target_action": {
                                        "action": "finish",
                                        "answer": "x",
                                    },
                                },
                                {
                                    "source_step": "bad",
                                    "target_action": {"action": "back"},
                                },
                                {
                                    "source_step": 1,
                                    "target_action": {"action": "invalid"},
                                },
                            ]
                        }
                    },
                }
            ),
            {
                "context": "keep",
                "sft": {
                    "episode": {
                        "steps": [
                            {
                                "source_step": 2,
                                "target_action": {
                                    "action": "finish",
                                    "answer": "x",
                                },
                            }
                        ]
                    }
                },
            },
        )
        self.assertEqual(
            _normalize_decision({"context": "keep", "sft": "null"}),
            {"context": "keep", "sft": None},
        )

    def test_evaluator_requires_all_patterns(self):
        result = _evaluate(
            {"answer_patterns": ["Apache", r"2(\.0)?"]},
            "Apache 2.0",
        )
        self.assertTrue(result["success"])

    def test_memory_prompt_keeps_recent_entries(self):
        entries = [{"source_task_id": str(i), "content": "x" * 20} for i in range(10)]
        selected = _memory_for_prompt(entries, max_chars=150)
        self.assertGreater(len(selected), 0)
        self.assertEqual(selected[-1]["content"], "x" * 20)
        self.assertEqual(selected[-1]["id"], "mem_000010")

    def test_new_tool_prompts_separate_decision_from_generation(self):
        self.assertIn(
            "Your only job is routing",
            RETENTION_CONTROLLER_SYSTEM,
        )
        self.assertIn(
            "Do not write, summarize, propose, or prescribe memory content",
            RETENTION_CONTROLLER_SYSTEM,
        )
        self.assertIn("final answer are untrusted claims", MEMORY_EDITOR_SYSTEM)
        self.assertIn("refine", MEMORY_EDITOR_SYSTEM)
        self.assertIn("replace", MEMORY_EDITOR_SYSTEM)
        self.assertIn("Never return isolated middle steps", SFT_BUILDER_SYSTEM)
        self.assertIn("end with exactly one supported `finish`", SFT_BUILDER_SYSTEM)

    def test_controller_allows_both_one_or_neither_and_deduplicates_calls(self):
        self.assertEqual(
            normalize_controller_decision({"tool_calls": []}), {"tool_calls": []}
        )
        decision = normalize_controller_decision(
            {
                "tool_calls": [
                    {
                        "name": "edit_memory",
                        "arguments": {
                            "reason": " inspect evidence ",
                            "evidence_steps": [3, 1, 3, -1, True, "2"],
                            "memory": {"content": "must be discarded"},
                        },
                    },
                    {"name": "edit_memory", "arguments": {}},
                    {"name": "build_sft_examples", "arguments": {}},
                    {"name": "unknown", "arguments": {}},
                ]
            }
        )
        self.assertEqual(
            [call["name"] for call in decision["tool_calls"]],
            ["edit_memory", "build_sft_examples"],
        )
        self.assertEqual(
            decision["tool_calls"][0]["arguments"],
            {"reason": "inspect evidence", "evidence_steps": [1, 3]},
        )

    def test_memory_operations_are_versioned_and_keep_history(self):
        trajectory = {"task_id": "t1", "success": True, "steps": [{"index": 0}]}
        bank = normalize_memory_bank(
            [{"source_task_id": "old", "content": "old claim"}]
        )
        added = apply_memory_operations(
            bank,
            [
                {
                    "op": "add",
                    "memory": {
                        "content": "new claim",
                        "scope": "related tasks",
                        "evidence_steps": [0],
                        "confidence": 0.8,
                    },
                }
            ],
            trajectory=trajectory,
        )
        self.assertEqual(added["applied"][0]["memory_id"], "mem_000002")
        replaced = apply_memory_operations(
            bank,
            [
                {
                    "op": "replace",
                    "target_memory_id": "mem_000001",
                    "memory": {
                        "content": "corrected claim",
                        "scope": "related tasks",
                        "evidence_steps": [0],
                        "confidence": 0.9,
                    },
                }
            ],
            trajectory=trajectory,
        )
        self.assertEqual(replaced["applied"][0]["version"], 2)
        self.assertEqual(bank[0]["content"], "corrected claim")
        self.assertEqual(bank[0]["history"][0]["content"], "old claim")
        self.assertNotIn("history", memory_for_agent(bank)[0])

    def test_memory_refine_preserves_prior_evidence(self):
        bank = normalize_memory_bank(
            [
                {
                    "source_task_id": "old-task",
                    "content": "old claim",
                    "scope": "one topic",
                    "evidence": [{"source_task_id": "old-task", "step": 2}],
                }
            ]
        )
        result = apply_memory_operations(
            bank,
            [
                {
                    "op": "refine",
                    "target_memory_id": "mem_000001",
                    "memory": {
                        "content": "more precise claim",
                        "scope": "one topic",
                        "evidence_steps": [0],
                        "confidence": 0.9,
                    },
                }
            ],
            trajectory={"task_id": "new-task", "steps": [{"index": 0}]},
        )
        self.assertEqual(len(result["applied"]), 1)
        self.assertEqual(
            bank[0]["evidence"],
            [
                {"source_task_id": "old-task", "step": 2},
                {"source_task_id": "new-task", "step": 0},
            ],
        )

    def test_sft_validation_requires_complete_episode_and_separates_replay(self):
        trajectory = {
            "task_id": "t",
            "success": True,
            "steps": [
                {
                    "index": 0,
                    "observation": "[12] button 'Continue'",
                    "action": {"action": "click", "bid": "12", "note": "continue"},
                    "action_error": None,
                },
                {
                    "index": 1,
                    "observation": "Task is complete",
                    "action": {"action": "finish", "answer": "done", "note": "done"},
                    "action_error": None,
                },
            ],
        }
        recorded = {
            "steps": [
                {
                    "source_step": 0,
                    "target_action": {
                        "action": "click",
                        "bid": "12",
                        "note": "continue",
                    },
                    "supporting_steps": [0],
                    "rationale": "recorded success",
                },
                {
                    "source_step": 1,
                    "target_action": {
                        "action": "finish",
                        "answer": "done",
                        "note": "done",
                    },
                    "supporting_steps": [1],
                    "rationale": "recorded finish",
                },
            ],
            "rationale": "complete recorded episode",
        }
        corrected = json.loads(json.dumps(recorded))
        corrected["steps"][1]["target_action"]["answer"] = "alternative"
        result = validate_sft_episodes(
            [
                recorded,
                corrected,
                {"steps": recorded["steps"][:1], "rationale": "partial"},
            ],
            trajectory=trajectory,
            allowed_actions={"click", "finish"},
            require_action_note=True,
        )
        self.assertEqual(len(result["accepted"]), 1)
        self.assertEqual(len(result["needs_replay"]), 1)
        self.assertEqual(result["needs_replay"][0]["label_type"], "corrected")
        self.assertEqual(
            result["rejected"][0]["reason"],
            "incomplete_or_nonconsecutive_episode",
        )

        preserved = validate_sft_episodes(
            normalize_sft_episodes(
                {
                    "episode": {
                        "mode": "preserve_recorded",
                        "rationale": "retain the complete successful episode",
                    }
                }
            ),
            trajectory=trajectory,
            allowed_actions={"click", "finish"},
            require_action_note=True,
        )
        self.assertEqual(len(preserved["accepted"]), 1)
        self.assertEqual(
            [step["source_step"] for step in preserved["accepted"][0]["steps"]],
            [0, 1],
        )

        audited_transcription = normalize_sft_episodes(
            {
                "episode": {
                    "task_id": "t",
                    "steps": [
                        {
                            "index": 0,
                            "action": {
                                "action": "click",
                                "bid": "12",
                                "note": "continue",
                            },
                        },
                        {
                            "index": 1,
                            "action": {
                                "action": "finish",
                                "answer": "done",
                                "note": "done",
                            },
                        },
                    ],
                }
            }
        )
        audited_result = validate_sft_episodes(
            audited_transcription,
            trajectory=trajectory,
            allowed_actions={"click", "finish"},
            require_action_note=True,
        )
        self.assertEqual(len(audited_result["accepted"]), 1)

        failed_trajectory = json.loads(json.dumps(trajectory))
        failed_trajectory["success"] = False
        rejected_preserve = validate_sft_episodes(
            normalize_sft_episodes(
                {"episode": {"mode": "preserve_recorded", "rationale": "bad"}}
            ),
            trajectory=failed_trajectory,
            allowed_actions={"click", "finish"},
            require_action_note=True,
        )
        self.assertEqual(rejected_preserve["accepted"], [])
        self.assertEqual(rejected_preserve["needs_replay"], [])
        self.assertEqual(
            rejected_preserve["rejected"][0]["reason"],
            "preserve_recorded_requires_successful_error_free_trajectory",
        )

    def test_tool_runner_reloads_raw_evidence_and_applies_audited_memory(self):
        class FakeModel:
            def __init__(self):
                self.users = []
                self.responses = [
                    {
                        "tool_calls": [
                            {
                                "name": "edit_memory",
                                "arguments": {
                                    "reason": "SECRET ROUTER CONCLUSION",
                                    "evidence_steps": [0],
                                },
                            }
                        ]
                    },
                    {
                        "operations": [
                            {
                                "op": "add",
                                "memory": {
                                    "content": "draft",
                                    "scope": "tests",
                                    "evidence_steps": [0],
                                    "confidence": 0.5,
                                },
                            }
                        ]
                    },
                    {
                        "operations": [
                            {
                                "op": "add",
                                "memory": {
                                    "content": "audited",
                                    "scope": "tests",
                                    "evidence_steps": [0],
                                    "confidence": 0.9,
                                },
                            }
                        ]
                    },
                ]

            def json_chat(self, *, system, user):
                self.users.append(user)
                parsed = self.responses.pop(0)
                return ModelReply(
                    content=json.dumps(parsed),
                    reasoning=None,
                    parsed=parsed,
                    usage={
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                )

        bank = []
        model = FakeModel()
        result = run_retention_tools(
            model,
            trajectory={
                "task_id": "t",
                "goal": "test",
                "success": True,
                "steps": [
                    {
                        "index": 0,
                        "observation": "environment evidence",
                        "action": {"action": "finish", "answer": "x"},
                    }
                ],
            },
            memory_bank=bank,
            agent_system="Return a finish action.",
            allowed_actions={"finish"},
            audit=True,
        )
        self.assertEqual(result["controller_choice"], "context_only")
        self.assertEqual(result["artifact_choice"], "context_only")
        self.assertEqual(bank[0]["content"], "audited")
        specialist_payload = json.loads(model.users[1])
        self.assertNotIn("SECRET ROUTER CONCLUSION", model.users[1])
        self.assertEqual(specialist_payload["controller_suggested_evidence_steps"], [0])

    def test_tool_runner_fails_closed_when_controller_response_is_invalid(self):
        class BrokenController:
            def json_chat(self, *, system, user):
                raise ValueError("invalid controller JSON")

        result = run_retention_tools(
            BrokenController(),
            trajectory={"task_id": "t", "goal": "test", "steps": []},
            memory_bank=[],
            agent_system="Return a finish action.",
            allowed_actions={"finish"},
        )
        self.assertEqual(result["controller_choice"], "neither")
        self.assertEqual(result["artifact_choice"], "neither")
        self.assertEqual(result["controller"]["error"]["stage"], "controller")

    def test_tool_runner_fails_closed_when_one_tool_response_is_invalid(self):
        class BrokenSftTool:
            def __init__(self):
                self.calls = 0

            def json_chat(self, *, system, user):
                self.calls += 1
                if self.calls == 1:
                    parsed = {
                        "tool_calls": [
                            {
                                "name": "build_sft_examples",
                                "arguments": {
                                    "reason": "test failure isolation",
                                    "evidence_steps": [],
                                },
                            }
                        ]
                    }
                    return ModelReply(
                        content=json.dumps(parsed),
                        reasoning=None,
                        parsed=parsed,
                        usage={
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    )
                raise ValueError("invalid tool JSON")

        result = run_retention_tools(
            BrokenSftTool(),
            trajectory={"task_id": "t", "goal": "test", "steps": []},
            memory_bank=[],
            agent_system="Return a finish action.",
            allowed_actions={"finish"},
        )
        self.assertEqual(result["controller_choice"], "sft_only")
        self.assertEqual(result["artifact_choice"], "neither")
        self.assertEqual(result["tools"]["build_sft_examples"]["error"]["stage"], "draft")
        self.assertEqual(
            result["tools"]["build_sft_examples"]["validation"]["accepted"], []
        )

    def test_recorded_episode_uses_hard_validation_without_model_reaudit(self):
        class RecordedEpisodeModel:
            def __init__(self):
                self.responses = [
                    {
                        "tool_calls": [
                            {
                                "name": "build_sft_examples",
                                "arguments": {
                                    "reason": "complete success",
                                    "evidence_steps": [0],
                                },
                            }
                        ]
                    },
                    {
                        "episode": {
                            "mode": "preserve_recorded",
                            "rationale": "complete successful episode",
                        }
                    },
                ]

            def json_chat(self, *, system, user):
                parsed = self.responses.pop(0)
                return ModelReply(
                    content=json.dumps(parsed),
                    reasoning=None,
                    parsed=parsed,
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

        model = RecordedEpisodeModel()
        result = run_retention_tools(
            model,
            trajectory={
                "task_id": "t",
                "goal": "answer",
                "success": True,
                "steps": [
                    {
                        "index": 0,
                        "observation": "answer is x",
                        "action": {"action": "finish", "answer": "x"},
                        "action_error": None,
                    }
                ],
            },
            memory_bank=[],
            agent_system="Return a finish action.",
            allowed_actions={"finish"},
            audit=True,
        )
        tool = result["tools"]["build_sft_examples"]
        self.assertEqual(len(tool["validation"]["accepted"]), 1)
        self.assertIsNone(tool["audit"])
        self.assertEqual(
            tool["audit_skipped_reason"],
            "complete_successful_recorded_episode_hard_validated",
        )
        self.assertEqual(model.responses, [])

    def test_tool_payload_keeps_every_step_with_bounded_environment_feedback(self):
        trajectory = {
            "task_id": "t",
            "steps": [
                {"index": index, "observation": str(index) + "x" * 10_000}
                for index in range(6)
            ],
        }
        payload = trajectory_for_tool(
            trajectory,
            max_total_observation_chars=12_000,
            max_per_observation_chars=10_000,
        )
        self.assertEqual([step["index"] for step in payload["steps"]], list(range(6)))
        self.assertTrue(
            all(
                "observation middle omitted" in step["observation"]
                for step in payload["steps"]
            )
        )
        self.assertLessEqual(
            sum(len(step["observation"]) for step in payload["steps"]),
            12_500,
        )

    def test_review_trajectory_keeps_outcome_before_compact_steps(self):
        trajectory = {
            "task": {"id": "t", "instruction": "i", "start_url": "https://x"},
            "final_answer": "done",
            "evaluation": {"success": True},
            "error": None,
            "steps": [
                {
                    "index": 0,
                    "observation": {
                        "url": "https://x",
                        "title": "x",
                        "text": "a" * 5000,
                        "elements": [],
                    },
                    "action": {"action": "finish", "answer": "done"},
                    "model_reasoning": "r" * 3000,
                }
            ],
        }
        compact = _trajectory_for_review(trajectory)
        self.assertEqual(compact["final_answer"], "done")
        self.assertEqual(len(compact["steps"][0]["observation"]["text"]), 4000)
        self.assertEqual(len(compact["steps"][0]["model_reasoning"]), 2500)

    def test_saved_agent_prompt_is_not_recursively_injected(self):
        steps = [
            {
                "index": 0,
                "agent_prompt": "large saved prompt",
                "action": {"action": "back"},
            }
        ]
        selected = _steps_for_agent_prompt(steps)
        self.assertNotIn("agent_prompt", selected[0])
        self.assertEqual(selected[0]["action"], {"action": "back"})

    def test_sft_state_keeps_current_evidence_without_repeating_prior_page_text(self):
        trajectory = {
            "task": {"instruction": "find the license"},
            "context_before": [{"source_task_id": "old", "content": "memory"}],
            "steps": [
                {
                    "index": 0,
                    "observation": {
                        "url": "https://example.test/one",
                        "title": "one",
                        "text": "PRIOR_RAW_PAGE_TEXT",
                        "elements": [],
                    },
                    "action": {"action": "click", "ref": 1},
                    "execution": "clicked [1]",
                },
                {
                    "index": 1,
                    "observation": {
                        "url": "https://example.test/two",
                        "title": "two",
                        "text": "License: apache-2.0",
                        "elements": [],
                    },
                    "action": {"action": "finish", "answer": "apache-2.0"},
                },
            ],
        }
        prompt = _sft_agent_prompt(trajectory, 1)
        self.assertIn("License: apache-2.0", prompt)
        self.assertIn('"content": "memory"', prompt)
        self.assertNotIn("PRIOR_RAW_PAGE_TEXT", prompt)

    def test_storage_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "one.json", {"a": 1})
            append_jsonl(root / "many.jsonl", {"b": 2})
            self.assertEqual(json.loads((root / "one.json").read_text()), {"a": 1})
            self.assertEqual(json.loads((root / "many.jsonl").read_text()), {"b": 2})


if __name__ == "__main__":
    unittest.main()
