"use strict";

(() => {
  const MAX_LOCAL_FILE_BYTES = 256 * 1024 * 1024;
  const LOCAL_VALIDATION_ERROR = "Replay rejected: invalid or unsupported replay document.";
  const IDENTIFIER = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
  const SHA256 = /^[0-9a-f]{64}$/;
  const RUN_STATUSES = new Set(["completed", "excluded", "failed", "interrupted"]);
  const STEP_TYPES = new Set(["natural_interaction_block", "target_event"]);
  const STEP_STATES = new Set(["complete", "partial", "unobserved", "out_of_scope"]);
  const COMPLETENESS = new Set(["complete", "partial"]);
  const FRAME_KINDS = new Set(["initialization", "main_turn", "reflection", "failure"]);
  const MEMORY_TYPES = new Set(["seed_persona", "action", "reflection"]);
  const USAGE_STATUSES = new Set(["success", "failed", "invalid_output"]);
  const CALL_TYPES = new Set(["plan", "natural_response", "target_event_response", "reflection", "importance_score"]);
  const PILOT_ROLES = new Set(["simulator", "embedding"]);
  const SAFE_FAILURE_MESSAGES = new Set([
    "Pilot run interrupted.",
    "Pilot run failed; inspect the canonical error type and attempt records.",
    "Run interrupted; inspect the canonical error type and persisted prefix.",
    "Run failed; inspect the canonical error type and persisted prefix.",
  ]);
  const FIXED_NATURAL_ORDER = [
    "teacher", "active_student", "academic_student", "quiet_student",
    "active_student", "teacher", "academic_student", "quiet_student",
    "academic_student", "active_student", "quiet_student", "teacher",
  ];
  const FORBIDDEN_KEYS = new Set([
    "api_key", "authorization", "embedding", "environment", "headers", "prompt",
    "query_embedding", "raw_prompt", "raw_provider_response", "request_body",
    "response_body", "traceback",
  ]);
  const state = { data: null, index: 0, timer: null };
  const byId = (id) => document.getElementById(id);

  const elements = {
    file: byId("replay-file"),
    empty: byId("empty-state"),
    replay: byId("replay"),
    runStatus: byId("run-status"),
    runId: byId("run-id"),
    runFacts: byId("run-facts"),
    stepList: byId("step-list"),
    previous: byId("previous"),
    play: byId("play"),
    next: byId("next"),
    speed: byId("speed"),
    frameCount: byId("frame-count"),
    frameKind: byId("frame-kind"),
    frameTitle: byId("frame-title"),
    logicalTime: byId("logical-time"),
    frameDescription: byId("frame-description"),
    frameContent: byId("frame-content"),
    retrieval: byId("retrieval-panel"),
    memory: byId("memory-panel"),
    overlay: byId("overlay-panel"),
    emptyPanel: byId("empty-panel-template"),
  };

  const clear = (node) => {
    while (node.firstChild) node.removeChild(node.firstChild);
  };

  const text = (tag, value, className) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = value == null ? "" : String(value);
    return node;
  };

  const addFact = (term, value) => {
    elements.runFacts.append(text("dt", term));
    elements.runFacts.append(text("dd", value));
  };

  const emptyPanel = (node, message = "No persisted evidence for this frame.") => {
    clear(node);
    node.append(text("p", message, "muted"));
  };

  const assertObject = (value, label) => {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`${label} must be an object`);
    }
  };

  const exactObject = (value, fields, label) => {
    assertObject(value, label);
    const keys = Object.keys(value);
    if (keys.length !== fields.length || fields.some((field) => !Object.prototype.hasOwnProperty.call(value, field))) {
      throw new Error(`${label} fields are invalid`);
    }
    return value;
  };

  const arrayValue = (value, label, minimum = 0) => {
    if (!Array.isArray(value) || value.length < minimum) throw new Error(`${label} must be an array`);
    return value;
  };

  const textValue = (value, label) => {
    if (typeof value !== "string" || value.trim().length === 0) throw new Error(`${label} must be text`);
    return value;
  };

  const nullableText = (value, label) => value === null ? null : textValue(value, label);

  const identifierValue = (value, label) => {
    if (typeof value !== "string" || !IDENTIFIER.test(value)) throw new Error(`${label} must be an identifier`);
    return value;
  };

  const nullableIdentifier = (value, label) => value === null ? null : identifierValue(value, label);

  const hashValue = (value, label) => {
    if (typeof value !== "string" || !SHA256.test(value)) throw new Error(`${label} must be a SHA-256`);
    return value;
  };

  const booleanValue = (value, label) => {
    if (typeof value !== "boolean") throw new Error(`${label} must be boolean`);
    return value;
  };

  const integerValue = (value, label, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) => {
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`${label} must be an integer`);
    return value;
  };

  const nullableInteger = (value, label, minimum = 0) => value === null ? null : integerValue(value, label, minimum);

  const finiteValue = (value, label, minimum = -Infinity, maximum = Infinity) => {
    if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
      throw new Error(`${label} must be finite`);
    }
    return value;
  };

  const enumValue = (value, allowed, label) => {
    if (!allowed.has(value)) throw new Error(`${label} is unsupported`);
    return value;
  };

  const unique = (values, label) => {
    const result = new Set(values);
    if (result.size !== values.length) throw new Error(`${label} contains duplicates`);
    return result;
  };

  const textArray = (value, label, { minimum = 0, identifiers = false, uniqueValues = false } = {}) => {
    const items = arrayValue(value, label, minimum);
    items.forEach((item, index) => (identifiers ? identifierValue : textValue)(item, `${label}[${index}]`));
    if (uniqueValues) unique(items, label);
    return items;
  };

  const validateSafeJson = (value, label, depth = 0) => {
    if (depth > 32) throw new Error(`${label} is too deeply nested`);
    if (value === null || typeof value === "string" || typeof value === "boolean") return;
    if (typeof value === "number") {
      finiteValue(value, label);
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item, index) => validateSafeJson(item, `${label}[${index}]`, depth + 1));
      return;
    }
    assertObject(value, label);
    Object.entries(value).forEach(([key, item]) => {
      if (FORBIDDEN_KEYS.has(key.toLowerCase())) throw new Error(`${label} contains a forbidden field`);
      validateSafeJson(item, `${label}.${key}`, depth + 1);
    });
  };

  const validateIssue = (issue, label) => {
    exactObject(issue, ["code", "severity", "scope", "message"], label);
    identifierValue(issue.code, `${label}.code`);
    enumValue(issue.severity, new Set(["warning", "incomplete"]), `${label}.severity`);
    textValue(issue.scope, `${label}.scope`);
    textValue(issue.message, `${label}.message`);
  };

  const validateUsageRecord = (record, label, expectedTurnId = null) => {
    exactObject(record, ["turn_id", "call_type", "model", "prompt_template_id", "status", "input_tokens", "output_tokens", "total_tokens", "error_type"], label);
    identifierValue(record.turn_id, `${label}.turn_id`);
    if (expectedTurnId !== null && record.turn_id !== expectedTurnId) throw new Error(`${label} has the wrong turn identity`);
    enumValue(record.call_type, CALL_TYPES, `${label}.call_type`);
    textValue(record.model, `${label}.model`);
    identifierValue(record.prompt_template_id, `${label}.prompt_template_id`);
    enumValue(record.status, USAGE_STATUSES, `${label}.status`);
    integerValue(record.input_tokens, `${label}.input_tokens`);
    integerValue(record.output_tokens, `${label}.output_tokens`);
    integerValue(record.total_tokens, `${label}.total_tokens`);
    if (record.total_tokens !== record.input_tokens + record.output_tokens) throw new Error(`${label} token totals are inconsistent`);
    nullableIdentifier(record.error_type, `${label}.error_type`);
    if ((record.status === "success") !== (record.error_type === null)) throw new Error(`${label} error identity is inconsistent`);
  };

  const validateCandidate = (candidate, label) => {
    exactObject(candidate, ["rank", "memory_id", "category", "raw_recency", "raw_importance", "raw_relevance", "normalized_recency", "normalized_importance", "normalized_relevance", "final_score", "selected"], label);
    integerValue(candidate.rank, `${label}.rank`, 1);
    identifierValue(candidate.memory_id, `${label}.memory_id`);
    textValue(candidate.category, `${label}.category`);
    for (const field of ["raw_recency", "raw_importance", "raw_relevance", "normalized_recency", "normalized_importance", "normalized_relevance", "final_score"]) {
      finiteValue(candidate[field], `${label}.${field}`);
    }
    booleanValue(candidate.selected, `${label}.selected`);
  };

  const validateRetrieval = (retrieval, label) => {
    exactObject(retrieval, ["query", "weights", "top_k", "candidate_count", "candidates", "selected_memory_ids", "accessed_memory_ids"], label);
    textValue(retrieval.query, `${label}.query`);
    exactObject(retrieval.weights, ["recency", "importance", "relevance"], `${label}.weights`);
    for (const field of ["recency", "importance", "relevance"]) finiteValue(retrieval.weights[field], `${label}.weights.${field}`);
    integerValue(retrieval.top_k, `${label}.top_k`, 1);
    if (retrieval.top_k !== 5) throw new Error(`${label}.top_k is unsupported`);
    integerValue(retrieval.candidate_count, `${label}.candidate_count`);
    const candidates = arrayValue(retrieval.candidates, `${label}.candidates`);
    if (retrieval.candidate_count !== candidates.length) throw new Error(`${label} candidate count is inconsistent`);
    candidates.forEach((candidate, index) => {
      validateCandidate(candidate, `${label}.candidates[${index}]`);
      if (candidate.rank !== index + 1) throw new Error(`${label} candidate rank order is invalid`);
      if (index > 0 && candidates[index - 1].final_score < candidate.final_score) throw new Error(`${label} candidate score order is invalid`);
    });
    unique(candidates.map((candidate) => candidate.memory_id), `${label}.candidates`);
    const selected = textArray(retrieval.selected_memory_ids, `${label}.selected_memory_ids`, { identifiers: true, uniqueValues: true });
    const accessed = textArray(retrieval.accessed_memory_ids, `${label}.accessed_memory_ids`, { identifiers: true, uniqueValues: true });
    const expectedSelected = candidates.slice(0, Math.min(retrieval.top_k, candidates.length)).map((candidate) => candidate.memory_id);
    if (selected.length !== expectedSelected.length || selected.some((id, index) => id !== expectedSelected[index])) throw new Error(`${label} selected Top-K prefix is invalid`);
    if (accessed.length !== selected.length || accessed.some((id, index) => id !== selected[index])) throw new Error(`${label} accessed order is invalid`);
    candidates.forEach((candidate, index) => {
      if (candidate.selected !== (index < expectedSelected.length)) throw new Error(`${label} selected flags are inconsistent`);
    });
  };

  const validateTurn = (turn, label, agentIds, expectedAgentId, expectedIndex) => {
    exactObject(turn, ["turn_id", "turn_index", "agent_id", "logical_timestamp", "completeness", "response", "plan_summary", "retrieval", "action_memory_id", "usage_records"], label);
    identifierValue(turn.turn_id, `${label}.turn_id`);
    integerValue(turn.turn_index, `${label}.turn_index`, 1);
    if (turn.turn_index !== expectedIndex) throw new Error(`${label} turn index is inconsistent`);
    identifierValue(turn.agent_id, `${label}.agent_id`);
    if (!agentIds.has(turn.agent_id) || turn.agent_id !== expectedAgentId) throw new Error(`${label} agent order is inconsistent`);
    integerValue(turn.logical_timestamp, `${label}.logical_timestamp`, 1, 104);
    enumValue(turn.completeness, COMPLETENESS, `${label}.completeness`);
    nullableText(turn.response, `${label}.response`);
    nullableText(turn.plan_summary, `${label}.plan_summary`);
    if (turn.retrieval !== null) validateRetrieval(turn.retrieval, `${label}.retrieval`);
    nullableIdentifier(turn.action_memory_id, `${label}.action_memory_id`);
    const usageRecords = arrayValue(turn.usage_records, `${label}.usage_records`);
    usageRecords.forEach((record, index) => validateUsageRecord(record, `${label}.usage_records[${index}]`, turn.turn_id));
    if (turn.completeness === "complete" && (turn.response === null || turn.plan_summary === null || turn.retrieval === null || turn.action_memory_id === null)) {
      throw new Error(`${label} is marked complete without complete persisted evidence`);
    }
  };

  const validateReflection = (reflection, label, agentIds, expectedAgentId, expectedIndex) => {
    exactObject(reflection, ["operation_id", "participant_index", "agent_id", "logical_timestamp", "completeness", "text", "category", "memory_id"], label);
    identifierValue(reflection.operation_id, `${label}.operation_id`);
    integerValue(reflection.participant_index, `${label}.participant_index`, 1);
    if (reflection.participant_index !== expectedIndex) throw new Error(`${label} participant index is inconsistent`);
    identifierValue(reflection.agent_id, `${label}.agent_id`);
    if (!agentIds.has(reflection.agent_id) || reflection.agent_id !== expectedAgentId) throw new Error(`${label} agent order is inconsistent`);
    integerValue(reflection.logical_timestamp, `${label}.logical_timestamp`, 1, 104);
    enumValue(reflection.completeness, COMPLETENESS, `${label}.completeness`);
    nullableText(reflection.text, `${label}.text`);
    nullableText(reflection.category, `${label}.category`);
    nullableIdentifier(reflection.memory_id, `${label}.memory_id`);
    if (reflection.completeness === "complete" && (reflection.text === null || reflection.category === null || reflection.memory_id === null)) {
      throw new Error(`${label} is marked complete without persisted evidence`);
    }
  };

  const validateStructuralSummary = (summary, label, run, step) => {
    if (summary === null) return;
    const shared = ["schema_version", "experiment_id", "run_id", "retrieval_config_id", "repetition_index", "step_id", "step_type", "is_target_event", "evaluation_enabled", "day", "period", "timestamp", "event_id", "ordered_main_turn_ids", "participant_ids", "per_agent_main_turn_counts", "completed_main_turn_count", "reflection_count", "final_logical_timestamp"];
    const fields = step.step_type === "target_event" ? [...shared, "ordered_responding_agent_ids", "response_count"] : shared;
    exactObject(summary, fields, label);
    if (summary.schema_version !== "1.0" || summary.experiment_id !== run.experiment_id || summary.run_id !== run.run_id || summary.retrieval_config_id !== run.retrieval_config_id || summary.repetition_index !== run.repetition_index) throw new Error(`${label} run identity is inconsistent`);
    if (summary.step_id !== step.step_id || summary.step_type !== step.step_type || summary.day !== step.day || summary.period !== step.period || summary.event_id !== step.event_id || summary.evaluation_enabled !== step.evaluation_enabled) throw new Error(`${label} step identity is inconsistent`);
    booleanValue(summary.is_target_event, `${label}.is_target_event`);
    if (summary.is_target_event !== (step.step_type === "target_event")) throw new Error(`${label} target flag is inconsistent`);
    integerValue(summary.timestamp, `${label}.timestamp`, 1, 104);
    integerValue(summary.completed_main_turn_count, `${label}.completed_main_turn_count`);
    integerValue(summary.reflection_count, `${label}.reflection_count`);
    integerValue(summary.final_logical_timestamp, `${label}.final_logical_timestamp`, 1, 104);
    const turnIds = textArray(summary.ordered_main_turn_ids, `${label}.ordered_main_turn_ids`, { identifiers: true, uniqueValues: true });
    if (turnIds.length !== step.turns.length || turnIds.some((id, index) => id !== step.turns[index].turn_id)) throw new Error(`${label} turn order is inconsistent`);
    const participants = textArray(summary.participant_ids, `${label}.participant_ids`, { identifiers: true, uniqueValues: true });
    if (participants.length !== step.participants.length || participants.some((id, index) => id !== step.participants[index])) throw new Error(`${label} participants are inconsistent`);
    assertObject(summary.per_agent_main_turn_counts, `${label}.per_agent_main_turn_counts`);
    if (Object.keys(summary.per_agent_main_turn_counts).length !== step.participants.length) throw new Error(`${label} per-agent counts are incomplete`);
    step.participants.forEach((agentId) => integerValue(summary.per_agent_main_turn_counts[agentId], `${label}.per_agent_main_turn_counts.${agentId}`));
    if (summary.completed_main_turn_count !== step.turns.length || summary.reflection_count !== step.reflections.length || summary.final_logical_timestamp !== step.final_logical_timestamp || summary.timestamp !== step.final_logical_timestamp) throw new Error(`${label} aggregate counts are inconsistent`);
    if (step.step_type === "target_event") {
      const responding = textArray(summary.ordered_responding_agent_ids, `${label}.ordered_responding_agent_ids`, { identifiers: true, uniqueValues: true });
      if (responding.length !== step.turns.length || responding.some((id, index) => id !== step.turns[index].agent_id) || summary.response_count !== step.turns.length) throw new Error(`${label} response order is inconsistent`);
      integerValue(summary.response_count, `${label}.response_count`);
    }
  };

  const validateMemory = (memory, label, agentIds) => {
    exactObject(memory, ["memory_id", "owner_agent_id", "memory_type", "category", "text", "importance", "created_at", "last_accessed_at_at_persistence", "source_step_id", "source_step_type", "source_turn_id", "source_turn_index", "seed_index", "persisted_access_timestamps"], label);
    identifierValue(memory.memory_id, `${label}.memory_id`);
    identifierValue(memory.owner_agent_id, `${label}.owner_agent_id`);
    if (!agentIds.has(memory.owner_agent_id)) throw new Error(`${label} owner is unknown`);
    enumValue(memory.memory_type, MEMORY_TYPES, `${label}.memory_type`);
    textValue(memory.category, `${label}.category`);
    textValue(memory.text, `${label}.text`);
    finiteValue(memory.importance, `${label}.importance`, 1, 10);
    integerValue(memory.created_at, `${label}.created_at`, 0, 104);
    integerValue(memory.last_accessed_at_at_persistence, `${label}.last_accessed_at_at_persistence`, memory.created_at, 104);
    identifierValue(memory.source_step_id, `${label}.source_step_id`);
    enumValue(memory.source_step_type, new Set(["agent_initialization", "natural_interaction_block", "target_event"]), `${label}.source_step_type`);
    nullableIdentifier(memory.source_turn_id, `${label}.source_turn_id`);
    nullableInteger(memory.source_turn_index, `${label}.source_turn_index`, 1);
    nullableInteger(memory.seed_index, `${label}.seed_index`, 1);
    const accesses = arrayValue(memory.persisted_access_timestamps, `${label}.persisted_access_timestamps`);
    accesses.forEach((timestamp, index) => {
      integerValue(timestamp, `${label}.persisted_access_timestamps[${index}]`, 1, 104);
      if (index > 0 && accesses[index - 1] >= timestamp) throw new Error(`${label} access timestamps are not ordered`);
    });
    if (memory.memory_type === "seed_persona") {
      if (memory.created_at !== 0 || memory.source_step_type !== "agent_initialization" || memory.source_turn_id !== null || memory.source_turn_index !== null || memory.seed_index === null) throw new Error(`${label} seed semantics are invalid`);
    } else if (memory.source_step_type === "agent_initialization" || memory.source_turn_id === null || memory.source_turn_index === null || memory.seed_index !== null) {
      throw new Error(`${label} simulation-memory source identity is invalid`);
    }
  };

  const validateFailure = (failure, run) => {
    if (failure === null) {
      if (run.source_status === "failed" || run.source_status === "interrupted") throw new Error("terminal run is missing failure metadata");
      return;
    }
    exactObject(failure, ["error_type", "safe_message", "recoverable", "retry_count", "step_id", "turn_id", "logical_timestamp"], "failure");
    if (run.source_status !== "failed" && run.source_status !== "interrupted") throw new Error("successful run contains failure metadata");
    identifierValue(failure.error_type, "failure.error_type");
    if (!SAFE_FAILURE_MESSAGES.has(failure.safe_message)) throw new Error("failure.safe_message is not approved");
    booleanValue(failure.recoverable, "failure.recoverable");
    nullableInteger(failure.retry_count, "failure.retry_count");
    nullableIdentifier(failure.step_id, "failure.step_id");
    nullableIdentifier(failure.turn_id, "failure.turn_id");
    nullableInteger(failure.logical_timestamp, "failure.logical_timestamp");
  };

  const validateEvaluationOverlay = (overlay, turnIndex, run) => {
    if (overlay === null) return;
    const hasJudgment = overlay.judgment !== null;
    const fields = ["schema_version", "evaluation_mode", "primary_eligible", "included_in_final_research_aggregation", "ground_truth_visibility", "units", "judgment", "source_bcr_export_sha256", "source_aggregate_sha256"];
    if (hasJudgment) fields.push("judgment_source_aggregate_sha256");
    exactObject(overlay, fields, "evaluation_overlay");
    if (overlay.schema_version !== "1.0") throw new Error("evaluation overlay schema is unsupported");
    enumValue(overlay.evaluation_mode, new Set(["primary", "diagnostic"]), "evaluation_overlay.evaluation_mode");
    booleanValue(overlay.primary_eligible, "evaluation_overlay.primary_eligible");
    booleanValue(overlay.included_in_final_research_aggregation, "evaluation_overlay.included_in_final_research_aggregation");
    const expectedEligibility = overlay.evaluation_mode === "primary";
    const expectedMode = run.source_status === "completed" ? "primary" : "diagnostic";
    if (overlay.evaluation_mode !== expectedMode || overlay.primary_eligible !== expectedEligibility || overlay.included_in_final_research_aggregation !== expectedEligibility || overlay.ground_truth_visibility !== "evaluation_overlay_only_not_agent_observation") throw new Error("evaluation overlay isolation is invalid");
    hashValue(overlay.source_bcr_export_sha256, "evaluation_overlay.source_bcr_export_sha256");
    hashValue(overlay.source_aggregate_sha256, "evaluation_overlay.source_aggregate_sha256");
    if (hasJudgment) hashValue(overlay.judgment_source_aggregate_sha256, "evaluation_overlay.judgment_source_aggregate_sha256");
    const units = arrayValue(overlay.units, "evaluation_overlay.units", 1);
    if (units.length !== (overlay.evaluation_mode === "primary" ? 20 : 4)) throw new Error("evaluation overlay unit count is invalid");
    const unitTurns = [];
    units.forEach((unit, index) => {
      const baseFields = ["event_id", "step_id", "turn_id", "agent_id", "logical_timestamp", "mra", "matched_count", "denominator", "expected_categories", "matched_memory_ids", "bcr_status", "source_bcr_record_sha256"];
      const judgedFields = ["bcr", "communication_style_score", "focus_consistency_score", "temporal_consistency_score", "communication_style_reason", "focus_consistency_reason", "temporal_consistency_reason"];
      exactObject(unit, hasJudgment ? [...baseFields, ...judgedFields] : baseFields, `evaluation_overlay.units[${index}]`);
      for (const field of ["event_id", "step_id", "turn_id", "agent_id"]) identifierValue(unit[field], `evaluation_overlay.units[${index}].${field}`);
      const sourceTurn = turnIndex.get(unit.turn_id);
      if (!sourceTurn || sourceTurn.step.step_type !== "target_event" || sourceTurn.step.step_id !== unit.step_id || sourceTurn.step.event_id !== unit.event_id || sourceTurn.turn.agent_id !== unit.agent_id || sourceTurn.turn.logical_timestamp !== unit.logical_timestamp) throw new Error("evaluation overlay unit does not join a source turn");
      integerValue(unit.logical_timestamp, `evaluation_overlay.units[${index}].logical_timestamp`, 1, 104);
      finiteValue(unit.mra, `evaluation_overlay.units[${index}].mra`, 0, 1);
      integerValue(unit.matched_count, `evaluation_overlay.units[${index}].matched_count`);
      integerValue(unit.denominator, `evaluation_overlay.units[${index}].denominator`, 1);
      if (unit.matched_count > unit.denominator) throw new Error("evaluation overlay matched count is invalid");
      textArray(unit.expected_categories, `evaluation_overlay.units[${index}].expected_categories`, { minimum: 1 });
      textArray(unit.matched_memory_ids, `evaluation_overlay.units[${index}].matched_memory_ids`, { identifiers: true, uniqueValues: true });
      enumValue(unit.bcr_status, new Set(["not_judged", "judged"]), `evaluation_overlay.units[${index}].bcr_status`);
      if ((unit.bcr_status === "judged") !== hasJudgment) throw new Error("evaluation overlay judgment state is inconsistent");
      hashValue(unit.source_bcr_record_sha256, `evaluation_overlay.units[${index}].source_bcr_record_sha256`);
      if (hasJudgment) {
        for (const field of ["bcr", "communication_style_score", "focus_consistency_score", "temporal_consistency_score"]) finiteValue(unit[field], `evaluation_overlay.units[${index}].${field}`, 0, 1);
        for (const field of ["communication_style_reason", "focus_consistency_reason", "temporal_consistency_reason"]) textValue(unit[field], `evaluation_overlay.units[${index}].${field}`);
      }
      unitTurns.push(unit.turn_id);
    });
    unique(unitTurns, "evaluation_overlay.units");
    if (hasJudgment) {
      if (overlay.evaluation_mode !== "primary") throw new Error("diagnostic evaluation cannot contain judgment output");
      exactObject(overlay.judgment, ["schema_version", "judged_unit_count", "run_mean_bcr", "model_metadata", "usage"], "evaluation_overlay.judgment");
      if (overlay.judgment.schema_version !== "1.0" || overlay.judgment.judged_unit_count !== units.length) throw new Error("evaluation judgment identity is invalid");
      finiteValue(overlay.judgment.run_mean_bcr, "evaluation_overlay.judgment.run_mean_bcr", 0, 1);
      assertObject(overlay.judgment.model_metadata, "evaluation_overlay.judgment.model_metadata");
      assertObject(overlay.judgment.usage, "evaluation_overlay.judgment.usage");
      validateSafeJson(overlay.judgment.model_metadata, "evaluation_overlay.judgment.model_metadata");
      validateSafeJson(overlay.judgment.usage, "evaluation_overlay.judgment.usage");
    }
  };

  const validatePilotOverlay = (overlay, run, failure) => {
    if (overlay === null) return;
    const hasReview = overlay.review !== null;
    const fields = ["schema_version", "pilot_id", "mode", "technical_success", "source_run_status", "acceptance_state", "eligible_for_manual_acceptance", "primary_eligible", "included_in_final_research_matrix", "terminal_error_type", "rejection_reasons", "usage", "attempts", "review", "source_aggregate_sha256"];
    if (hasReview) fields.push("review_source_aggregate_sha256");
    exactObject(overlay, fields, "pilot_overlay");
    if (overlay.schema_version !== "1.0" || overlay.pilot_id !== run.run_id) throw new Error("pilot overlay identity is invalid");
    enumValue(overlay.mode, new Set(["dummy", "real"]), "pilot_overlay.mode");
    booleanValue(overlay.technical_success, "pilot_overlay.technical_success");
    enumValue(overlay.source_run_status, new Set(["excluded", "failed", "interrupted"]), "pilot_overlay.source_run_status");
    if (overlay.source_run_status !== run.source_status) throw new Error("pilot overlay source status is inconsistent");
    enumValue(overlay.acceptance_state, new Set(["pending", "accepted", "rejected"]), "pilot_overlay.acceptance_state");
    booleanValue(overlay.eligible_for_manual_acceptance, "pilot_overlay.eligible_for_manual_acceptance");
    if (overlay.primary_eligible !== false || overlay.included_in_final_research_matrix !== false) throw new Error("pilot overlay research eligibility is invalid");
    nullableIdentifier(overlay.terminal_error_type, "pilot_overlay.terminal_error_type");
    const reasons = textArray(overlay.rejection_reasons, "pilot_overlay.rejection_reasons", { identifiers: true });
    if (overlay.technical_success) {
      if (overlay.terminal_error_type !== null || reasons.length !== 0 || overlay.source_run_status !== "excluded") throw new Error("successful pilot overlay is inconsistent");
    } else if (overlay.terminal_error_type === null || reasons.length === 0) {
      throw new Error("failed pilot overlay is incomplete");
    }
    if (failure !== null && overlay.source_run_status !== "excluded" && overlay.terminal_error_type !== failure.error_type) throw new Error("pilot/source terminal error identity is inconsistent");
    hashValue(overlay.source_aggregate_sha256, "pilot_overlay.source_aggregate_sha256");
    if (hasReview) hashValue(overlay.review_source_aggregate_sha256, "pilot_overlay.review_source_aggregate_sha256");

    exactObject(overlay.usage, ["max_total_tokens", "reported_known_total_tokens", "logical_call_count", "physical_attempt_count", "retry_count", "missing_total_usage_attempt_count", "budget_exceeded", "actual_total_tokens"], "pilot_overlay.usage");
    nullableInteger(overlay.usage.max_total_tokens, "pilot_overlay.usage.max_total_tokens", 1);
    for (const field of ["reported_known_total_tokens", "logical_call_count", "physical_attempt_count", "retry_count", "missing_total_usage_attempt_count"]) integerValue(overlay.usage[field], `pilot_overlay.usage.${field}`);
    booleanValue(overlay.usage.budget_exceeded, "pilot_overlay.usage.budget_exceeded");
    nullableInteger(overlay.usage.actual_total_tokens, "pilot_overlay.usage.actual_total_tokens");
    if (overlay.usage.retry_count !== overlay.usage.physical_attempt_count - overlay.usage.logical_call_count) throw new Error("pilot usage retry count is invalid");
    if (overlay.usage.missing_total_usage_attempt_count > 0) {
      if (overlay.usage.actual_total_tokens !== null) throw new Error("pilot missing usage was converted to a token total");
    } else if (overlay.usage.actual_total_tokens !== overlay.usage.reported_known_total_tokens) throw new Error("pilot actual token total is invalid");

    const attempts = arrayValue(overlay.attempts, "pilot_overlay.attempts");
    if (attempts.length !== overlay.usage.physical_attempt_count) throw new Error("pilot attempt count is inconsistent");
    let missing = 0;
    const attemptIds = [];
    attempts.forEach((attempt, index) => {
      exactObject(attempt, ["service_role", "logical_call_id", "attempt_index", "call_type", "model", "prompt_template_id", "prompt_version", "prompt_hash", "input_tokens", "output_tokens", "total_tokens", "status", "error_type"], `pilot_overlay.attempts[${index}]`);
      enumValue(attempt.service_role, PILOT_ROLES, `pilot_overlay.attempts[${index}].service_role`);
      identifierValue(attempt.logical_call_id, `pilot_overlay.attempts[${index}].logical_call_id`);
      integerValue(attempt.attempt_index, `pilot_overlay.attempts[${index}].attempt_index`, 1, 2);
      identifierValue(attempt.call_type, `pilot_overlay.attempts[${index}].call_type`);
      textValue(attempt.model, `pilot_overlay.attempts[${index}].model`);
      nullableIdentifier(attempt.prompt_template_id, `pilot_overlay.attempts[${index}].prompt_template_id`);
      nullableText(attempt.prompt_version, `pilot_overlay.attempts[${index}].prompt_version`);
      if (attempt.prompt_hash !== null) hashValue(attempt.prompt_hash, `pilot_overlay.attempts[${index}].prompt_hash`);
      for (const field of ["input_tokens", "output_tokens", "total_tokens"]) nullableInteger(attempt[field], `pilot_overlay.attempts[${index}].${field}`);
      enumValue(attempt.status, USAGE_STATUSES, `pilot_overlay.attempts[${index}].status`);
      nullableIdentifier(attempt.error_type, `pilot_overlay.attempts[${index}].error_type`);
      if ((attempt.status === "success") !== (attempt.error_type === null)) throw new Error("pilot attempt error identity is inconsistent");
      if (attempt.service_role === "embedding") {
        if (attempt.prompt_template_id !== null || attempt.prompt_version !== null || attempt.prompt_hash !== null) throw new Error("embedding attempt claims prompt identity");
      } else if (attempt.prompt_template_id === null || attempt.prompt_version === null || attempt.prompt_hash === null) throw new Error("simulator attempt lacks prompt identity");
      if (attempt.total_tokens === null) missing += 1;
      else if (attempt.input_tokens !== null && attempt.output_tokens !== null && attempt.total_tokens !== attempt.input_tokens + attempt.output_tokens) throw new Error("pilot attempt token arithmetic is invalid");
      attemptIds.push(`${attempt.service_role}:${attempt.logical_call_id}:${attempt.attempt_index}`);
    });
    unique(attemptIds, "pilot_overlay.attempts");
    const logicalCalls = new Set(attempts.map((attempt) => `${attempt.service_role}:${attempt.logical_call_id}`));
    const reportedKnownTokens = attempts.reduce((total, attempt) => total + (attempt.total_tokens === null ? 0 : attempt.total_tokens), 0);
    if (missing !== overlay.usage.missing_total_usage_attempt_count || logicalCalls.size !== overlay.usage.logical_call_count || reportedKnownTokens !== overlay.usage.reported_known_total_tokens) throw new Error("pilot usage totals are invalid");
    if (overlay.mode === "real" ? overlay.usage.max_total_tokens === null : overlay.usage.max_total_tokens !== null) throw new Error("pilot token ceiling is invalid for its mode");
    const budgetExceeded = overlay.usage.max_total_tokens !== null && overlay.usage.reported_known_total_tokens > overlay.usage.max_total_tokens;
    if (overlay.usage.budget_exceeded !== budgetExceeded) throw new Error("pilot budget state is invalid");
    const expectedManualEligibility = overlay.technical_success && overlay.mode === "real" && missing === 0 && !budgetExceeded;
    if (overlay.eligible_for_manual_acceptance !== expectedManualEligibility) throw new Error("pilot manual-acceptance eligibility is invalid");

    if (overlay.review === null) {
      if (overlay.acceptance_state !== "pending") throw new Error("unreviewed pilot has a manual decision");
    } else {
      exactObject(overlay.review, ["acceptance_state", "manual_review", "acceptance_hash", "freeze"], "pilot_overlay.review");
      enumValue(overlay.review.acceptance_state, new Set(["accepted", "rejected"]), "pilot_overlay.review.acceptance_state");
      if (overlay.review.acceptance_state !== overlay.acceptance_state || overlay.review.manual_review !== true) throw new Error("pilot review state is inconsistent");
      hashValue(overlay.review.acceptance_hash, "pilot_overlay.review.acceptance_hash");
      if (overlay.review.acceptance_state === "accepted") {
        exactObject(overlay.review.freeze, ["freeze_state", "freeze_hash", "manifest_hash", "current_drift_validation"], "pilot_overlay.review.freeze");
        if (overlay.review.freeze.freeze_state !== "accepted" || overlay.review.freeze.current_drift_validation !== "not_performed_by_visualizer") throw new Error("pilot freeze state is invalid");
        hashValue(overlay.review.freeze.freeze_hash, "pilot_overlay.review.freeze.freeze_hash");
        hashValue(overlay.review.freeze.manifest_hash, "pilot_overlay.review.freeze.manifest_hash");
        if (!overlay.technical_success || !overlay.eligible_for_manual_acceptance || overlay.mode !== "real" || overlay.usage.missing_total_usage_attempt_count !== 0) throw new Error("accepted pilot overlay is ineligible");
      } else if (overlay.review.freeze !== null) throw new Error("rejected pilot has a freeze");
    }
  };

  const validateReplay = (value) => {
    exactObject(value, ["schema_version", "source", "run", "agents", "steps", "memories", "playback", "usage", "failure", "evaluation_overlay", "pilot_overlay", "issues"], "Replay");
    if (value.schema_version !== "1.0") throw new Error("Unsupported replay schema");

    exactObject(value.source, ["source_schema_version", "aggregate_sha256", "artifact_hashes", "artifact_kind", "replay_is_research_data", "ground_truth_included", "embeddings_included", "prompts_included", "provider_responses_included"], "source");
    if (value.source.source_schema_version !== "1.0" || value.source.artifact_kind !== "phase12_read_only_replay_projection") throw new Error("source metadata is unsupported");
    hashValue(value.source.aggregate_sha256, "source.aggregate_sha256");
    const artifactPaths = [];
    arrayValue(value.source.artifact_hashes, "source.artifact_hashes", 1).forEach((item, index) => {
      exactObject(item, ["relative_path", "sha256"], `source.artifact_hashes[${index}]`);
      textValue(item.relative_path, `source.artifact_hashes[${index}].relative_path`);
      if (item.relative_path.startsWith("/") || item.relative_path.includes("\\") || item.relative_path.split("/").some((part) => part === "" || part === "." || part === "..")) throw new Error("source artifact path is invalid");
      hashValue(item.sha256, `source.artifact_hashes[${index}].sha256`);
      artifactPaths.push(item.relative_path);
    });
    unique(artifactPaths, "source.artifact_hashes");
    for (const field of ["replay_is_research_data", "ground_truth_included", "embeddings_included", "prompts_included", "provider_responses_included"]) {
      if (value.source[field] !== false) throw new Error("Replay isolation flags are invalid");
    }

    exactObject(value.run, ["experiment_id", "run_id", "retrieval_config_id", "repetition_index", "source_status", "replay_completeness", "completed_natural_blocks", "completed_target_events", "completed_main_turns", "error_count", "intended_scope", "primary_research_data"], "run");
    for (const field of ["experiment_id", "run_id", "retrieval_config_id"]) identifierValue(value.run[field], `run.${field}`);
    integerValue(value.run.repetition_index, "run.repetition_index", 1);
    enumValue(value.run.source_status, RUN_STATUSES, "run.source_status");
    enumValue(value.run.replay_completeness, new Set(["complete_source_scope", "partial_persisted_prefix"]), "run.replay_completeness");
    for (const field of ["completed_natural_blocks", "completed_target_events", "completed_main_turns", "error_count"]) integerValue(value.run[field], `run.${field}`);
    enumValue(value.run.intended_scope, new Set(["day_1_through_6", "day_1_through_2_excluded_pilot", "day_1_through_6_partial", "unknown_full_or_day_1_through_2_partial"]), "run.intended_scope");
    booleanValue(value.run.primary_research_data, "run.primary_research_data");
    const terminal = value.run.source_status === "failed" || value.run.source_status === "interrupted";
    if (value.run.primary_research_data !== (value.run.source_status === "completed") || value.run.replay_completeness !== (terminal ? "partial_persisted_prefix" : "complete_source_scope") || value.run.error_count !== (terminal ? 1 : 0)) throw new Error("run status fields are inconsistent");

    const agents = arrayValue(value.agents, "agents", 4);
    if (agents.length !== 4) throw new Error("agents must contain the fixed four participants");
    const agentIds = new Set();
    agents.forEach((agent, index) => {
      exactObject(agent, ["agent_id", "display_name", "role", "archetype", "persona_summary", "traits", "goals", "communication_style", "motivations"], `agents[${index}]`);
      identifierValue(agent.agent_id, `agents[${index}].agent_id`);
      for (const field of ["display_name", "role", "archetype", "persona_summary", "communication_style"]) textValue(agent[field], `agents[${index}].${field}`);
      for (const field of ["traits", "goals", "motivations"]) textArray(agent[field], `agents[${index}].${field}`, { minimum: 1 });
      if (agentIds.has(agent.agent_id)) throw new Error("agent IDs are not unique");
      agentIds.add(agent.agent_id);
    });

    const steps = arrayValue(value.steps, "steps", 12);
    if (steps.length !== 12) throw new Error("steps must contain the fixed timeline snapshot");
    const stepIndex = new Map();
    const turnIndex = new Map();
    const reflectionIndex = new Map();
    steps.forEach((step, index) => {
      exactObject(step, ["step_id", "step_type", "day", "period", "title", "description", "participants", "event_id", "evaluation_enabled", "state", "turns", "reflections", "final_logical_timestamp", "structural_summary", "issues"], `steps[${index}]`);
      identifierValue(step.step_id, `steps[${index}].step_id`);
      if (stepIndex.has(step.step_id)) throw new Error("step IDs are not unique");
      stepIndex.set(step.step_id, step);
      enumValue(step.step_type, STEP_TYPES, `steps[${index}].step_type`);
      integerValue(step.day, `steps[${index}].day`, 1, 6);
      textValue(step.period, `steps[${index}].period`);
      textValue(step.title, `steps[${index}].title`);
      textValue(step.description, `steps[${index}].description`);
      const participants = textArray(step.participants, `steps[${index}].participants`, { minimum: 4, identifiers: true, uniqueValues: true });
      if (participants.length !== 4 || participants.some((id) => !agentIds.has(id))) throw new Error("step participants are invalid");
      nullableIdentifier(step.event_id, `steps[${index}].event_id`);
      if ((step.step_type === "target_event") !== (step.event_id !== null)) throw new Error("step event identity is invalid");
      booleanValue(step.evaluation_enabled, `steps[${index}].evaluation_enabled`);
      enumValue(step.state, STEP_STATES, `steps[${index}].state`);
      const turns = arrayValue(step.turns, `steps[${index}].turns`);
      const expectedSpeakers = step.step_type === "natural_interaction_block" ? FIXED_NATURAL_ORDER : participants;
      if (turns.length > expectedSpeakers.length) throw new Error("step has too many turns");
      turns.forEach((turn, turnOffset) => {
        validateTurn(turn, `steps[${index}].turns[${turnOffset}]`, agentIds, expectedSpeakers[turnOffset], turnOffset + 1);
        if (turnIndex.has(turn.turn_id)) throw new Error("turn IDs are not unique");
        turnIndex.set(turn.turn_id, { step, turn });
      });
      const reflections = arrayValue(step.reflections, `steps[${index}].reflections`);
      if (reflections.length > participants.length) throw new Error("step has too many reflections");
      reflections.forEach((reflection, reflectionOffset) => {
        validateReflection(reflection, `steps[${index}].reflections[${reflectionOffset}]`, agentIds, participants[reflectionOffset], reflectionOffset + 1);
        if (reflectionIndex.has(reflection.operation_id)) throw new Error("reflection IDs are not unique");
        reflectionIndex.set(reflection.operation_id, { step, reflection });
      });
      nullableInteger(step.final_logical_timestamp, `steps[${index}].final_logical_timestamp`, 1);
      const expectedFinal = turns.length ? turns[turns.length - 1].logical_timestamp : null;
      if (step.final_logical_timestamp !== expectedFinal) throw new Error("step final timestamp is inconsistent");
      reflections.forEach((reflection) => {
        if (reflection.logical_timestamp !== step.final_logical_timestamp) throw new Error("reflection timestamp is inconsistent");
      });
      const completeExpected = turns.length === expectedSpeakers.length && reflections.length === participants.length && turns.every((turn) => turn.completeness === "complete") && reflections.every((reflection) => reflection.completeness === "complete");
      if ((step.state === "complete") !== completeExpected || (["unobserved", "out_of_scope"].includes(step.state) && (turns.length || reflections.length || step.structural_summary !== null))) throw new Error("step state is inconsistent");
      const issues = arrayValue(step.issues, `steps[${index}].issues`);
      issues.forEach((issue, issueIndex) => validateIssue(issue, `steps[${index}].issues[${issueIndex}]`));
      if ((step.state === "complete") !== (step.structural_summary !== null)) throw new Error("step structural summary state is inconsistent");
      validateStructuralSummary(step.structural_summary, `steps[${index}].structural_summary`, value.run, step);
    });

    const completeSteps = steps.filter((step) => step.state === "complete");
    if (value.run.completed_natural_blocks !== completeSteps.filter((step) => step.step_type === "natural_interaction_block").length || value.run.completed_target_events !== completeSteps.filter((step) => step.step_type === "target_event").length || value.run.completed_main_turns !== steps.flatMap((step) => step.turns).filter((turn) => turn.completeness === "complete").length) throw new Error("run completed counts do not match steps");
    if (value.run.source_status === "completed" && (value.run.completed_natural_blocks !== 7 || value.run.completed_target_events !== 5 || value.run.completed_main_turns !== 104 || value.run.intended_scope !== "day_1_through_6" || steps.some((step) => step.state !== "complete"))) throw new Error("completed run scope is invalid");
    if (value.run.source_status === "excluded" && (value.run.completed_natural_blocks !== 3 || value.run.completed_target_events !== 1 || value.run.completed_main_turns !== 40 || value.run.intended_scope !== "day_1_through_2_excluded_pilot" || steps.slice(0, 4).some((step) => step.state !== "complete") || steps.slice(4).some((step) => step.state !== "out_of_scope"))) throw new Error("excluded pilot scope is invalid");

    const memories = arrayValue(value.memories, "memories");
    const memoryIndex = new Map();
    memories.forEach((memory, index) => {
      validateMemory(memory, `memories[${index}]`, agentIds);
      if (memoryIndex.has(memory.memory_id)) throw new Error("memory IDs are not unique");
      memoryIndex.set(memory.memory_id, memory);
    });
    memories.forEach((memory) => {
      if (memory.memory_type === "seed_persona") return;
      const sourceStep = stepIndex.get(memory.source_step_id);
      if (!sourceStep || sourceStep.step_type !== memory.source_step_type) throw new Error("memory source step is invalid");
      if (memory.memory_type === "action") {
        const source = turnIndex.get(memory.source_turn_id);
        if (!source || source.step !== sourceStep || source.turn.turn_index !== memory.source_turn_index || source.turn.agent_id !== memory.owner_agent_id || source.turn.action_memory_id !== memory.memory_id) throw new Error("action memory source is invalid");
      } else {
        const source = reflectionIndex.get(memory.source_turn_id);
        if (!source || source.step !== sourceStep || source.reflection.participant_index !== memory.source_turn_index || source.reflection.agent_id !== memory.owner_agent_id || source.reflection.memory_id !== memory.memory_id) throw new Error("reflection memory source is invalid");
      }
    });
    turnIndex.forEach(({ turn }) => {
      if (turn.action_memory_id !== null && !memoryIndex.has(turn.action_memory_id)) throw new Error("turn action memory is missing");
      if (turn.retrieval !== null) turn.retrieval.candidates.forEach((candidate) => {
        const memory = memoryIndex.get(candidate.memory_id);
        if (!memory || memory.category !== candidate.category) throw new Error("retrieval candidate memory join is invalid");
      });
    });

    exactObject(value.usage, ["record_count", "known_input_tokens", "known_output_tokens", "known_total_tokens", "missing_usage_count", "records", "note"], "usage");
    for (const field of ["record_count", "known_input_tokens", "known_output_tokens", "known_total_tokens", "missing_usage_count"]) integerValue(value.usage[field], `usage.${field}`);
    if (value.usage.missing_usage_count !== 0) throw new Error("canonical source usage cannot claim missing tokens");
    textValue(value.usage.note, "usage.note");
    const usageRecords = arrayValue(value.usage.records, "usage.records");
    usageRecords.forEach((record, index) => {
      validateUsageRecord(record, `usage.records[${index}]`);
      if (!turnIndex.has(record.turn_id) && !reflectionIndex.has(record.turn_id)) throw new Error("usage record references an unknown operation");
    });
    if (value.usage.record_count !== usageRecords.length || value.usage.known_input_tokens !== usageRecords.reduce((total, record) => total + record.input_tokens, 0) || value.usage.known_output_tokens !== usageRecords.reduce((total, record) => total + record.output_tokens, 0) || value.usage.known_total_tokens !== usageRecords.reduce((total, record) => total + record.total_tokens, 0)) throw new Error("usage aggregates are inconsistent");
    turnIndex.forEach(({ turn }) => {
      const rootRecords = usageRecords.filter((record) => record.turn_id === turn.turn_id);
      if (JSON.stringify(rootRecords) !== JSON.stringify(turn.usage_records)) throw new Error("turn usage projection is inconsistent");
    });

    validateFailure(value.failure, value.run);
    const playback = arrayValue(value.playback, "playback", 1);
    const referencedTurns = [];
    const referencedReflections = [];
    let failureFrames = 0;
    playback.forEach((frame, index) => {
      exactObject(frame, ["playback_index", "frame_kind", "logical_timestamp", "step_id", "entity_id"], `playback[${index}]`);
      integerValue(frame.playback_index, `playback[${index}].playback_index`);
      if (frame.playback_index !== index) throw new Error("Playback order is corrupt");
      enumValue(frame.frame_kind, FRAME_KINDS, `playback[${index}].frame_kind`);
      nullableInteger(frame.logical_timestamp, `playback[${index}].logical_timestamp`);
      nullableIdentifier(frame.step_id, `playback[${index}].step_id`);
      identifierValue(frame.entity_id, `playback[${index}].entity_id`);
      if (frame.frame_kind === "initialization") {
        if (index !== 0 || frame.logical_timestamp !== 0 || frame.step_id !== null || frame.entity_id !== "seed_memory_initialization") throw new Error("initialization frame is invalid");
      } else if (frame.frame_kind === "main_turn") {
        const source = turnIndex.get(frame.entity_id);
        if (!source || source.step.step_id !== frame.step_id || source.turn.logical_timestamp !== frame.logical_timestamp) throw new Error("turn frame reference is invalid");
        referencedTurns.push(frame.entity_id);
      } else if (frame.frame_kind === "reflection") {
        const source = reflectionIndex.get(frame.entity_id);
        if (!source || source.step.step_id !== frame.step_id || source.reflection.logical_timestamp !== frame.logical_timestamp) throw new Error("reflection frame reference is invalid");
        referencedReflections.push(frame.entity_id);
      } else {
        failureFrames += 1;
        if (value.failure === null || frame.entity_id !== value.failure.error_type || frame.step_id !== value.failure.step_id || frame.logical_timestamp !== value.failure.logical_timestamp) throw new Error("failure frame reference is invalid");
      }
    });
    if (unique(referencedTurns, "playback turn references").size !== turnIndex.size || unique(referencedReflections, "playback reflection references").size !== reflectionIndex.size || failureFrames !== (value.failure === null ? 0 : 1)) throw new Error("playback entity coverage is incomplete");

    const issues = arrayValue(value.issues, "issues");
    issues.forEach((issue, index) => validateIssue(issue, `issues[${index}]`));
    validateEvaluationOverlay(value.evaluation_overlay, turnIndex, value.run);
    validatePilotOverlay(value.pilot_overlay, value.run, value.failure);
    return value;
  };

  const indexes = () => {
    const data = state.data;
    return {
      agents: new Map(data.agents.map((item) => [item.agent_id, item])),
      steps: new Map(data.steps.map((item) => [item.step_id, item])),
      memories: new Map(data.memories.map((item) => [item.memory_id, item])),
      turns: new Map(data.steps.flatMap((step) => step.turns.map((turn) => [turn.turn_id, { step, turn }]))),
      reflections: new Map(data.steps.flatMap((step) => step.reflections.map((item) => [item.operation_id, { step, reflection: item }]))),
    };
  };

  const renderRun = () => {
    const run = state.data.run;
    elements.runStatus.className = `status-pill ${run.source_status}`;
    elements.runStatus.textContent = run.source_status;
    elements.runId.textContent = run.run_id;
    clear(elements.runFacts);
    addFact("Experiment", run.experiment_id);
    addFact("Retrieval", run.retrieval_config_id);
    addFact("Repetition", run.repetition_index);
    addFact("Replay", run.replay_completeness);
    addFact("Main turns", run.completed_main_turns);
    addFact("Scope", run.intended_scope);
    addFact("Usage records", state.data.usage.record_count);
    addFact("Known tokens", state.data.usage.known_total_tokens);

    clear(elements.stepList);
    state.data.steps.forEach((step) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.stepId = step.step_id;
      button.append(text("span", `Day ${step.day} · ${step.title}`));
      button.append(text("span", step.state, "step-state"));
      button.addEventListener("click", () => {
        const frameIndex = state.data.playback.findIndex((frame) => frame.step_id === step.step_id);
        if (frameIndex >= 0) setFrame(frameIndex);
      });
      item.append(button);
      elements.stepList.append(item);
    });
  };

  const renderRetrieval = (turn, memoryIndex) => {
    if (!turn || !turn.retrieval) {
      emptyPanel(elements.retrieval);
      return;
    }
    clear(elements.retrieval);
    elements.retrieval.append(text("p", `Query: ${turn.retrieval.query}`, "query"));
    const weights = turn.retrieval.weights;
    elements.retrieval.append(text("p", `Weights — R ${weights.recency} · I ${weights.importance} · Rel ${weights.relevance}`, "muted"));
    turn.retrieval.candidates.forEach((candidate) => {
      const card = document.createElement("div");
      card.className = `candidate${candidate.selected ? " selected" : ""}`;
      const memory = memoryIndex.get(candidate.memory_id);
      card.append(text("p", `#${candidate.rank} · ${candidate.memory_id}${candidate.selected ? " · selected" : ""}`));
      card.append(text("p", memory ? memory.text : "Missing joined memory", memory ? "muted" : "corruption"));
      const scores = document.createElement("dl");
      [["Recency", candidate.normalized_recency], ["Importance", candidate.normalized_importance], ["Relevance", candidate.normalized_relevance], ["Final", candidate.final_score]].forEach(([label, value]) => {
        scores.append(text("dt", label));
        scores.append(text("dd", value));
      });
      card.append(scores);
      elements.retrieval.append(card);
    });
  };

  const renderMemory = (turn, reflection, memoryIndex) => {
    const ids = [];
    if (turn && turn.retrieval) ids.push(...turn.retrieval.selected_memory_ids);
    if (turn && turn.action_memory_id) ids.push(turn.action_memory_id);
    if (reflection && reflection.memory_id) ids.push(reflection.memory_id);
    if (!ids.length) {
      emptyPanel(elements.memory);
      return;
    }
    clear(elements.memory);
    [...new Set(ids)].forEach((id) => {
      const memory = memoryIndex.get(id);
      const card = document.createElement("div");
      card.className = "memory-item";
      card.append(text("p", id));
      if (memory) {
        card.append(text("p", `${memory.memory_type} · ${memory.category} · t=${memory.created_at}`, "muted"));
        card.append(text("p", memory.text));
      } else {
        card.append(text("p", "Missing joined memory", "corruption"));
      }
      elements.memory.append(card);
    });
  };

  const renderOverlay = (turn, operationId) => {
    const evaluation = state.data.evaluation_overlay;
    const pilot = state.data.pilot_overlay;
    clear(elements.overlay);
    let rendered = false;
    const usageRecords = state.data.usage.records.filter((item) => item.turn_id === operationId);
    usageRecords.forEach((usage) => {
      const card = document.createElement("div");
      card.className = "overlay-item";
      card.append(text("p", `Persisted provider usage: ${usage.call_type} · ${usage.status}`));
      card.append(text("p", `Model: ${usage.model}`, "muted"));
      card.append(text("p", `Tokens: ${usage.input_tokens} input · ${usage.output_tokens} output · ${usage.total_tokens} total`, "muted"));
      if (usage.error_type) card.append(text("p", `Safe error type: ${usage.error_type}`, "corruption"));
      elements.overlay.append(card);
      rendered = true;
    });
    if (turn && evaluation) {
      const unit = evaluation.units.find((item) => item.turn_id === turn.turn_id);
      if (unit) {
        const card = document.createElement("div");
        card.className = "overlay-item";
        card.append(text("p", `Persisted MRA: ${unit.mra}`));
        card.append(text("p", `BCR: ${unit.bcr_status}${unit.bcr == null ? "" : ` · ${unit.bcr}`}`, "muted"));
        card.append(text("p", "Evaluation overlay only — not agent-observed evidence.", "muted"));
        elements.overlay.append(card);
        rendered = true;
      }
    }
    if (pilot) {
      const card = document.createElement("div");
      card.className = "overlay-item";
      card.append(text("p", `Pilot technical success: ${pilot.technical_success}`));
      card.append(text("p", `Manual acceptance: ${pilot.acceptance_state}`, "muted"));
      card.append(text("p", `Missing usage attempts: ${pilot.usage.missing_total_usage_attempt_count}`, "muted"));
      elements.overlay.append(card);
      rendered = true;
    }
    if (!rendered) emptyPanel(elements.overlay, "No explicitly supplied overlay for this frame.");
  };

  const renderFrame = () => {
    const data = state.data;
    const frame = data.playback[state.index];
    const index = indexes();
    const step = frame.step_id ? index.steps.get(frame.step_id) : null;
    const turnEntry = index.turns.get(frame.entity_id);
    const reflectionEntry = index.reflections.get(frame.entity_id);
    const turn = turnEntry ? turnEntry.turn : null;
    const reflection = reflectionEntry ? reflectionEntry.reflection : null;
    const agent = turn ? index.agents.get(turn.agent_id) : reflection ? index.agents.get(reflection.agent_id) : null;

    elements.frameKind.textContent = frame.frame_kind.replaceAll("_", " ");
    elements.frameTitle.textContent = step ? `Day ${step.day} · ${step.title}` : frame.frame_kind === "initialization" ? "Seed memory initialization" : "Terminal run outcome";
    elements.logicalTime.textContent = frame.logical_timestamp == null ? "No timestamp" : `Logical t=${frame.logical_timestamp}`;
    elements.frameDescription.textContent = step ? step.description : "Persisted replay metadata.";
    clear(elements.frameContent);

    if (turn) {
      const meta = document.createElement("div");
      meta.className = "meta-row";
      meta.append(text("span", agent ? agent.display_name : turn.agent_id, "chip"));
      meta.append(text("span", `Turn ${turn.turn_index}`, "chip"));
      meta.append(text("span", turn.completeness, "chip"));
      elements.frameContent.append(meta);
      elements.frameContent.append(text("p", turn.response || "Response was not persisted before failure.", turn.response ? "response" : "corruption"));
      if (turn.plan_summary) elements.frameContent.append(text("p", `Persisted plan summary: ${turn.plan_summary}`, "muted"));
    } else if (reflection) {
      const meta = document.createElement("div");
      meta.className = "meta-row";
      meta.append(text("span", agent ? agent.display_name : reflection.agent_id, "chip"));
      meta.append(text("span", "post-step reflection", "chip"));
      elements.frameContent.append(meta);
      elements.frameContent.append(text("p", reflection.text || "Reflection was not persisted.", reflection.text ? "response" : "corruption"));
    } else if (frame.frame_kind === "failure" && data.failure) {
      elements.frameContent.append(text("p", data.failure.error_type, "corruption"));
      elements.frameContent.append(text("p", data.failure.safe_message));
    } else {
      elements.frameContent.append(text("p", `${data.memories.filter((item) => item.memory_type === "seed_persona").length} seed memories persisted at logical timestamp 0.`, "response"));
    }

    renderRetrieval(turn, index.memories);
    renderMemory(turn, reflection, index.memories);
    renderOverlay(turn, frame.entity_id);
    elements.frameCount.textContent = `${state.index + 1} / ${data.playback.length}`;
    elements.previous.disabled = state.index === 0;
    elements.next.disabled = state.index >= data.playback.length - 1;
    document.querySelectorAll(".step-list button").forEach((button) => {
      button.classList.toggle("active", Boolean(step && button.dataset.stepId === step.step_id));
    });
  };

  const stop = () => {
    if (state.timer !== null) window.clearInterval(state.timer);
    state.timer = null;
    elements.play.textContent = "Play";
  };

  const clearRejectedReplay = () => {
    state.data = null;
    state.index = 0;
    for (const node of [elements.runFacts, elements.stepList, elements.frameContent, elements.retrieval, elements.memory, elements.overlay]) clear(node);
    for (const node of [elements.runStatus, elements.runId, elements.frameCount, elements.frameKind, elements.frameTitle, elements.logicalTime, elements.frameDescription]) node.textContent = "";
    elements.replay.classList.add("hidden");
    elements.empty.classList.remove("hidden");
  };

  const setFrame = (index) => {
    if (!state.data) return;
    state.index = Math.max(0, Math.min(index, state.data.playback.length - 1));
    renderFrame();
  };

  const play = () => {
    if (state.timer !== null) {
      stop();
      return;
    }
    if (state.index >= state.data.playback.length - 1) setFrame(0);
    elements.play.textContent = "Pause";
    state.timer = window.setInterval(() => {
      if (state.index >= state.data.playback.length - 1) {
        stop();
      } else {
        setFrame(state.index + 1);
      }
    }, Number(elements.speed.value));
  };

  const loadFile = (file) => {
    stop();
    if (!file || file.size > MAX_LOCAL_FILE_BYTES) {
      elements.empty.textContent = "Replay file is missing or exceeds the 256 MiB viewer limit.";
      return;
    }
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      try {
        const parsed = validateReplay(JSON.parse(String(reader.result)));
        if (!parsed.playback.length) throw new Error("Replay has no playback frames");
        state.data = parsed;
        state.index = 0;
        elements.empty.classList.add("hidden");
        elements.replay.classList.remove("hidden");
        renderRun();
        renderFrame();
      } catch (error) {
        clearRejectedReplay();
        elements.empty.textContent = LOCAL_VALIDATION_ERROR;
      }
    });
    reader.addEventListener("error", () => {
      clearRejectedReplay();
      elements.empty.textContent = "The local replay file could not be read.";
    });
    reader.readAsText(file, "utf-8");
  };

  elements.file.addEventListener("change", () => loadFile(elements.file.files[0]));
  elements.previous.addEventListener("click", () => { stop(); setFrame(state.index - 1); });
  elements.next.addEventListener("click", () => { stop(); setFrame(state.index + 1); });
  elements.play.addEventListener("click", play);
  elements.speed.addEventListener("change", () => { if (state.timer !== null) { stop(); play(); } });
})();
