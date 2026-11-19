import assert from "node:assert/strict";
import test from "node:test";
import { collapseSystemMessages, type AssistantMessage, type Message, type Model, type TranscriptContext } from "@earendil-works/pi-ai";
import type { NativeFrame } from "../src/native-types.ts";
import { makeReplayDiagnostic, prepareFrames, prepareMessages } from "../src/transcript.ts";

// Scenario tests for the Anthropic message rules, checked on the final frames that are sent:
// every assistant tool_use is answered by a tool_result in the next user frame, every tool_result
// answers a tool_use of the previous assistant frame, no frame or text block is empty, roles alternate,
// tool ids match the Anthropic pattern, and an error tool_result holds text only.

const usage = {
  input: 1,
  output: 1,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 2,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const model = {
  id: "claude-opus-5-5",
  api: "claude-code-stream-json",
  provider: "claude-code",
  input: ["text", "image"],
} as unknown as Model<any>;

type Block = { type: string; id?: string; tool_use_id?: string; text?: string; is_error?: boolean; content?: unknown };

export function anthropicRuleViolations(frames: NativeFrame[]): string[] {
  const problems: string[] = [];
  for (let index = 0; index < frames.length; index++) {
    const frame = frames[index]!;
    const blocks = frame.message.content as Block[];
    if (frame.message.role !== frame.type) problems.push(`frame ${index}: role ${frame.message.role} != type ${frame.type}`);
    if (index > 0 && frames[index - 1]!.type === frame.type) problems.push(`frame ${index}: two ${frame.type} frames in a row`);
    if (blocks.length === 0) problems.push(`frame ${index}: empty content`);
    for (const block of blocks) {
      if (block.type === "text" && !block.text?.trim()) problems.push(`frame ${index}: empty text block`);
      if (block.type === "tool_use" && !/^[a-zA-Z0-9_-]{1,64}$/u.test(block.id ?? "")) {
        problems.push(`frame ${index}: invalid tool_use id`);
      }
      if (block.type === "tool_result" && block.is_error && Array.isArray(block.content)) {
        if ((block.content as Block[]).some((item) => item.type !== "text")) {
          problems.push(`frame ${index}: error tool_result with non-text content`);
        }
      }
    }
    if (frame.type === "assistant") {
      const calls = blocks.filter((block) => block.type === "tool_use").map((block) => block.id!);
      if (calls.length === 0) continue;
      const next = frames[index + 1];
      const answered = new Set(
        next?.type === "user"
          ? (next.message.content as Block[]).filter((block) => block.type === "tool_result").map((block) => block.tool_use_id!)
          : [],
      );
      for (const id of calls) if (!answered.has(id)) problems.push(`frame ${index}: tool_use without tool_result`);
    } else {
      const results = blocks.filter((block) => block.type === "tool_result");
      if (results.length === 0) continue;
      const previous = frames[index - 1];
      const calls = new Set(
        previous?.type === "assistant"
          ? (previous.message.content as Block[]).filter((block) => block.type === "tool_use").map((block) => block.id!)
          : [],
      );
      for (const result of results) {
        if (!calls.has(result.tool_use_id!)) problems.push(`frame ${index}: tool_result without tool_use`);
      }
      const firstOther = blocks.findIndex((block) => block.type !== "tool_result");
      const lastResult = blocks.findLastIndex((block) => block.type === "tool_result");
      if (firstOther !== -1 && firstOther < lastResult) problems.push(`frame ${index}: tool_result after other content`);
    }
  }
  return problems;
}

/** Same order as streamClaudeCode on the collapsed (non-native) path. */
function framesFor(messages: Message[], target: Model<any> = model): NativeFrame[] {
  const prepared = prepareMessages(messages, target);
  return prepareFrames(collapseSystemMessages({ messages: prepared } as unknown as TranscriptContext).messages);
}

function assistant(
  content: AssistantMessage["content"],
  extra: Partial<AssistantMessage> = {},
): AssistantMessage {
  return {
    role: "assistant",
    api: model.api,
    provider: model.provider,
    model: model.id,
    content,
    usage,
    stopReason: content.some((block) => block.type === "toolCall") ? "toolUse" : "stop",
    timestamp: 1,
    ...extra,
  };
}

function result(id: string, text: string, isError = false, extra: Message[] = []): Message[] {
  return [
    { role: "toolResult", toolCallId: id, toolName: "read", content: [{ type: "text", text }], isError, timestamp: 2 },
    ...extra,
  ];
}

test("scenario: restart leaves tool calls without results", () => {
  const frames = framesFor([
    { role: "system", content: "base", timestamp: 0 },
    { role: "user", content: "start", timestamp: 0 },
    assistant([
      { type: "toolCall", id: "orphan1", name: "edit", arguments: {} },
      { type: "toolCall", id: "orphan2", name: "read", arguments: {} },
    ]),
    { role: "user", content: "continue after restart", timestamp: 3 },
  ]);
  assert.deepEqual(anthropicRuleViolations(frames), []);
});

test("scenario: aborted assistant with partial tool calls", () => {
  const frames = framesFor([
    { role: "user", content: "start", timestamp: 0 },
    assistant(
      [
        { type: "text", text: "let me look" },
        { type: "toolCall", id: "partial1", name: "read", arguments: {} },
      ],
      { stopReason: "aborted", errorMessage: "aborted" },
    ),
    { role: "user", content: "try again", timestamp: 3 },
  ]);
  assert.deepEqual(anthropicRuleViolations(frames), []);
  // An aborted turn is not replayed: the model retries from the last valid state, like pi's own provider.
  assert.deepEqual(frames.map((frame) => frame.type), ["user"]);
});

test("scenario: errored assistant message", () => {
  const frames = framesFor([
    { role: "user", content: "start", timestamp: 0 },
    assistant(
      [
        { type: "text", text: "half an answer" },
        { type: "toolCall", id: "broken1", name: "read", arguments: {} },
      ],
      { stopReason: "error", errorMessage: "overloaded" },
    ),
    { role: "user", content: "retry", timestamp: 3 },
  ]);
  assert.deepEqual(anthropicRuleViolations(frames), []);
  assert.deepEqual(frames.map((frame) => frame.type), ["user"]);
});

test("scenario: model switch from Codex with | in tool call ids", () => {
  const id = "call_abc|fc_" + "x".repeat(400);
  const codex = assistant(
    [
      { type: "thinking", thinking: "", thinkingSignature: "encrypted-reasoning" },
      { type: "toolCall", id, name: "read", arguments: {}, thoughtSignature: "foreign" },
    ],
    { api: "openai-codex-responses", provider: "openai-codex", model: "gpt-6" },
  );
  const frames = framesFor([
    { role: "user", content: "start", timestamp: 0 },
    codex,
    ...result(id, "A"),
    { role: "user", content: "now on claude", timestamp: 3 },
  ]);
  assert.deepEqual(anthropicRuleViolations(frames), []);
});

test("scenario: system message between a tool call and its result", () => {
  const frames = framesFor([
    { role: "system", content: "base", timestamp: 0 },
    { role: "user", content: "start", timestamp: 0 },
    assistant([{ type: "toolCall", id: "t1", name: "read", arguments: {} }]),
    { role: "system", content: "tools changed", timestamp: 2 },
    ...result("t1", "A"),
  ]);
  assert.deepEqual(anthropicRuleViolations(frames), []);
  const results = frames.flatMap((frame) => frame.message.content as Block[]).filter((block) => block.type === "tool_result");
  assert.deepEqual(results.map((block) => [block.tool_use_id, block.is_error]), [["t1", false]]);
});

test("scenario: same-model replay keeps the native carrier and signed thinking", () => {
  const message = assistant([
    { type: "thinking", thinking: "plan", thinkingSignature: "sig" },
    { type: "text", text: "checking" },
    { type: "toolCall", id: "toolu_1", name: "read", arguments: { path: "a" } },
  ]);
  const native = {
    role: "assistant" as const,
    content: [
      { type: "thinking" as const, thinking: "plan", signature: "sig", extra: "kept" },
      { type: "text" as const, text: "checking" },
      { type: "tool_use" as const, id: "toolu_1", name: "mcp__pi__read", input: { path: "a" } },
    ],
  };
  message.diagnostics = [makeReplayDiagnostic(native, message) as never];
  const frames = framesFor([{ role: "user", content: "start", timestamp: 0 }, message, ...result("toolu_1", "A")]);
  assert.deepEqual(anthropicRuleViolations(frames), []);
  assert.deepEqual(frames[1]?.message.content, native.content);

  // Without a carrier (an old session), the signed thinking block is still replayed.
  const plain = assistant(message.content);
  const canonical = framesFor([{ role: "user", content: "start", timestamp: 0 }, plain, ...result("toolu_1", "A")]);
  assert.deepEqual((canonical[1]?.message.content as Block[]).map((block) => block.type), ["thinking", "text", "tool_use"]);
});

test("scenario: image in an error tool result", () => {
  const frames = framesFor([
    { role: "user", content: "screenshot it", timestamp: 0 },
    assistant([
      { type: "toolCall", id: "shot1", name: "screenshot", arguments: {} },
      { type: "toolCall", id: "shot2", name: "read", arguments: {} },
    ]),
    {
      role: "toolResult",
      toolCallId: "shot1",
      toolName: "screenshot",
      content: [
        { type: "text", text: "partial failure" },
        { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
      ],
      isError: true,
      timestamp: 2,
    },
    ...result("shot2", "B"),
    { role: "user", content: "next", timestamp: 3 },
  ]);
  assert.deepEqual(anthropicRuleViolations(frames), []);
  // The image is not lost: it follows the tool results, before the next user text, in one user frame.
  const last = frames.at(-1)!;
  assert.deepEqual(
    (last.message.content as Block[]).map((block) => block.type),
    ["tool_result", "tool_result", "text", "image", "text"],
  );
  assert.equal((last.message.content as Block[]).at(-1)?.text, "next");
  assert.equal(last.shouldQuery, undefined);
  assert.equal(frames[0]?.shouldQuery, false);
});
