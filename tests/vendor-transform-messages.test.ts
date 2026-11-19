import assert from "node:assert/strict";
import test from "node:test";
import type { AssistantMessage, Message, Model, ToolResultMessage } from "@earendil-works/pi-ai";
import * as upstream from "@earendil-works/pi-ai/api/transform-messages";
import * as vendored from "../src/vendor/transform-messages.ts";

// src/vendor/transform-messages.ts is a copy of pi-ai's transformMessages, because pi does not
// expose it to extensions. This test fails when the installed pi-ai behaves differently, so a
// pi-ai update that changes the rules is noticed. It checks the development dependency, not the
// pi runtime that loads the extension.

const usage = {
  input: 1,
  output: 1,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 2,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const claude = { id: "claude-opus-5-5", api: "claude-code-stream-json", provider: "claude-code", input: ["text", "image"] } as unknown as Model<any>;
const textOnly = { ...claude, input: ["text"] } as unknown as Model<any>;
const normalize = (id: string) => id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);

function assistant(content: AssistantMessage["content"], extra: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    role: "assistant",
    api: claude.api,
    provider: claude.provider,
    model: claude.id,
    content,
    usage,
    stopReason: content.some((block) => block.type === "toolCall") ? "toolUse" : "stop",
    timestamp: 1,
    ...extra,
  };
}

const foreignId = "call_abc|fc_" + "x".repeat(300);
const errorWithImage: ToolResultMessage = {
  role: "toolResult",
  toolCallId: "img",
  toolName: "screenshot",
  content: [
    { type: "text", text: "failed" },
    { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
    { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
  ],
  isError: true,
  timestamp: 2,
};

const transcript: Message[] = [
  { role: "system", content: "base", timestamp: 0 },
  { role: "user", content: [{ type: "text", text: "go" }, { type: "image", data: "aGVsbG8=", mimeType: "image/png" }], timestamp: 0 },
  assistant([
    { type: "thinking", thinking: "plan", thinkingSignature: "sig" },
    { type: "thinking", thinking: " ", thinkingSignature: "" },
    { type: "toolCall", id: "a1", name: "read", arguments: {} },
    { type: "toolCall", id: "a2", name: "read", arguments: {} },
  ]),
  { role: "system", content: "between", timestamp: 2 },
  { role: "toolResult", toolCallId: "a1", toolName: "read", content: [{ type: "text", text: "A" }], isError: false, timestamp: 2 },
  { role: "user", content: "interrupted", timestamp: 3 },
  assistant([{ type: "toolCall", id: "aborted", name: "read", arguments: {} }], { stopReason: "aborted" }),
  assistant([{ type: "text", text: "boom" }], { stopReason: "error" }),
  assistant(
    [
      { type: "thinking", thinking: "", thinkingSignature: "encrypted", redacted: true },
      { type: "text", text: "foreign", textSignature: "x" },
      { type: "toolCall", id: foreignId, name: "read", arguments: {}, thoughtSignature: "t" },
    ],
    { api: "openai-codex-responses", provider: "openai-codex", model: "gpt-6" },
  ),
  { role: "toolResult", toolCallId: foreignId, toolName: "read", content: [{ type: "text", text: "B" }], isError: false, timestamp: 4 },
  assistant([{ type: "toolCall", id: "img", name: "screenshot", arguments: {} }]),
  errorWithImage,
  assistant([{ type: "toolCall", id: "tail", name: "read", arguments: {} }]),
];

test("vendored transformMessages matches the installed pi-ai", (t) => {
  t.mock.method(Date, "now", () => 42);
  for (const model of [claude, textOnly]) {
    for (const normalizer of [normalize, undefined]) {
      assert.deepEqual(
        vendored.transformMessages(structuredClone(transcript), model, normalizer),
        upstream.transformMessages(structuredClone(transcript), model, normalizer),
      );
    }
  }
  assert.deepEqual(vendored.splitErrorToolResultImages(errorWithImage), upstream.splitErrorToolResultImages(errorWithImage));
});
