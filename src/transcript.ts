import {
  getDeclaredTools,
  getInitialSystemMessage,
  hasToolRedefinitions,
  renderSystemMessageUpdate,
  type AssistantMessage,
  type ImageContent,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type SystemMessage,
  type TextContent,
  type Tool,
  type ToolResultMessage,
} from "@earendil-works/pi-ai";
import { supportsAdaptiveThinking, supportsDisabledThinking } from "./model-catalog.ts";
import type { NativeContentBlock, NativeFrame, NativeMessage } from "./native-types.ts";
import { splitErrorToolResultImages, transformMessages } from "./vendor/transform-messages.ts";

export const TOOL_PREFIX = "mcp__pi__";

/**
 * Map a tool_use name from Claude Code to Pi's tool name. The model sometimes drops the MCP prefix
 * ("powershell" for "mcp__pi__powershell") or changes case ("Bash"): exact match first, then a unique
 * case-insensitive match. Unknown names retain upstream behavior (only the exact prefix is stripped),
 * so Pi answers "Tool X not found"
 * and the model can correct itself, instead of the whole turn failing.
 */
export function resolveToolName(nativeName: string, toolNames: Set<string>): { name: string; known: boolean } {
  const prefixed = nativeName.slice(0, TOOL_PREFIX.length).toLowerCase() === TOOL_PREFIX;
  const stripped = prefixed ? nativeName.slice(TOOL_PREFIX.length) : nativeName;
  if (toolNames.has(stripped)) return { name: stripped, known: true };
  const folded = stripped.toLowerCase();
  const matches = [...toolNames].filter((name) => name.toLowerCase() === folded);
  if (matches.length === 1) return { name: matches[0]!, known: true };
  return { name: nativeName.startsWith(TOOL_PREFIX) ? nativeName.slice(TOOL_PREFIX.length) : nativeName, known: false };
}

/** Rewrite known tool_use names to their declared native form so history replays the name the API was given. */
export function canonicalizeToolUseNames(message: NativeMessage, toolNames: Set<string>): void {
  for (const block of message.content) {
    if (block.type !== "tool_use") continue;
    const resolved = resolveToolName(block.name, toolNames);
    if (resolved.known) block.name = `${TOOL_PREFIX}${resolved.name}`;
  }
}

export const NATIVE_REPLAY_DIAGNOSTIC = "claude-code.native-assistant.v1";
export const NATIVE_TOOL_CHANGES_BETA = "mid-conversation-tool-changes-2026-07-01";

const DEFERRED_TOOL_PLACEHOLDER = {
  name: "__pi_deferred_placeholder__",
  description: "Reserved placeholder. Never available. Never call this.",
  input_schema: { type: "object", properties: {}, required: [] },
  defer_loading: true,
};

export interface NativeSystemInjection {
  /** Insert immediately before this zero-based assistant message, or at the request tail. */
  beforeAssistant: number | null;
  content: Array<Record<string, unknown>>;
}

export interface NativeTranscriptPlan {
  requestTools: Array<Record<string, unknown>>;
  systemInjections: NativeSystemInjection[];
}

interface ReplayCarrier {
  version: 1;
  message: NativeMessage;
  projection: AssistantProjection;
  initialSystemMessages?: unknown[];
}

interface AssistantProjection {
  text: string;
  toolCalls: Array<{ id: string; name: string; input: Record<string, unknown> }>;
}

function sanitizeSurrogates(text: string): string {
  return text.replace(/[\uD800-\uDFFF]/gu, "\uFFFD");
}

/** Same rule as pi-ai's Anthropic provider: ids must match ^[a-zA-Z0-9_-]+$ and be at most 64 characters. */
function normalizeToolCallId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
}

/**
 * Apply pi-ai's shared transcript rules first, as pi's own Anthropic provider does: synthetic
 * results for unanswered tool calls, no errored or aborted assistant turns, foreign thinking and
 * signatures dropped, foreign tool call ids normalized, system messages held after tool results.
 * On top of that, a native replay carrier is only valid for the model that produced it.
 */
export function prepareMessages(messages: Message[], model: Model<any>): Message[] {
  return transformMessages(messages, model, normalizeToolCallId).map((message) => {
    if (
      message.role !== "assistant" ||
      (message.provider === model.provider && message.api === model.api && message.model === model.id)
    ) {
      return message;
    }
    return {
      ...message,
      diagnostics: message.diagnostics?.filter((item) => item.type !== NATIVE_REPLAY_DIAGNOSTIC),
    };
  });
}

function clone<T>(value: T): T {
  return structuredClone(value);
}

function assistantProjection(message: AssistantMessage): AssistantProjection {
  const text = message.content
    .filter((block): block is TextContent => block.type === "text")
    .map((block) => block.text)
    .join("")
    .trim();
  const toolCalls = message.content
    .filter((block) => block.type === "toolCall")
    .map((block) => ({ id: block.id, name: block.name, input: clone(block.arguments) as Record<string, unknown> }));
  return { text, toolCalls };
}

function isReplayCarrier(value: unknown): value is ReplayCarrier {
  if (!value || typeof value !== "object") return false;
  const carrier = value as Partial<ReplayCarrier>;
  return carrier.version === 1 && !!carrier.message && !!carrier.projection;
}

function replayCarrier(message: AssistantMessage): ReplayCarrier | undefined {
  const diagnostic = message.diagnostics?.find((item) => item.type === NATIVE_REPLAY_DIAGNOSTIC);
  const candidate = diagnostic?.details?.carrier;
  return isReplayCarrier(candidate) ? candidate : undefined;
}

export function getInitialSystemMessages(messages: Message[]): unknown[] | undefined {
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (message?.role !== "assistant") continue;
    const value = replayCarrier(message)?.initialSystemMessages;
    if (value) return clone(value);
  }
  return undefined;
}

function projectionsEqual(left: AssistantProjection, right: AssistantProjection): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function imageBlock(block: ImageContent): Record<string, unknown> {
  return {
    type: "image",
    source: { type: "base64", media_type: block.mimeType, data: block.data },
  };
}

function userContent(content: string | (TextContent | ImageContent)[]): unknown[] {
  // Claude Code drops blank text blocks from user messages, so one kept here
  // would have no counterpart in its request.
  if (typeof content === "string") return content.trim() ? [{ type: "text", text: sanitizeSurrogates(content) }] : [];
  return content
    .filter((block) => block.type !== "text" || block.text.trim())
    .map((block) =>
      block.type === "text" ? { type: "text", text: sanitizeSurrogates(block.text) } : imageBlock(block),
    );
}

function toolResultContent(content: ToolResultMessage["content"]): string | unknown[] {
  const hasImages = content.some((block) => block.type === "image");
  if (!hasImages) {
    return sanitizeSurrogates(
      content
        .filter((block): block is TextContent => block.type === "text")
        .map((block) => block.text)
        .join("\n"),
    );
  }
  return content.map((block) =>
    block.type === "text" ? { type: "text", text: sanitizeSurrogates(block.text) } : imageBlock(block),
  );
}

function canonicalAssistantContent(message: AssistantMessage): NativeContentBlock[] {
  const blocks: NativeContentBlock[] = [];
  for (const block of message.content) {
    if (block.type === "text" && block.text.trim()) {
      blocks.push({ type: "text", text: sanitizeSurrogates(block.text) });
    } else if (block.type === "thinking" && block.thinking.trim() && block.thinkingSignature) {
      if (block.redacted) {
        blocks.push({ type: "redacted_thinking", data: block.thinkingSignature });
      } else {
        blocks.push({
          type: "thinking",
          thinking: sanitizeSurrogates(block.thinking),
          signature: block.thinkingSignature,
        });
      }
    } else if (block.type === "toolCall") {
      blocks.push({
        type: "tool_use",
        id: block.id,
        name: `${TOOL_PREFIX}${block.name}`,
        input: clone(block.arguments) as Record<string, unknown>,
      });
    }
  }
  return blocks;
}

function assistantContent(message: AssistantMessage): NativeContentBlock[] {
  const carrier = replayCarrier(message);
  if (carrier && projectionsEqual(assistantProjection(message), carrier.projection)) {
    return clone(carrier.message.content);
  }

  // A changed projection must not restore stale signed thinking. For messages
  // produced by this provider without a carrier (for example an old session),
  // canonical signed blocks are still the best available replay.
  if (carrier) {
    return canonicalAssistantContent({
      ...message,
      content: message.content.filter((block) => block.type !== "thinking"),
    });
  }
  return canonicalAssistantContent(message);
}

export function prepareFrames(messages: Message[]): NativeFrame[] {
  const frames: NativeFrame[] = [];
  // Anthropic rejects images in an error tool result, and tool results must come first in their
  // user message: the images follow the last tool result of the run, as in pi-ai (#10794).
  let errorImages: unknown[] = [];
  const flushErrorImages = () => {
    if (errorImages.length) frames.at(-1)!.message.content.push(...errorImages);
    errorImages = [];
  };

  for (const message of messages) {
    if (message.role === "system") continue;
    if (message.role !== "toolResult") flushErrorImages();

    let role: "user" | "assistant";
    let blocks: unknown[];
    let model: string | undefined;

    if (message.role === "user") {
      role = "user";
      blocks = userContent(message.content);
    } else if (message.role === "assistant") {
      role = "assistant";
      blocks = assistantContent(message);
      // Claude Code strips signed thinking from replayed turns whose model it can't confirm.
      // Prefer the model that answered: aliases like "default" can't be confirmed and may move.
      model = message.responseModel ?? message.model;
    } else {
      role = "user";
      const split = splitErrorToolResultImages(message);
      blocks = [
        {
          type: "tool_result",
          tool_use_id: message.toolCallId,
          content: toolResultContent(split?.content ?? message.content),
          is_error: message.isError,
        },
      ];
      if (split) {
        errorImages.push({ type: "text", text: sanitizeSurrogates(split.caption) }, ...split.images.map(imageBlock));
      }
    }

    if (blocks.length === 0) continue;
    const previous = frames.at(-1);
    if (role === "user" && previous?.type === "user") {
      previous.message.content.push(...blocks);
    } else {
      frames.push({ type: role, message: { role, content: blocks, ...(model && { model }) } });
    }
  }
  flushErrorImages();

  const final = frames.at(-1);
  if (!final || final.type !== "user" || final.message.content.length === 0) {
    throw new Error("Claude Code history must end in a non-empty user or tool-result message");
  }

  for (let index = 0; index < frames.length - 1; index++) {
    if (frames[index]?.type === "user") frames[index]!.shouldQuery = false;
  }
  return frames;
}

function withoutNullVariant(schema: unknown): unknown {
  if (Array.isArray(schema)) return schema.map(withoutNullVariant);
  if (!schema || typeof schema !== "object") return schema;

  const input = schema as Record<string, unknown>;
  const output: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) output[key] = withoutNullVariant(value);

  if (Array.isArray(output.type) && output.type.includes("null")) {
    const remaining = output.type.filter((value) => value !== "null");
    output.type = remaining.length === 1 ? remaining[0] : remaining;
  }

  if (Array.isArray(output.anyOf)) {
    const nonNull = output.anyOf.filter(
      (branch) => !(branch && typeof branch === "object" && (branch as Record<string, unknown>).type === "null"),
    );
    if (nonNull.length === 1 && nonNull.length !== output.anyOf.length) {
      const replacement = nonNull[0];
      delete output.anyOf;
      if (replacement && typeof replacement === "object") Object.assign(output, replacement);
    }
  }
  return output;
}

export function normalizeInputSchema(schema: unknown): Record<string, unknown> {
  const normalized = withoutNullVariant(schema) as Record<string, unknown>;
  for (const key of ["oneOf", "allOf", "anyOf"]) delete normalized[key];
  if (!normalized.type) normalized.type = "object";
  if (normalized.type === "object" && (!normalized.properties || typeof normalized.properties !== "object")) {
    normalized.properties = {};
  }
  return normalized;
}

/**
 * A model sometimes sends an array- or object-typed argument as a JSON-encoded
 * string (`{"questions": "[{...}]"}`). Pi stores the raw arguments in the session,
 * and renderers read them unvalidated, so a string where the schema declares an
 * array or object crashes the TUI on every re-render. Parse such a value back
 * when, and only when, it parses to the declared type.
 */
export function coerceStringifiedArguments<T>(args: T, schema: Record<string, unknown> | undefined): T {
  const properties = schema?.properties;
  if (!args || typeof args !== "object" || Array.isArray(args) || !properties || typeof properties !== "object") {
    return args;
  }
  let result: Record<string, unknown> | undefined;
  for (const [key, value] of Object.entries(args as Record<string, unknown>)) {
    if (typeof value !== "string") continue;
    const declared = (properties as Record<string, { type?: unknown } | undefined>)[key]?.type;
    if (declared !== "array" && declared !== "object") continue;
    try {
      const parsed = JSON.parse(value) as unknown;
      const matches = declared === "array" ? Array.isArray(parsed) : !!parsed && typeof parsed === "object" && !Array.isArray(parsed);
      if (matches) (result ??= { ...(args as Record<string, unknown>) })[key] = parsed;
    } catch {
      // Not JSON: leave the value for the tool's own validation to reject.
    }
  }
  return (result ?? args) as T;
}

export interface ToolManifestEntry {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

/** Pi tools are declared as `mcp__pi__<name>`; the native API caps tool names at 64 characters. */
const VALID_TOOL_NAME = /^[A-Za-z0-9_-]{1,50}$/u;

export interface SkippedTool {
  name: string;
  reason: "invalid-name" | "duplicate";
}

/**
 * A tool Claude Code cannot declare is left out of the request instead of
 * failing it. Otherwise one long MCP tool name loaded into the transcript makes
 * every later turn of that branch fail. Only the declaration is skipped: the
 * tool stays registered in Pi.
 */
export function prepareTools(tools: Tool[]): {
  manifest: ToolManifestEntry[];
  nativeTools: Array<Record<string, unknown>>;
  names: Set<string>;
  skipped: SkippedTool[];
} {
  const names = new Set<string>();
  const manifest: ToolManifestEntry[] = [];
  const nativeTools: Array<Record<string, unknown>> = [];
  const skipped: SkippedTool[] = [];

  for (const tool of tools) {
    if (!VALID_TOOL_NAME.test(tool.name)) {
      skipped.push({ name: tool.name, reason: "invalid-name" });
      continue;
    }
    if (names.has(tool.name)) {
      skipped.push({ name: tool.name, reason: "duplicate" });
      continue;
    }
    names.add(tool.name);
    const schema = normalizeInputSchema(tool.parameters);
    manifest.push({ name: tool.name, description: tool.description, inputSchema: schema });
    nativeTools.push({
      name: `${TOOL_PREFIX}${tool.name}`,
      description: tool.description,
      input_schema: schema,
    });
  }
  return { manifest, nativeTools, names, skipped };
}

function nativeSystemBlocks(message: SystemMessage, declared: Set<string>): Array<Record<string, unknown>> {
  const blocks: Array<Record<string, unknown>> = [];
  const text = renderSystemMessageUpdate(message);
  if (text) blocks.push({ type: "text", text: sanitizeSurrogates(text) });
  // A reference to a tool that was left out of the request would be rejected.
  for (const tool of (message.toolsRemoved ?? []).filter((tool) => declared.has(tool.name))) {
    blocks.push({
      type: "tool_removal",
      tool: { type: "tool_reference", name: `${TOOL_PREFIX}${tool.name}` },
    });
  }
  for (const tool of (message.toolsAdded ?? []).filter((tool) => declared.has(tool.name))) {
    blocks.push({
      type: "tool_addition",
      tool: { type: "tool_reference", name: `${TOOL_PREFIX}${tool.name}` },
    });
  }
  return blocks;
}

/**
 * Build Anthropic's cache-preserving native representation of transcript tool changes.
 * The placeholder keeps deferred-tool prompt scaffolding stable from the first request.
 */
export function prepareNativeTranscript(
  messages: Message[],
  supported: boolean,
): NativeTranscriptPlan | undefined {
  const initial = getInitialSystemMessage(messages);
  const initialTools = initial?.toolsAdded ?? [];
  if (!supported || initialTools.length === 0 || hasToolRedefinitions(messages)) return undefined;

  const declared = getDeclaredTools(messages);
  const prepared = prepareTools(declared);
  const nativeByName = new Map(
    prepared.nativeTools.map((tool) => [String(tool.name).slice(TOOL_PREFIX.length), tool]),
  );
  const initialNames = new Set(initialTools.map((tool) => tool.name));
  const initialNative = initialTools
    .filter((tool) => nativeByName.has(tool.name))
    .map((tool) => structuredClone(nativeByName.get(tool.name)!));
  if (initialNative.length === 0) return undefined;
  initialNative.at(-1)!.cache_control = { type: "ephemeral", ttl: "1h" };
  const laterNative = declared
    .filter((tool) => !initialNames.has(tool.name) && nativeByName.has(tool.name))
    .map((tool) => ({ ...structuredClone(nativeByName.get(tool.name)!), defer_loading: true }));

  const systemInjections: NativeSystemInjection[] = [];
  let pending: Array<Array<Record<string, unknown>>> = [];
  let assistantOrdinal = 0;
  const flush = (beforeAssistant: number | null) => {
    for (const content of pending) systemInjections.push({ beforeAssistant, content });
    pending = [];
  };

  for (let index = 1; index < messages.length; index++) {
    const message = messages[index]!;
    if (message.role === "system") {
      const blocks = nativeSystemBlocks(message, prepared.names);
      if (blocks.length) pending.push(blocks);
    } else if (message.role === "assistant" && assistantContent(message).length > 0) {
      flush(assistantOrdinal);
      assistantOrdinal += 1;
    }
  }
  flush(null);

  return {
    requestTools: [...initialNative, structuredClone(DEFERRED_TOOL_PLACEHOLDER), ...laterNative],
    systemInjections,
  };
}

export function buildExtraBody(
  modelId: string,
  nativeTools: Array<Record<string, unknown>>,
  options?: SimpleStreamOptions,
): Record<string, unknown> {
  const body: Record<string, unknown> = { tools: nativeTools };
  if (options?.maxTokens !== undefined) body.max_tokens = options.maxTokens;

  if (options?.reasoning) {
    if (supportsAdaptiveThinking(modelId)) body.thinking = { type: "adaptive", display: "summarized" };
    body.output_config = {
      effort: options.reasoning === "minimal" ? "low" : options.reasoning,
    };
  } else if (!supportsDisabledThinking(modelId)) {
    // Models such as Opus 5.5 reject `disabled`; use the lowest adaptive effort instead.
    body.thinking = { type: "adaptive" };
    body.output_config = { effort: "low" };
  } else {
    body.thinking = { type: "disabled" };
    // Native clear-thinking edits are invalid when thinking is disabled.
    body.context_management = { edits: [] };
  }
  return body;
}

export function makeReplayDiagnostic(
  message: NativeMessage,
  output: AssistantMessage,
  initialSystemMessages?: unknown[],
) {
  return {
    type: NATIVE_REPLAY_DIAGNOSTIC,
    timestamp: Date.now(),
    details: {
      carrier: {
        version: 1,
        message: clone(message),
        projection: assistantProjection(output),
        ...(initialSystemMessages ? { initialSystemMessages: clone(initialSystemMessages) } : {}),
      },
    },
  };
}
