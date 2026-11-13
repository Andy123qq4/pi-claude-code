import assert from "node:assert/strict";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createClaudeCodeProvider } from "../src/provider.ts";

test("shares concurrent status checks, isolates cancellation, and rechecks after completion", async () => {
  const root = await mkdtemp(join(tmpdir(), "pi-status-sharing-"));
  try {
    const count = join(root, "count");
    const status = join(root, "status");
    const script = join(root, "claude.js");
    await writeFile(status, "true");
    await writeFile(script, `#!/usr/bin/env node\nconst fs = require('node:fs');\nfs.appendFileSync(${JSON.stringify(count)}, 'x');\nsetTimeout(() => console.log(JSON.stringify({loggedIn: fs.readFileSync(${JSON.stringify(status)}, 'utf8') === 'true', subscriptionType: 'max'})), 300);\n`);
    const executable = process.platform === "win32" ? join(root, "claude.cmd") : script;
    if (executable !== script) await writeFile(executable, `@echo off\r\n"${process.execPath}" "${script}" %*\r\n`);
    await chmod(executable, 0o755);
    const provider = createClaudeCodeProvider();
    const ctx = { env: async (name: string) => name === "PI_CLAUDE_CODE_COMMAND" ? executable : undefined, fileExists: async () => false };
    const cancelled = new AbortController();
    const live = new AbortController();
    const first = provider.auth.apiKey!.check!({ ctx, signal: cancelled.signal });
    const second = provider.auth.apiKey!.check!({ ctx, signal: live.signal });
    setTimeout(() => cancelled.abort(), 100);
    await assert.rejects(first);
    assert.equal((await second)?.type, "oauth");
    assert.equal(await readFile(count, "utf8"), "x");
    await writeFile(status, "false");
    assert.equal(await provider.auth.apiKey!.check!({ ctx, signal: live.signal }), undefined);
    assert.equal(await readFile(count, "utf8"), "xx");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
