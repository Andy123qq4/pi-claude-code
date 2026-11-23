import { chmod, writeFile } from "node:fs/promises";
import { join } from "node:path";

/**
 * Write a fake Claude Code script and return the command to put in PI_CLAUDE_CODE_COMMAND.
 *
 * Windows cannot execute a shebang script directly (spawn fails with EFTYPE), so there the command is a .cmd
 * shim that runs the script with this Node, which also exercises the extension's cmd.exe launch path.
 */
export async function writeFakeClaude(root: string, source: string, name = "fake-claude"): Promise<string> {
  const script = join(root, `${name}.mjs`);
  await writeFile(script, source, "utf8");
  await chmod(script, 0o755);
  if (process.platform !== "win32") return script;
  const shim = join(root, `${name}.cmd`);
  await writeFile(shim, `@ECHO off\r\n"${process.execPath}" "${script}" %*\r\n`, "utf8");
  return shim;
}
