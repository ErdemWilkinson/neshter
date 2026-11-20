import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { rmSync, mkdirSync } from 'node:fs';
import { createServer } from 'node:net';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const E2E_ROOT = 'D:/neshter/e2e-work';

export interface RunningServer {
  url: string;
  origin: string;
  port: number;
  token: string;
  child: ChildProcess;
  /** Resolves when the server process has exited. */
  exited: Promise<void>;
  stop(): Promise<void>;
}

/** Wipes and recreates `D:\neshter\e2e-work\<name>`. */
export function freshWorkdir(name: string): string {
  if (!/^[\w-]+$/.test(name)) throw new Error(`bad workdir name: ${name}`);
  const dir = join(E2E_ROOT, name);
  rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
  mkdirSync(dir, { recursive: true });
  return dir;
}

function freePort(): Promise<number> {
  return new Promise((resolvePort, reject) => {
    const probe = createServer();
    probe.once('error', reject);
    probe.listen(0, '127.0.0.1', () => {
      const address = probe.address();
      const port = typeof address === 'object' && address !== null ? address.port : 0;
      probe.close(() => resolvePort(port));
    });
  });
}

function killTree(child: ChildProcess): void {
  if (child.pid === undefined || child.exitCode !== null) return;
  // The venv python.exe may be a launcher that owns the real interpreter: take the whole tree down.
  spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
}

export async function startServer(opts: { workdir: string; port?: number; token?: string }): Promise<RunningServer> {
  const port = opts.port ?? (await freePort());
  const token = opts.token ?? randomBytes(18).toString('base64url');
  const python = process.env['NESHTER_PYTHON'] ?? 'D:/neshter/venv/Scripts/python.exe';
  const child = spawn(python, [resolve(here, 'serve_e2e.py'), '--port', String(port), '--workdir', opts.workdir, `--token=${token}`], {
    env: { ...process.env, PYTHONIOENCODING: 'utf-8', CUDA_VISIBLE_DEVICES: '-1' }, // '-1', not '': an empty value does not hide the GPU here, and the e2e flows expect the CPU defaults
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });
  let log = '';
  child.stdout?.on('data', (chunk: Buffer) => (log += chunk.toString('utf8')));
  child.stderr?.on('data', (chunk: Buffer) => (log += chunk.toString('utf8')));
  const exited = new Promise<void>((done) => child.once('exit', () => done()));

  const origin = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + 120_000; // the first start trains a tokenizer and writes the model
  for (;;) {
    if (child.exitCode !== null) throw new Error(`E2E server exited early (${child.exitCode}):\n${log}`);
    try {
      const response = await fetch(`${origin}/api/session`, { headers: { 'X-Neshter-Token': token } });
      if (response.ok) break;
    } catch {
      // not listening yet
    }
    if (Date.now() > deadline) {
      killTree(child);
      throw new Error(`E2E server did not start:\n${log}`);
    }
    await new Promise((r) => setTimeout(r, 250));
  }

  // The app starts in English unless a language is saved; these flows read the Turkish UI, so save Turkish.
  const headers = { 'X-Neshter-Token': token, 'Content-Type': 'application/json' };
  const saved: unknown = await (await fetch(`${origin}/api/prefs`, { headers })).json();
  await fetch(`${origin}/api/prefs`, { method: 'PUT', headers, body: JSON.stringify({ ...(saved as object), lang: 'tr' }) });

  return {
    url: `${origin}/#token=${token}`,
    origin,
    port,
    token,
    child,
    exited,
    stop: async () => {
      killTree(child);
      await exited;
    },
  };
}
