import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("https://photonic-signal-lab.example/", {
      headers: {
        accept: "text/html",
        host: "photonic-signal-lab.example",
        "x-forwarded-host": "photonic-signal-lab.example",
        "x-forwarded-proto": "https",
      },
    }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the interactive photonic signal lab", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /PHOTONIC SIGNAL LAB/);
  assert.match(html, /Stress the fibre\. Test the receiver\./);
  assert.match(html, /Photonic reservoir/);
  assert.match(html, /Python-verified benchmark/);
  assert.match(html, /74\.2%/);
  assert.match(html, /role="tablist"/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("ships accessible controls and absolute social metadata", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /aria-label="Recovery model"/);
  assert.match(html, /type="range"/);
  assert.match(html, /role="img"/);
  assert.match(html, /https:\/\/photonic-signal-lab\.example\/og\.png/);
  assert.match(html, /summary_large_image/);
  await access(new URL("../public/og.png", import.meta.url));
  const packageJson = await readFile(new URL("../package.json", import.meta.url), "utf8");
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});
