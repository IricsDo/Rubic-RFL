const { test, expect } = require("@playwright/test");
const fs = require("node:fs");
const path = require("node:path");

const API_BASE = "http://127.0.0.1:8000";
const REPORT_PATH = path.resolve(__dirname, "../../reports/frontend-fps-smoke.json");
const TARGET_FPS = 60;
const SAMPLE_DURATION_MS = Number(process.env.RUBIC_FRONTEND_FPS_SAMPLE_MS || 1800);
const MIN_AVERAGE_FPS = Number(
  process.env.RUBIC_FRONTEND_MIN_AVERAGE_FPS || (process.env.CI ? 45 : 55)
);
const MIN_P10_FPS = Number(process.env.RUBIC_FRONTEND_MIN_P10_FPS || (process.env.CI ? 30 : 40));

async function routeBackendOffline(page) {
  await page.route(`${API_BASE}/**`, (route) => route.abort("failed"));
}

async function sampleCubeTransformFps(page, durationMs = SAMPLE_DURATION_MS) {
  return page.evaluate(async ({ durationMs, targetFps }) => {
    const cube = document.querySelector("#cube");
    if (!cube) throw new Error("Cube element was not found.");

    const frameDeltas = [];
    let startedAt = 0;
    let previousFrameAt = 0;

    return new Promise((resolve) => {
      function summarize(finishedAt) {
        const elapsedMs = Math.max(1, finishedAt - startedAt);
        const elapsedSeconds = elapsedMs / 1000;
        const fpsSamples = frameDeltas
          .filter((delta) => delta > 0)
          .map((delta) => 1000 / delta)
          .sort((left, right) => left - right);
        const p10Index = Math.max(0, Math.floor(fpsSamples.length * 0.1) - 1);
        const averageFps = frameDeltas.length / elapsedSeconds;
        const droppedFrameThresholdMs = 1000 / (targetFps / 2);
        const droppedFrames = frameDeltas.filter((delta) => delta > droppedFrameThresholdMs).length;

        resolve({
          targetFps,
          elapsedMs: Number(elapsedMs.toFixed(1)),
          frames: frameDeltas.length,
          averageFps: Number(averageFps.toFixed(1)),
          p10Fps: Number((fpsSamples[p10Index] || 0).toFixed(1)),
          slowestFrameMs: Number((Math.max(0, ...frameDeltas) || 0).toFixed(1)),
          droppedFrames,
          droppedFrameRatio: Number((droppedFrames / Math.max(1, frameDeltas.length)).toFixed(4)),
        });
      }

      function tick(now) {
        if (!startedAt) {
          startedAt = now;
          previousFrameAt = now;
        } else {
          frameDeltas.push(now - previousFrameAt);
          previousFrameAt = now;
        }

        const elapsed = now - startedAt;
        const orbit = elapsed / 10;
        const wobble = Math.sin(elapsed / 280) * 8;
        cube.style.setProperty("--rx", `${-26 + wobble}deg`);
        cube.style.setProperty("--ry", `${-38 + orbit}deg`);

        if (elapsed < durationMs) {
          window.requestAnimationFrame(tick);
          return;
        }
        summarize(now);
      }

      window.requestAnimationFrame(tick);
    });
  }, { durationMs, targetFps: TARGET_FPS });
}

async function writeFpsReport(testInfo, metrics) {
  const report = {
    name: "frontend-animation-fps",
    timestamp: new Date().toISOString(),
    budgets: {
      targetFps: TARGET_FPS,
      minAverageFps: MIN_AVERAGE_FPS,
      minP10Fps: MIN_P10_FPS,
      sampleDurationMs: SAMPLE_DURATION_MS,
    },
    metrics,
  };

  fs.mkdirSync(path.dirname(REPORT_PATH), { recursive: true });
  fs.writeFileSync(REPORT_PATH, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await testInfo.attach("frontend-fps-smoke", {
    body: JSON.stringify(report, null, 2),
    contentType: "application/json",
  });
}

test.beforeEach(async ({ page }) => {
  await routeBackendOffline(page);
  await page.goto("/");
});

test("keeps the cube viewport near the 60 FPS animation target", async ({ page }, testInfo) => {
  await expect(page.getByLabel("Interactive cube viewport")).toBeVisible();
  await expect(page.locator("#cube .sticker")).toHaveCount(54);

  const metrics = await sampleCubeTransformFps(page);
  await writeFpsReport(testInfo, metrics);

  expect(metrics.averageFps).toBeGreaterThanOrEqual(MIN_AVERAGE_FPS);
  expect(metrics.p10Fps).toBeGreaterThanOrEqual(MIN_P10_FPS);
  expect(metrics.frames).toBeGreaterThan(0);
});
