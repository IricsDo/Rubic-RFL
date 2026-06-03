const { test, expect } = require("@playwright/test");

const API_BASE = "http://127.0.0.1:8000";
const SOLVED_STICKERS = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB";

async function routeBackendOffline(page) {
  await page.route(`${API_BASE}/**`, (route) => route.abort("failed"));
}

async function setScrambleDepth(page, targetDepth) {
  const depthInput = page.locator("#depthInput");
  const currentDepth = Number(await depthInput.inputValue());
  const stepCount = Math.abs(targetDepth - currentDepth);
  const buttonName = targetDepth > currentDepth ? "Increase depth" : "Decrease depth";
  const button = page.getByRole("button", { name: buttonName });

  for (let index = 0; index < stepCount; index += 1) {
    await button.click();
  }

  await expect(depthInput).toHaveValue(String(targetDepth));
}

test.beforeEach(async ({ page }) => {
  await routeBackendOffline(page);
  await page.goto("/");
});

test("loads the cube workspace and local controls", async ({ page }) => {
  await expect(
    page.getByRole("heading", { name: "Real-time Rubik's Cube Solver" })
  ).toBeVisible();
  await expect(page.getByLabel("Interactive cube viewport")).toBeVisible();
  await expect(page.locator("#cube .sticker")).toHaveCount(54);
  await expect(page.getByRole("button", { name: "Scramble", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Classical" })).toHaveAttribute(
    "aria-pressed",
    "true"
  );
  await expect(page.getByRole("button", { name: "RL" })).toHaveAttribute(
    "aria-pressed",
    "false"
  );
  await expect(page.locator("#backendBadge")).toHaveText("Local mode");
  await expect(page.locator("#savedSessionsList")).toContainText("Connect the API to load history.");
});

test("switches the UI language between English and Vietnamese", async ({ page }) => {
  const languageBox = await page.locator(".language-control").boundingBox();
  const statusBox = await page.locator(".status-strip").boundingBox();
  expect(languageBox).not.toBeNull();
  expect(statusBox).not.toBeNull();
  expect(statusBox.y).toBeGreaterThan(languageBox.y + languageBox.height + 6);

  await page.locator("#languageVnButton").click();

  await expect(page.locator("html")).toHaveAttribute("lang", "vi");
  await expect(page.getByRole("heading", { name: "Trình giải Rubik thời gian thực" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Xáo trộn", exact: true })).toBeVisible();
  await expect(page.locator("#backendBadge")).toHaveText("Chế độ cục bộ");
  await expect(page.locator("#savedSessionsList")).toContainText("Kết nối API để tải lịch sử.");

  await page.locator("#languageEnButton").click();

  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(
    page.getByRole("heading", { name: "Real-time Rubik's Cube Solver" })
  ).toBeVisible();
});

test("scrambles, solves automatically, and can replay after stepping back", async ({ page }) => {
  await setScrambleDepth(page, 3);
  await page.getByRole("button", { name: "Scramble", exact: true }).click();

  await expect(page.locator("#solverStatus")).toHaveText("Generated 3-move scramble locally.");
  await expect(page.locator("#moveCountBadge")).toHaveText("3 moves");
  await expect(page.locator("#historyList li")).toHaveCount(3);

  await page.getByRole("button", { name: "Solve" }).click();
  await expect(page.locator("#solutionList li")).toHaveCount(3);
  await expect(page.locator("#solvedBadge")).toHaveText("Solved", { timeout: 5_000 });
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeDisabled();

  await expect(page.getByRole("button", { name: "Step Back" })).toBeEnabled();
  await page.getByRole("button", { name: "Step Back" }).click();
  await expect(page.locator("#solverStatus")).toHaveText("Rewound move 3.");
  await expect(page.locator("#solvedBadge")).toHaveText("Unsolved");
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeEnabled();

  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.locator("#solvedBadge")).toHaveText("Solved", { timeout: 2_000 });
});

test("prevents direct depth typing and caps the stepper at 30 moves", async ({ page }) => {
  const depthInput = page.locator("#depthInput");
  const increaseButton = page.getByRole("button", { name: "Increase depth" });
  const decreaseButton = page.getByRole("button", { name: "Decrease depth" });

  await expect(depthInput).toHaveAttribute("readonly", "");
  await expect(depthInput).toHaveAttribute("tabindex", "-1");
  await expect(depthInput).toHaveCSS("pointer-events", "none");
  await setScrambleDepth(page, 30);
  await expect(depthInput).toHaveValue("30");
  await expect(increaseButton).toBeDisabled();
  await expect(decreaseButton).toBeEnabled();

  await page.getByRole("button", { name: "Scramble", exact: true }).click();

  await expect(depthInput).toHaveValue("30");
  await expect(page.locator("#solverStatus")).toHaveText("Generated 30-move scramble locally.");
  await expect(page.locator("#moveCountBadge")).toHaveText("30 moves");
  await expect(page.locator("#historyList li")).toHaveCount(30);
});

test("keeps solve disabled while a scramble request is pending", async ({ page }) => {
  await page.unroute(`${API_BASE}/**`);
  await page.route(`${API_BASE}/**`, async (route) => {
    if (route.request().url().endsWith("/cube/scramble")) {
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    await route.abort("failed");
  });

  await setScrambleDepth(page, 3);
  await page.getByRole("button", { name: "Scramble", exact: true }).click();

  await expect(page.locator("#solverStatus")).toHaveText("Generating 3-move scramble...");
  await expect(page.getByRole("button", { name: "Solve" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeDisabled();
  await expect(page.locator("#solverStatus")).toHaveText("Generated 3-move scramble locally.");
  await expect(page.getByRole("button", { name: "Solve" })).toBeEnabled();
});

test("imports an RL replay package and renders decision/search traces", async ({ page }) => {
  const replayPackage = {
    schema_version: "rubic-rfl-replay-v1",
    session_id: "e2e-rl-replay",
    solver: "rl-policy-search",
    initial_state: {
      stickers: SOLVED_STICKERS,
      history: ["R"],
    },
    moves: ["R'"],
    model: {
      version: "e2e-model",
      checkpoint: "e2e-policy.npz",
    },
    search: {
      strategy: "policy-guided-search",
      max_depth: 1,
      beam_width: 2,
      top_k: 2,
      expanded_states: 1,
      visited_states: 1,
      depth_reached: 1,
      trace: {
        records: [
          {
            node_id: "root",
            depth: 0,
            path: [],
            score: 0.9,
            stickers: SOLVED_STICKERS,
            candidates: [
              { move: "R'", probability: 0.8, outcome: "kept" },
              { move: "U", probability: 0.2, outcome: "pruned" },
            ],
          },
        ],
      },
    },
    steps: [
      {
        step: 1,
        move: "R'",
        decision: {
          selected_move: "R'",
          confidence: 0.8,
          top_candidates: [
            { move: "R'", probability: 0.8 },
            { move: "U", probability: 0.2 },
          ],
        },
      },
    ],
  };

  await page.getByLabel("State or replay package JSON").fill(JSON.stringify(replayPackage));
  await page.getByRole("button", { name: "Import" }).click();

  await expect(page.locator("#solverStatus")).toHaveText("Replay package imported.");
  await expect(page.getByRole("button", { name: "RL" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#rlDecisionPanel")).toBeVisible();
  await expect(page.locator("#rlStrategyBadge")).toHaveText("e2e-model");
  await expect(page.locator("#rlDecisionStats")).toContainText("policy guided search");
  await expect(page.locator("#rlDecisionList")).toContainText("1. R'");
  await expect(page.locator("#rlSearchTraceBadge")).toHaveText("1 records");
  await expect(page.locator("#rlSearchTraceList")).toContainText("score 0.90");
  await expect(page.getByRole("button", { name: "Export Replay" })).toBeEnabled();
});
