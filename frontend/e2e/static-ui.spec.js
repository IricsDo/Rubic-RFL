const { test, expect } = require("@playwright/test");

const API_BASE = "http://127.0.0.1:8000";
const SOLVED_STICKERS = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB";

async function routeBackendOffline(page) {
  await page.route(`${API_BASE}/**`, (route) => route.abort("failed"));
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
  await expect(page.getByRole("button", { name: "Scramble" })).toBeVisible();
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

test("scrambles, prepares a local inverse replay, and steps through it", async ({ page }) => {
  await page.getByLabel("Depth").fill("3");
  await page.getByRole("button", { name: "Scramble" }).click();

  await expect(page.locator("#solverStatus")).toHaveText("Generated 3-move scramble locally.");
  await expect(page.locator("#moveCountBadge")).toHaveText("3 moves");
  await expect(page.locator("#historyList li")).toHaveCount(3);

  await page.getByRole("button", { name: "Solve" }).click();
  await expect(page.locator("#solverStatus")).toHaveText(
    "Prepared 3 inverse-history moves locally."
  );
  await expect(page.locator("#solutionList li")).toHaveCount(3);
  await expect(page.getByRole("button", { name: "Step Forward" })).toBeEnabled();

  await page.getByRole("button", { name: "Step Forward" }).click();
  await expect(page.locator("#solverStatus")).toContainText("Applied solution move 1:");
  await expect(page.getByRole("button", { name: "Step Back" })).toBeEnabled();

  await page.getByRole("button", { name: "Step Back" }).click();
  await expect(page.locator("#solverStatus")).toHaveText("Rewound move 1.");
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
