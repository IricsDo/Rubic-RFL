const FACE_ORDER = ["U", "R", "F", "D", "L", "B"];
const FACE_TO_AXIS = {
  U: ["y", 1],
  D: ["y", -1],
  R: ["x", 1],
  L: ["x", -1],
  F: ["z", 1],
  B: ["z", -1],
};
const AXIS_INDEX = { x: 0, y: 1, z: 2 };
const SUFFIXES = ["", "'", "2"];
const ALL_MOVES = FACE_ORDER.flatMap((face) => SUFFIXES.map((suffix) => face + suffix));
const DEFAULT_API_BASE = "http://127.0.0.1:8000";

const elements = {
  cube: document.querySelector("#cube"),
  cubeStage: document.querySelector("#cubeStage"),
  moveButtons: document.querySelector("#moveButtons"),
  depthInput: document.querySelector("#depthInput"),
  seedInput: document.querySelector("#seedInput"),
  scrambleButton: document.querySelector("#scrambleButton"),
  resetButton: document.querySelector("#resetButton"),
  solveButton: document.querySelector("#solveButton"),
  classicalModeButton: document.querySelector("#classicalModeButton"),
  rlModeButton: document.querySelector("#rlModeButton"),
  playButton: document.querySelector("#playButton"),
  stepBackButton: document.querySelector("#stepBackButton"),
  stepForwardButton: document.querySelector("#stepForwardButton"),
  progressBar: document.querySelector("#progressBar"),
  solverStatus: document.querySelector("#solverStatus"),
  historyList: document.querySelector("#historyList"),
  solutionList: document.querySelector("#solutionList"),
  sequenceInput: document.querySelector("#sequenceInput"),
  applySequenceButton: document.querySelector("#applySequenceButton"),
  stateJson: document.querySelector("#stateJson"),
  exportButton: document.querySelector("#exportButton"),
  exportReplayButton: document.querySelector("#exportReplayButton"),
  importButton: document.querySelector("#importButton"),
  refreshSessionsButton: document.querySelector("#refreshSessionsButton"),
  saveReplayButton: document.querySelector("#saveReplayButton"),
  savedSessionsBadge: document.querySelector("#savedSessionsBadge"),
  savedSessionsList: document.querySelector("#savedSessionsList"),
  apiBaseInput: document.querySelector("#apiBaseInput"),
  apiCheckButton: document.querySelector("#apiCheckButton"),
  backendBadge: document.querySelector("#backendBadge"),
  validityBadge: document.querySelector("#validityBadge"),
  solvedBadge: document.querySelector("#solvedBadge"),
  moveCountBadge: document.querySelector("#moveCountBadge"),
  viewResetButton: document.querySelector("#viewResetButton"),
  rlDecisionPanel: document.querySelector("#rlDecisionPanel"),
  rlStrategyBadge: document.querySelector("#rlStrategyBadge"),
  rlDecisionStats: document.querySelector("#rlDecisionStats"),
  rlCurrentState: document.querySelector("#rlCurrentState"),
  rlDecisionList: document.querySelector("#rlDecisionList"),
  rlSearchTraceBlock: document.querySelector("#rlSearchTraceBlock"),
  rlSearchTraceBadge: document.querySelector("#rlSearchTraceBadge"),
  rlSearchTraceList: document.querySelector("#rlSearchTraceList"),
};

function key(position, normal) {
  return `${position.join(",")}|${normal.join(",")}`;
}

function faceLocation(face, row, col) {
  if (face === "U") return { position: [col - 1, 1, row - 1], normal: [0, 1, 0] };
  if (face === "D") return { position: [col - 1, -1, 1 - row], normal: [0, -1, 0] };
  if (face === "R") return { position: [1, 1 - row, 1 - col], normal: [1, 0, 0] };
  if (face === "L") return { position: [-1, 1 - row, col - 1], normal: [-1, 0, 0] };
  if (face === "F") return { position: [col - 1, 1 - row, 1], normal: [0, 0, 1] };
  if (face === "B") return { position: [1 - col, 1 - row, -1], normal: [0, 0, -1] };
  throw new Error(`Unknown face ${face}`);
}

const indexToLocation = [];
const locationToIndex = new Map();

for (const face of FACE_ORDER) {
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) {
      const location = faceLocation(face, row, col);
      locationToIndex.set(key(location.position, location.normal), indexToLocation.length);
      indexToLocation.push(location);
    }
  }
}

const cubiePositions = [];
for (const x of [-1, 0, 1]) {
  for (const y of [-1, 0, 1]) {
    for (const z of [-1, 0, 1]) {
      if (x !== 0 || y !== 0 || z !== 0) cubiePositions.push([x, y, z]);
    }
  }
}

function solvedStickers() {
  return FACE_ORDER.flatMap((face) => Array(9).fill(face));
}

function rotateVector(vector, axis, quarterTurns) {
  let [x, y, z] = vector;
  const turns = ((quarterTurns % 4) + 4) % 4;
  for (let index = 0; index < turns; index += 1) {
    if (axis === "x") [x, y, z] = [x, -z, y];
    if (axis === "y") [x, y, z] = [z, y, -x];
    if (axis === "z") [x, y, z] = [-y, x, z];
  }
  return [x, y, z];
}

function normalizeMove(move) {
  const value = String(move).trim();
  if (!ALL_MOVES.includes(value)) throw new Error(`Unsupported move: ${value}`);
  return value;
}

function parseMoves(value) {
  if (Array.isArray(value)) return value.map(normalizeMove);
  const text = String(value || "").trim();
  if (!text) return [];
  if (/[\s,]+/.test(text)) {
    return text.split(/[\s,]+/).filter(Boolean).map(normalizeMove);
  }

  const moves = [];
  const pattern = /([URFDLB])(['2]?)/g;
  let match = pattern.exec(text);
  let cursor = 0;
  while (match) {
    if (match.index !== cursor) throw new Error(`Cannot parse near: ${text.slice(cursor)}`);
    moves.push(normalizeMove(match[0]));
    cursor = pattern.lastIndex;
    match = pattern.exec(text);
  }
  if (cursor !== text.length) throw new Error(`Cannot parse near: ${text.slice(cursor)}`);
  return moves;
}

function inverseMove(move) {
  const value = normalizeMove(move);
  if (value.endsWith("'")) return value[0];
  if (value.endsWith("2")) return value;
  return `${value}'`;
}

function inverseSequence(moves) {
  return parseMoves(moves).slice().reverse().map(inverseMove);
}

function applyMoveToStickers(stickers, move) {
  const value = normalizeMove(move);
  const face = value[0];
  const suffix = value.slice(1);
  const [axis, layerSign] = FACE_TO_AXIS[face];
  const turns = suffix === "2" ? 2 : suffix === "'" ? -1 : 1;
  const quarterTurns = -layerSign * turns;
  const axisIndex = AXIS_INDEX[axis];
  const next = stickers.slice();

  indexToLocation.forEach((location, sourceIndex) => {
    if (location.position[axisIndex] !== layerSign) return;
    const nextPosition = rotateVector(location.position, axis, quarterTurns);
    const nextNormal = rotateVector(location.normal, axis, quarterTurns);
    const targetIndex = locationToIndex.get(key(nextPosition, nextNormal));
    next[targetIndex] = stickers[sourceIndex];
  });

  return next;
}

function applyMoves(moves, recordHistory = true) {
  parseMoves(moves).forEach((move) => {
    state.stickers = applyMoveToStickers(state.stickers, move);
    if (recordHistory) state.history.push(move);
  });
  state.validation = null;
}

function isSolved(stickers) {
  return stickers.every((value, index) => value === solvedStickers()[index]);
}

function validateCounts(stickers) {
  const counts = Object.fromEntries(FACE_ORDER.map((face) => [face, 0]));
  const errors = [];
  stickers.forEach((sticker) => {
    if (sticker in counts) counts[sticker] += 1;
    else errors.push(`Unknown sticker ${sticker}`);
  });
  FACE_ORDER.forEach((face) => {
    if (counts[face] !== 9) errors.push(`${face} appears ${counts[face]} times`);
  });
  return { valid: errors.length === 0, errors, counts };
}

function cubePayload() {
  return {
    stickers: state.stickers.join(""),
    history: state.history,
  };
}

function applyApiCube(payload) {
  state.stickers = String(payload.stickers || "").split("");
  state.history = parseMoves(payload.history || []);
  state.validation = payload.validation || null;
}

async function requestBackend(path, payload, options = {}) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), options.timeout || 1500);
  const method = options.method || "POST";

  try {
    const response = await fetch(`${state.apiBase}${path}`, {
      method,
      headers: method === "GET" ? undefined : { "Content-Type": "application/json" },
      body: method === "GET" ? undefined : JSON.stringify(payload || {}),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`API ${response.status}`);
    return await response.json();
  } finally {
    window.clearTimeout(timeout);
  }
}

async function tryBackend(path, payload, options = {}) {
  try {
    const data = await requestBackend(path, payload, options);
    state.backendOnline = true;
    return data;
  } catch (error) {
    state.backendOnline = false;
    state.lastApiError = error.name === "AbortError" ? "API request timed out" : error.message;
    return null;
  }
}

function websocketSolveUrl(sessionId) {
  const base = new URL(state.apiBase);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/ws/solve/${encodeURIComponent(sessionId)}`;
  base.search = "";
  base.hash = "";
  return base.toString();
}

function definedEntries(payload) {
  return Object.fromEntries(
    Object.entries(payload).filter(([, value]) => value !== undefined && value !== null)
  );
}

function mergeRlStreamMetadata(payload) {
  const existingSteps = rlDecisionSteps().slice();
  state.rlDetails = {
    ...(state.rlDetails || {}),
    ...definedEntries({
      strategy: payload.strategy,
      model_version: payload.model_version || payload.solver,
      model_checkpoint: payload.model_checkpoint,
      max_depth: payload.max_depth,
      beam_width: payload.beam_width,
      top_k: payload.top_k,
      expanded_states: payload.expanded_states,
      visited_states: payload.visited_states,
      depth_reached: payload.search_depth,
    }),
    steps: existingSteps,
  };
  return state.rlDetails;
}

function applyRlDecisionEvent(payload) {
  const details = mergeRlStreamMetadata(payload);
  const steps = rlDecisionSteps().slice();
  const stepNumber = Number(payload.step) || steps.length + 1;
  steps[stepNumber - 1] = {
    step: stepNumber,
    selected_move: payload.selected_move,
    confidence: payload.confidence,
    top_candidates: Array.isArray(payload.top_candidates) ? payload.top_candidates : [],
  };
  details.steps = steps.filter(Boolean);
  state.replayIndex = Math.max(0, Math.min(state.solution.length, stepNumber - 1));
}

function applyRlMoveEvent(payload) {
  const move = normalizeMove(payload.move);
  const stepNumber = Number(payload.step) || state.solution.length + 1;
  const nextSolution = state.solution.slice();
  nextSolution[stepNumber - 1] = move;
  state.solution = nextSolution.filter(Boolean);

  const stickers = String(payload.stickers || "");
  state.stickers = stickers.length === 54 ? stickers.split("") : applyMoveToStickers(state.stickers, move);
  state.validation = null;
  state.replayIndex = state.solution.length;
}

function requestRlSolveStream(payload, options = {}) {
  return new Promise((resolve, reject) => {
    if (!("WebSocket" in window)) {
      reject(new Error("WebSocket is not available"));
      return;
    }

    const sessionId = `rl-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const socket = new WebSocket(websocketSolveUrl(sessionId));
    const timeoutMs = options.timeout || 8000;
    let settled = false;

    function settle(callback) {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeout);
      callback();
    }

    const timeout = window.setTimeout(() => {
      settle(() => {
        socket.close();
        reject(new Error("RL stream timed out"));
      });
    }, timeoutMs);

    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({ ...payload, solver: "rl" }));
    });

    socket.addEventListener("message", (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (error) {
        settle(() => reject(error));
        return;
      }

      if (message.event === "started") {
        state.backendOnline = true;
        state.replayPackage = null;
        state.rlDetails = {
          strategy: "policy-guided-stream",
          model_version: message.solver || "rl-policy",
          steps: [],
        };
        updateUi("RL stream started.");
        return;
      }

      if (message.event === "decision") {
        applyRlDecisionEvent(message);
        updateUi(`RL selected ${message.selected_move || "a move"}.`);
        return;
      }

      if (message.event === "move") {
        applyRlMoveEvent(message);
        updateUi(`RL applied move ${message.step}: ${message.move}.`);
        return;
      }

      if (message.event === "completed") {
        const result = message.result || {};
        if (message.replay_package) state.replayPackage = message.replay_package;
        if (result.details) state.rlDetails = result.details;
        if (Array.isArray(result.moves)) state.solution = parseMoves(result.moves);
        state.replayIndex = state.solution.length;
        settle(() => {
          socket.close();
          resolve(result);
        });
        return;
      }

      if (message.event === "error") {
        settle(() => reject(new Error(message.message || "RL stream failed")));
      }
    });

    socket.addEventListener("error", () => {
      settle(() => reject(new Error("RL stream connection failed")));
    });

    socket.addEventListener("close", () => {
      if (!settled) settle(() => reject(new Error("RL stream closed before completion")));
    });
  });
}

async function tryRlSolveStream(payload, options = {}) {
  try {
    const data = await requestRlSolveStream(payload, options);
    state.backendOnline = true;
    return data;
  } catch (error) {
    state.backendOnline = false;
    state.lastApiError = error.message;
    return null;
  }
}

async function checkBackend(announce = true) {
  state.apiBase = elements.apiBaseInput.value.trim() || DEFAULT_API_BASE;
  localStorage.setItem("rubic-rfl-api-base", state.apiBase);
  const data = await tryBackend("/health", null, { method: "GET", timeout: 1200 });
  if (data?.status === "ok") {
    state.backendOnline = true;
    if (announce) updateUi("Backend API connected.");
    return true;
  }
  state.backendOnline = false;
  if (announce) updateUi(`Backend offline. Using local fallback. ${state.lastApiError || ""}`.trim());
  return false;
}

function hashSeed(seed) {
  let value = 2166136261;
  const text = String(seed || Date.now());
  for (let index = 0; index < text.length; index += 1) {
    value ^= text.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

function seededRandom(seed) {
  let value = hashSeed(seed);
  return () => {
    value = (Math.imul(1664525, value) + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function generateScramble(depth, seed) {
  const random = seededRandom(seed);
  const moves = [];
  let previousAxis = null;
  for (let index = 0; index < depth; index += 1) {
    const faces = FACE_ORDER.filter((face) => FACE_TO_AXIS[face][0] !== previousAxis);
    const face = faces[Math.floor(random() * faces.length)];
    const suffix = SUFFIXES[Math.floor(random() * SUFFIXES.length)];
    moves.push(face + suffix);
    previousAxis = FACE_TO_AXIS[face][0];
  }
  return moves;
}

function normalClass(normal) {
  const [x, y, z] = normal;
  if (z === 1) return "face-front";
  if (z === -1) return "face-back";
  if (x === 1) return "face-right";
  if (x === -1) return "face-left";
  if (y === 1) return "face-up";
  return "face-down";
}

function rlDecisionSteps() {
  return Array.isArray(state.rlDetails?.steps) ? state.rlDetails.steps : [];
}

function searchTracePayload() {
  const directTrace = state.rlDetails?.search_trace;
  if (directTrace && Array.isArray(directTrace.records)) return directTrace;

  const replayTrace = state.replayPackage?.search?.trace;
  if (replayTrace && Array.isArray(replayTrace.records)) return replayTrace;
  return null;
}

function activeDecisionIndex() {
  const steps = rlDecisionSteps();
  if (!steps.length) return -1;
  return Math.min(state.replayIndex, steps.length - 1);
}

function activeRlMoveForHighlight() {
  const steps = rlDecisionSteps();
  if (!steps.length || state.solverMode !== "rl" || state.replayIndex >= steps.length) return null;
  try {
    return normalizeMove(steps[state.replayIndex]?.selected_move || "");
  } catch {
    return null;
  }
}

function isLocationAffectedByMove(location, move) {
  if (!move) return false;
  const face = move[0];
  const [axis, layerSign] = FACE_TO_AXIS[face];
  return location.position[AXIS_INDEX[axis]] === layerSign;
}

function visibleNormals(position) {
  const normals = [];
  if (position[0] === 1) normals.push([1, 0, 0]);
  if (position[0] === -1) normals.push([-1, 0, 0]);
  if (position[1] === 1) normals.push([0, 1, 0]);
  if (position[1] === -1) normals.push([0, -1, 0]);
  if (position[2] === 1) normals.push([0, 0, 1]);
  if (position[2] === -1) normals.push([0, 0, -1]);
  return normals;
}

function renderCube() {
  elements.cube.innerHTML = "";
  elements.cube.style.setProperty("--rx", `${view.rx}deg`);
  elements.cube.style.setProperty("--ry", `${view.ry}deg`);
  const highlightedMove = activeRlMoveForHighlight();

  cubiePositions.forEach((position) => {
    const cubie = document.createElement("div");
    cubie.className = "cubie";
    cubie.style.transform = `translate3d(calc(${position[0]} * var(--gap)), calc(${-position[1]} * var(--gap)), calc(${position[2]} * var(--gap)))`;

    visibleNormals(position).forEach((normal) => {
      const index = locationToIndex.get(key(position, normal));
      const sticker = state.stickers[index];
      const stickerLocation = { position, normal };
      const face = document.createElement("div");
      face.className = `sticker ${sticker.toLowerCase()} ${normalClass(normal)}`;
      if (isLocationAffectedByMove(stickerLocation, highlightedMove)) {
        face.classList.add("rl-highlight");
      }
      face.setAttribute("aria-hidden", "true");
      cubie.append(face);
    });

    elements.cube.append(cubie);
  });
}

function renderTimeline(listElement, moves, activeIndex = -1) {
  listElement.innerHTML = "";
  if (!moves.length) {
    const item = document.createElement("li");
    item.textContent = "None";
    listElement.append(item);
    return;
  }
  moves.forEach((move, index) => {
    const item = document.createElement("li");
    if (index === activeIndex) item.classList.add("active");
    item.textContent = `${index + 1}. ${move}`;
    listElement.append(item);
  });
}

function formatSessionTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown time";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderSavedSessions() {
  const sessions = Array.isArray(state.savedSessions) ? state.savedSessions : [];
  const label = state.backendOnline ? `${sessions.length} saved` : "API offline";
  setBadge(elements.savedSessionsBadge, label, state.backendOnline ? "info" : "");
  elements.savedSessionsList.innerHTML = "";

  if (!sessions.length) {
    const item = document.createElement("li");
    item.className = "session-empty";
    item.textContent = state.backendOnline ? "No saved sessions." : "Connect the API to load history.";
    elements.savedSessionsList.append(item);
    return;
  }

  const fragment = document.createDocumentFragment();
  sessions.forEach((session) => {
    const item = document.createElement("li");
    item.className = "session-item";

    const summary = document.createElement("div");
    summary.className = "session-summary";

    const title = document.createElement("strong");
    title.textContent = session.session_id || "unknown session";

    const meta = document.createElement("span");
    meta.textContent = `${session.solver || "solver"} | ${session.status || "status"} | ${
      session.move_count || 0
    } moves | ${formatSessionTime(session.created_at)}`;

    summary.append(title, meta);

    const actions = document.createElement("div");
    actions.className = "session-actions";

    const loadButton = document.createElement("button");
    loadButton.type = "button";
    loadButton.textContent = "Load";
    loadButton.addEventListener("click", () => loadSavedSession(session.session_id));

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.textContent = "Delete";
    deleteButton.className = "danger-action";
    deleteButton.addEventListener("click", () => deleteSavedSession(session.session_id));

    actions.append(loadButton, deleteButton);
    item.append(summary, actions);
    fragment.append(item);
  });

  elements.savedSessionsList.append(fragment);
}

function setBadge(element, text, variant) {
  element.textContent = text;
  element.className = `badge ${variant || ""}`.trim();
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : "n/a";
}

function formatPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "n/a";
}

function formatScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "n/a";
}

function appendMetric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";

  const labelElement = document.createElement("span");
  labelElement.className = "metric-label";
  labelElement.textContent = label;

  const valueElement = document.createElement("strong");
  valueElement.className = "metric-value";
  valueElement.textContent = value;

  item.append(labelElement, valueElement);
  elements.rlDecisionStats.append(item);
}

function traceOutcomeLabel(outcome) {
  return String(outcome || "queued").replaceAll("_", " ");
}

function isActiveTraceRecord(record) {
  const activeIndex = activeDecisionIndex();
  if (activeIndex < 0 || !Array.isArray(record.path)) return false;
  const expectedPath = state.solution.slice(0, activeIndex);
  return (
    Number(record.depth) === activeIndex &&
    record.path.length === expectedPath.length &&
    record.path.every((move, index) => move === expectedPath[index])
  );
}

function renderSearchTrace() {
  const trace = searchTracePayload();
  elements.rlSearchTraceBlock.hidden = !trace;
  if (!trace) return;

  const records = Array.isArray(trace.records) ? trace.records : [];
  const cappedText = trace.truncated ? " capped" : "";
  elements.rlSearchTraceBadge.textContent = `${records.length} records${cappedText}`;
  elements.rlSearchTraceList.innerHTML = "";

  if (!records.length) {
    const item = document.createElement("li");
    item.className = "search-trace-record";
    item.textContent = "No branch records captured.";
    elements.rlSearchTraceList.append(item);
    return;
  }

  const fragment = document.createDocumentFragment();
  records.forEach((record) => {
    const item = document.createElement("li");
    item.className = "search-trace-record";
    if (isActiveTraceRecord(record)) item.classList.add("active");

    const header = document.createElement("div");
    header.className = "search-trace-header";

    const path = Array.isArray(record.path) && record.path.length ? record.path.join(" ") : "root";
    const title = document.createElement("strong");
    title.className = "search-trace-title";
    title.textContent = `#${record.node_id || "?"} d${record.depth || 0} ${path}`;

    const score = document.createElement("span");
    score.className = "confidence-value";
    score.textContent = `score ${formatScore(record.score)}`;

    header.append(title, score);
    item.append(header);

    const stateFingerprint = document.createElement("code");
    stateFingerprint.className = "trace-state-fingerprint";
    stateFingerprint.textContent = String(record.stickers || "").slice(0, 18);
    item.append(stateFingerprint);

    const candidateList = document.createElement("div");
    candidateList.className = "trace-candidate-list";
    const candidates = Array.isArray(record.candidates) ? record.candidates : [];
    candidates.forEach((candidate) => {
      const row = document.createElement("div");
      row.className = "trace-candidate-row";

      const move = document.createElement("span");
      move.className = "candidate-move";
      move.textContent = candidate.move || "?";

      const probability = document.createElement("span");
      probability.className = "candidate-probability";
      probability.textContent = formatPercent(candidate.probability);

      const status = document.createElement("span");
      const outcomeClass = String(candidate.outcome || "queued").replaceAll("_", "-");
      status.className = `candidate-status status-${outcomeClass}`;
      status.textContent = traceOutcomeLabel(candidate.outcome);

      row.append(move, probability, status);
      candidateList.append(row);
    });
    item.append(candidateList);
    fragment.append(item);
  });

  elements.rlSearchTraceList.append(fragment);
}

function renderRlDecisionPanel() {
  const details = state.rlDetails;
  const steps = rlDecisionSteps();
  const shouldShow = state.solverMode === "rl" && Boolean(details);
  elements.rlDecisionPanel.hidden = !shouldShow;
  if (!shouldShow) {
    elements.rlSearchTraceBlock.hidden = true;
    return;
  }

  const modelVersion = details.model_version || details.model_checkpoint || "unknown model";
  const strategy = String(details.strategy || "policy-guided search").replaceAll("-", " ");
  elements.rlStrategyBadge.textContent = modelVersion;
  elements.rlCurrentState.textContent = state.stickers.join("");

  elements.rlDecisionStats.innerHTML = "";
  appendMetric("Strategy", strategy);
  appendMetric("Depth", `${formatNumber(details.depth_reached)} / ${formatNumber(details.max_depth)}`);
  appendMetric("Expanded", formatNumber(details.expanded_states));
  appendMetric("Visited", formatNumber(details.visited_states));
  appendMetric("Beam", formatNumber(details.beam_width));
  appendMetric("Top-K", formatNumber(details.top_k));

  elements.rlDecisionList.innerHTML = "";
  if (!steps.length) {
    const item = document.createElement("li");
    item.className = "decision-step";
    item.textContent = "No decision steps recorded.";
    elements.rlDecisionList.append(item);
    renderSearchTrace();
    return;
  }

  const activeIndex = activeDecisionIndex();
  steps.forEach((step, index) => {
    const item = document.createElement("li");
    item.className = "decision-step";
    if (index === activeIndex) item.classList.add("active");

    const header = document.createElement("div");
    header.className = "decision-step-header";

    const move = document.createElement("strong");
    move.className = "move-pill";
    move.textContent = `${step.step || index + 1}. ${step.selected_move || "?"}`;

    const confidence = document.createElement("span");
    confidence.className = "confidence-value";
    confidence.textContent = formatPercent(step.confidence);

    header.append(move, confidence);
    item.append(header);

    const candidates = document.createElement("div");
    candidates.className = "candidate-list";
    const topCandidates = Array.isArray(step.top_candidates) ? step.top_candidates : [];
    topCandidates.forEach((candidate) => {
      const row = document.createElement("div");
      row.className = "candidate-row";

      const candidateMove = document.createElement("span");
      candidateMove.className = "candidate-move";
      candidateMove.textContent = candidate.move || "?";

      const bar = document.createElement("span");
      bar.className = "candidate-bar";
      const fill = document.createElement("span");
      const probability = Math.max(0, Math.min(1, Number(candidate.probability) || 0));
      fill.style.width = `${Math.round(probability * 100)}%`;
      bar.append(fill);

      const probabilityText = document.createElement("span");
      probabilityText.className = "candidate-probability";
      probabilityText.textContent = formatPercent(candidate.probability);

      row.append(candidateMove, bar, probabilityText);
      candidates.append(row);
    });
    item.append(candidates);
    elements.rlDecisionList.append(item);
  });
  renderSearchTrace();
}

function stopPlayback() {
  if (state.playTimer) window.clearInterval(state.playTimer);
  state.playTimer = null;
  elements.playButton.textContent = "Play";
}

function updateSolverModeControls() {
  const isClassical = state.solverMode === "classical";
  elements.classicalModeButton.setAttribute("aria-pressed", String(isClassical));
  elements.rlModeButton.setAttribute("aria-pressed", String(!isClassical));
}

function updateUi(message) {
  const localValidation = validateCounts(state.stickers);
  const validation = state.validation || localValidation;
  setBadge(
    elements.backendBadge,
    state.backendOnline ? "API online" : "Local mode",
    state.backendOnline ? "info" : ""
  );
  setBadge(elements.validityBadge, validation.valid ? "Valid" : "Invalid", validation.valid ? "ok" : "warn");
  setBadge(elements.solvedBadge, isSolved(state.stickers) ? "Solved" : "Unsolved", isSolved(state.stickers) ? "ok" : "");
  setBadge(elements.moveCountBadge, `${state.history.length} moves`, "");

  const progress = state.solution.length
    ? Math.round((state.replayIndex / state.solution.length) * 100)
    : 0;
  elements.progressBar.style.width = `${progress}%`;
  elements.solverStatus.textContent = message || state.message;
  renderTimeline(elements.historyList, state.history);
  renderTimeline(elements.solutionList, state.solution, state.replayIndex);
  renderRlDecisionPanel();
  renderSavedSessions();
  renderCube();
  updateSolverModeControls();
  exportState(false);

  const hasSolution = state.solution.length > 0;
  elements.stepForwardButton.disabled = !hasSolution || state.replayIndex >= state.solution.length;
  elements.stepBackButton.disabled = !hasSolution || state.replayIndex <= 0;
  elements.playButton.disabled = !hasSolution || state.replayIndex >= state.solution.length;
  elements.exportReplayButton.disabled = !state.replayPackage;
  elements.saveReplayButton.disabled = !state.backendOnline || !state.replayPackage;
}

function stateExportPayload() {
  return {
    stickers: state.stickers.join(""),
    history: state.history,
    solution: state.solution,
    replay_index: state.replayIndex,
    solver_mode: state.solverMode,
    rl_details: state.rlDetails,
    replay_package: state.replayPackage,
    validation: state.validation,
  };
}

function writeJsonPayload(payload) {
  elements.stateJson.value = JSON.stringify(payload, null, 2);
}

function exportState(announce = true) {
  if (announce) state.jsonMode = "state";
  if (!announce && state.jsonMode === "replay" && state.replayPackage) {
    writeJsonPayload(state.replayPackage);
  } else {
    writeJsonPayload(stateExportPayload());
  }
  if (announce) updateUi("State exported.");
}

function exportReplayPackage() {
  if (!state.replayPackage) {
    state.jsonMode = "state";
    updateUi("No replay package available. Solve with RL first.");
    return;
  }
  state.jsonMode = "replay";
  updateUi("Replay package exported.");
}

function resetSolution(message) {
  stopPlayback();
  state.solution = [];
  state.replayIndex = 0;
  state.rlDetails = null;
  state.replayPackage = null;
  state.jsonMode = "state";
  state.message = message;
}

function stepForward() {
  if (state.replayIndex >= state.solution.length) {
    stopPlayback();
    updateUi("Replay complete.");
    return;
  }
  const move = state.solution[state.replayIndex];
  state.stickers = applyMoveToStickers(state.stickers, move);
  state.replayIndex += 1;
  updateUi(`Applied solution move ${state.replayIndex}: ${move}`);
  if (state.replayIndex >= state.solution.length) stopPlayback();
}

function stepBack() {
  if (state.replayIndex <= 0) return;
  state.replayIndex -= 1;
  const move = inverseMove(state.solution[state.replayIndex]);
  state.stickers = applyMoveToStickers(state.stickers, move);
  updateUi(`Rewound move ${state.replayIndex + 1}.`);
}

function buildMoveButtons() {
  ALL_MOVES.forEach((move) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = move;
    button.setAttribute("aria-label", `Apply move ${move}`);
    button.addEventListener("click", async () => {
      resetSolution(`Applied move ${move}.`);
      const payload = await tryBackend("/cube/apply-move", { ...cubePayload(), move });
      if (payload) {
        applyApiCube(payload);
        updateUi(`API applied move ${move}.`);
        return;
      }
      applyMoves([move], true);
      updateUi(`Applied move ${move} locally.`);
    });
    elements.moveButtons.append(button);
  });
}

function isReplayPackagePayload(payload) {
  return payload?.schema_version === "rubic-rfl-replay-v1";
}

function rlDetailsFromReplayPackage(payload) {
  const solverResult = payload.solver_result || {};
  const search = payload.search || {};
  if (solverResult.details) {
    return {
      ...solverResult.details,
      search_trace: solverResult.details.search_trace || search.trace,
    };
  }

  const model = payload.model || {};
  const steps = Array.isArray(payload.steps) ? payload.steps : [];
  return {
    strategy: search.strategy || "replay-package",
    model_version: model.version,
    model_checkpoint: model.checkpoint,
    max_depth: search.max_depth,
    beam_width: search.beam_width,
    top_k: search.top_k,
    expanded_states: search.expanded_states,
    visited_states: search.visited_states,
    depth_reached: search.depth_reached,
    search_trace: search.trace,
    steps: steps.map((step, index) => ({
      step: step.step || index + 1,
      selected_move: step.decision?.selected_move || step.move,
      confidence: step.decision?.confidence,
      top_candidates: step.decision?.top_candidates || [],
    })),
  };
}

function importReplayPackage(payload) {
  const initialState = payload.initial_state || {};
  const stickers = String(initialState.stickers || "").split("");
  if (stickers.length !== 54) throw new Error("Replay package must contain 54 initial stickers.");

  const solverName = String(payload.solver || payload.solver_result?.solver || "");
  state.stickers = stickers;
  state.history = parseMoves(initialState.history || []);
  state.solution = parseMoves(payload.moves || payload.solver_result?.moves || []);
  state.replayIndex = 0;
  state.solverMode = solverName.includes("rl") ? "rl" : "classical";
  state.rlDetails = state.solverMode === "rl" ? rlDetailsFromReplayPackage(payload) : null;
  state.replayPackage = payload;
  state.validation = initialState.validation || null;
  state.jsonMode = "replay";
  localStorage.setItem("rubic-rfl-solver-mode", state.solverMode);
  stopPlayback();
  updateUi("Replay package imported.");
}

async function loadSavedSessions(announce = true) {
  const payload = await tryBackend("/sessions?limit=25", null, { method: "GET", timeout: 3000 });
  if (!payload) {
    state.savedSessions = [];
    if (announce) updateUi(`Could not load sessions. ${state.lastApiError || ""}`.trim());
    else updateUi();
    return;
  }

  state.savedSessions = Array.isArray(payload.sessions) ? payload.sessions : [];
  updateUi(announce ? `Loaded ${state.savedSessions.length} saved sessions.` : undefined);
}

async function saveCurrentReplay() {
  if (!state.replayPackage) {
    updateUi("No replay package available to save.");
    return;
  }
  const payload = await tryBackend("/sessions", { replay_package: state.replayPackage }, { timeout: 3000 });
  if (!payload?.saved) {
    updateUi(`Replay could not be saved. ${state.lastApiError || ""}`.trim());
    return;
  }
  await loadSavedSessions(false);
  updateUi(`Saved replay ${payload.session?.session_id || state.replayPackage.session_id}.`);
}

async function loadSavedSession(sessionId) {
  if (!sessionId) return;
  const payload = await tryBackend(`/sessions/${encodeURIComponent(sessionId)}`, null, {
    method: "GET",
    timeout: 3000,
  });
  if (!payload) {
    updateUi(`Could not load session. ${state.lastApiError || ""}`.trim());
    return;
  }
  importReplayPackage(payload);
}

async function deleteSavedSession(sessionId) {
  if (!sessionId) return;
  const payload = await tryBackend(`/sessions/${encodeURIComponent(sessionId)}`, null, {
    method: "DELETE",
    timeout: 3000,
  });
  if (!payload?.deleted) {
    updateUi(`Could not delete session. ${state.lastApiError || ""}`.trim());
    return;
  }
  state.savedSessions = state.savedSessions.filter((session) => session.session_id !== sessionId);
  updateUi(`Deleted saved session ${sessionId}.`);
}

function importState() {
  const payload = JSON.parse(elements.stateJson.value);
  if (isReplayPackagePayload(payload)) {
    importReplayPackage(payload);
    return;
  }

  const stickers = Array.isArray(payload.stickers)
    ? payload.stickers
    : String(payload.stickers || "").split("");
  if (stickers.length !== 54) throw new Error("Imported stickers must contain 54 values.");
  state.stickers = stickers;
  state.history = parseMoves(payload.history || []);
  state.solution = parseMoves(payload.solution || []);
  state.replayIndex = Number(payload.replay_index || 0);
  state.rlDetails = payload.rl_details || null;
  if (payload.solver_mode === "classical" || payload.solver_mode === "rl") {
    state.solverMode = payload.solver_mode;
    localStorage.setItem("rubic-rfl-solver-mode", state.solverMode);
  }
  state.replayPackage = isReplayPackagePayload(payload.replay_package) ? payload.replay_package : null;
  state.validation = payload.validation || null;
  state.jsonMode = "state";
  stopPlayback();
  updateUi("State imported.");
}

function savedSolverMode() {
  return localStorage.getItem("rubic-rfl-solver-mode") === "rl" ? "rl" : "classical";
}

const state = {
  stickers: solvedStickers(),
  history: [],
  solution: [],
  replayIndex: 0,
  playTimer: null,
  validation: null,
  apiBase: localStorage.getItem("rubic-rfl-api-base") || DEFAULT_API_BASE,
  backendOnline: false,
  lastApiError: "",
  message: "Ready for a scramble.",
  solverMode: savedSolverMode(),
  rlDetails: null,
  replayPackage: null,
  savedSessions: [],
  jsonMode: "state",
};

const view = {
  rx: -26,
  ry: -38,
  dragging: false,
  startX: 0,
  startY: 0,
  startRx: -26,
  startRy: -38,
};

buildMoveButtons();
elements.apiBaseInput.value = state.apiBase;

elements.scrambleButton.addEventListener("click", async () => {
  const depth = Math.max(0, Math.min(100, Number(elements.depthInput.value || 0)));
  const seed = elements.seedInput.value.trim() || Date.now();
  state.stickers = solvedStickers();
  state.history = [];
  resetSolution(`Generated ${depth}-move scramble.`);
  const payload = await tryBackend("/cube/scramble", { depth, seed });
  if (payload) {
    applyApiCube(payload);
    updateUi(`API generated ${depth}-move scramble.`);
    return;
  }
  applyMoves(generateScramble(depth, seed), true);
  updateUi(`Generated ${depth}-move scramble locally.`);
});

elements.resetButton.addEventListener("click", () => {
  state.stickers = solvedStickers();
  state.history = [];
  state.validation = null;
  resetSolution("Cube reset.");
  updateUi();
});

elements.solveButton.addEventListener("click", async () => {
  stopPlayback();
  if (isSolved(state.stickers)) {
    state.solution = [];
    state.replayIndex = 0;
    state.rlDetails = null;
    state.replayPackage = null;
    updateUi("Cube is already solved.");
    return;
  }
  const isRlMode = state.solverMode === "rl";
  const originalStickers = state.stickers.slice();
  const originalHistory = state.history.slice();
  state.replayPackage = null;
  if (isRlMode) {
    state.solution = [];
    state.replayIndex = 0;
    state.rlDetails = {
      strategy: "opening RL stream",
      model_version: "rl-policy",
      steps: [],
    };
    updateUi("Opening RL stream.");

    const streamResult = await tryRlSolveStream(
      { stickers: originalStickers.join(""), history: originalHistory },
      { timeout: 8000 }
    );

    if (streamResult?.status === "solved") {
      state.rlDetails = streamResult.details || state.rlDetails;
      state.solution = parseMoves(streamResult.moves || state.solution);
      state.replayIndex = state.solution.length;
      loadSavedSessions(false);
      updateUi(`RL stream solved in ${state.solution.length} moves.`);
      return;
    }

    if (streamResult) {
      state.stickers = originalStickers;
      state.history = originalHistory;
      state.solution = [];
      state.replayIndex = 0;
      state.rlDetails = streamResult.details || state.rlDetails;
      updateUi(streamResult.message || "RL stream finished without a solution.");
      return;
    }

    state.stickers = originalStickers;
    state.history = originalHistory;
    state.solution = [];
    state.replayIndex = 0;
    state.rlDetails = null;
  }

  let result;
  if (isRlMode) {
    const replayPackage = await tryBackend(
      "/solve/rl/replay-package",
      {
        ...cubePayload(),
        session_id: `rl-rest-${Date.now()}`,
      },
      { timeout: 6000 }
    );
    if (isReplayPackagePayload(replayPackage)) {
      state.replayPackage = replayPackage;
      result = replayPackage.solver_result || null;
    }
  } else {
    result = await tryBackend("/solve/classical", cubePayload(), { timeout: 6000 });
  }
  state.rlDetails = isRlMode ? result?.details || null : null;
  if (result?.status === "solved") {
    state.solution = parseMoves(result.moves || []);
    state.replayIndex = 0;
    loadSavedSessions(false);
    updateUi(`API prepared ${state.solution.length} ${result.solver} moves.`);
    return;
  }
  if (result?.message) {
    state.solution = [];
    state.replayIndex = 0;
    updateUi(result.message);
    return;
  }
  if (isRlMode) {
    state.solution = [];
    state.replayIndex = 0;
    updateUi("Backend RL solver is required for RL replay.");
    return;
  }
  if (!state.history.length) {
    state.solution = [];
    state.replayIndex = 0;
    updateUi("Backend solver is required for imported states without move history.");
    return;
  }
  state.solution = inverseSequence(state.history);
  state.replayIndex = 0;
  updateUi(`Prepared ${state.solution.length} inverse-history moves locally.`);
});

elements.classicalModeButton.addEventListener("click", () => {
  state.solverMode = "classical";
  state.rlDetails = null;
  state.replayPackage = null;
  localStorage.setItem("rubic-rfl-solver-mode", state.solverMode);
  updateUi("Classical solver selected.");
});

elements.rlModeButton.addEventListener("click", () => {
  state.solverMode = "rl";
  localStorage.setItem("rubic-rfl-solver-mode", state.solverMode);
  updateUi("RL solver selected.");
});

elements.playButton.addEventListener("click", () => {
  if (state.playTimer) {
    stopPlayback();
    updateUi("Replay paused.");
    return;
  }
  elements.playButton.textContent = "Pause";
  state.playTimer = window.setInterval(stepForward, 520);
  stepForward();
});

elements.stepForwardButton.addEventListener("click", stepForward);
elements.stepBackButton.addEventListener("click", stepBack);

elements.applySequenceButton.addEventListener("click", async () => {
  try {
    const moves = parseMoves(elements.sequenceInput.value);
    resetSolution(`Applied ${moves.length} sequence moves.`);
    let appliedByApi = true;
    for (const move of moves) {
      const payload = await tryBackend("/cube/apply-move", { ...cubePayload(), move });
      if (!payload) {
        appliedByApi = false;
        applyMoves([move], true);
        continue;
      }
      applyApiCube(payload);
    }
    updateUi(
      appliedByApi
        ? `API applied ${moves.length} sequence moves.`
        : `Applied ${moves.length} sequence moves with local fallback.`
    );
  } catch (error) {
    updateUi(error.message);
  }
});

elements.exportButton.addEventListener("click", () => exportState(true));
elements.exportReplayButton.addEventListener("click", exportReplayPackage);
elements.refreshSessionsButton.addEventListener("click", () => loadSavedSessions(true));
elements.saveReplayButton.addEventListener("click", saveCurrentReplay);
elements.importButton.addEventListener("click", async () => {
  try {
    importState();
    const payload = await tryBackend("/cube/validate", cubePayload());
    if (payload) {
      applyApiCube(payload);
      updateUi("State imported and validated by API.");
    }
  } catch (error) {
    updateUi(error.message);
  }
});

elements.apiCheckButton.addEventListener("click", () => {
  checkBackend(true).then((connected) => {
    if (connected) loadSavedSessions(false);
  });
});

elements.apiBaseInput.addEventListener("change", () => {
  state.apiBase = elements.apiBaseInput.value.trim() || DEFAULT_API_BASE;
  localStorage.setItem("rubic-rfl-api-base", state.apiBase);
  state.backendOnline = false;
  updateUi("API URL updated.");
});

elements.viewResetButton.addEventListener("click", () => {
  view.rx = -26;
  view.ry = -38;
  renderCube();
});

elements.cubeStage.addEventListener("pointerdown", (event) => {
  view.dragging = true;
  view.startX = event.clientX;
  view.startY = event.clientY;
  view.startRx = view.rx;
  view.startRy = view.ry;
  elements.cubeStage.setPointerCapture(event.pointerId);
});

elements.cubeStage.addEventListener("pointermove", (event) => {
  if (!view.dragging) return;
  const dx = event.clientX - view.startX;
  const dy = event.clientY - view.startY;
  view.ry = view.startRy + dx * 0.45;
  view.rx = Math.max(-80, Math.min(80, view.startRx - dy * 0.45));
  renderCube();
});

elements.cubeStage.addEventListener("pointerup", (event) => {
  view.dragging = false;
  elements.cubeStage.releasePointerCapture(event.pointerId);
});

elements.cubeStage.addEventListener("pointercancel", () => {
  view.dragging = false;
});

updateUi();
checkBackend(false).then((connected) => {
  if (connected) loadSavedSessions(false);
});
