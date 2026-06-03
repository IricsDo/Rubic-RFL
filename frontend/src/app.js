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
const MIN_SCRAMBLE_DEPTH = 0;
const MAX_SCRAMBLE_DEPTH = 30;
const DEFAULT_LANGUAGE = "en";
const SUPPORTED_LANGUAGES = new Set(["en", "vn"]);
const LANGUAGE_NAMES = {
  en: "English",
  vn: "Tiếng Việt",
};
const LOCALES = {
  en: undefined,
  vn: "vi-VN",
};
const TRANSLATIONS = {
  en: {
    "document.title": "Rubic RFL Solver MVP",
    "skip.workspace": "Skip to workspace",
    "app.title": "Real-time Rubik's Cube Solver",
    "language.label": "Language",
    "language.switchEnglish": "Switch UI language to English",
    "language.switchVietnamese": "Switch UI language to Vietnamese",
    "section.cubeViewport": "Interactive cube viewport",
    "section.cubeControls": "Cube controls",
    "section.stateTimeline": "State and timeline",
    "section.moves": "Moves",
    "section.backend": "Backend",
    "section.scramble": "Scramble",
    "section.replay": "Replay",
    "section.savedSessions": "Saved Sessions",
    "section.moveTimeline": "Move Timeline",
    "section.solution": "Solution",
    "section.rlDecisionTrace": "RL Decision Trace",
    "section.searchTrace": "Search Trace",
    "section.moveSequence": "Move Sequence",
    "section.stateJson": "State JSON",
    "hint.viewport": "Drag the cube to rotate the camera.",
    "label.apiBase": "API base URL",
    "label.depth": "Depth",
    "label.seed": "Seed",
    "control.depthDecrease": "Decrease depth",
    "control.depthIncrease": "Increase depth",
    "label.solverMode": "Solver mode",
    "label.replayProgress": "Replay progress",
    "label.currentCubeState": "Current cube state",
    "label.applyMoves": "Apply moves to current state",
    "label.stateJson": "State or replay package JSON",
    "button.resetView": "Reset View",
    "button.checkApi": "Check API",
    "button.scramble": "Scramble",
    "button.reset": "Reset",
    "button.classical": "Classical",
    "button.solve": "Solve",
    "button.play": "Play",
    "button.pause": "Pause",
    "button.stepBack": "Step Back",
    "button.stepForward": "Step Forward",
    "button.refresh": "Refresh",
    "button.saveReplay": "Save Replay",
    "button.applySequence": "Apply Sequence",
    "button.export": "Export",
    "button.exportReplay": "Export Replay",
    "button.import": "Import",
    "button.load": "Load",
    "button.delete": "Delete",
    "badge.apiUnchecked": "API unchecked",
    "badge.apiOnline": "API online",
    "badge.localMode": "Local mode",
    "badge.valid": "Valid",
    "badge.invalid": "Invalid",
    "badge.solved": "Solved",
    "badge.unsolved": "Unsolved",
    "badge.moveCount": "{count} moves",
    "saved.notLoaded": "Not loaded",
    "saved.count": "{count} saved",
    "saved.apiOffline": "API offline",
    "saved.none": "No saved sessions.",
    "saved.connectApi": "Connect the API to load history.",
    "session.unknown": "unknown session",
    "session.solver": "solver",
    "session.status": "status",
    "timeline.none": "None",
    "time.unknown": "unknown time",
    "rl.policySearch": "Policy search",
    "rl.unknownModel": "unknown model",
    "rl.noDecisionSteps": "No decision steps recorded.",
    "metric.strategy": "Strategy",
    "metric.depth": "Depth",
    "metric.expanded": "Expanded",
    "metric.visited": "Visited",
    "metric.beam": "Beam",
    "metric.topK": "Top-K",
    "trace.records": "{count} records{capped}",
    "trace.zeroRecords": "0 records",
    "trace.capped": " capped",
    "trace.noBranchRecords": "No branch records captured.",
    "trace.root": "root",
    "trace.score": "score {score}",
    "traceOutcome.queued": "queued",
    "traceOutcome.kept": "kept",
    "traceOutcome.pruned": "pruned",
    "traceOutcome.solution": "solution",
    "traceOutcome.kept_in_beam": "kept in beam",
    "traceOutcome.pruned_by_beam": "pruned by beam",
    "traceOutcome.skipped_inverse": "skipped inverse",
    "traceOutcome.skipped_visited": "skipped visited",
    "move.applyAria": "Apply move {move}",
    "message.ready": "Ready for a scramble.",
    "message.generatingScramble": "Generating {depth}-move scramble...",
    "message.preparingSolution": "Preparing solution...",
    "message.playingPreparedSolution": "Solution ready. Playing {count} moves.",
    "message.languageUpdated": "UI language changed to {language}.",
    "message.rlStreamStarted": "RL stream started.",
    "message.rlSelected": "RL selected {move}.",
    "message.rlApplied": "RL applied move {step}: {move}.",
    "message.backendConnected": "Backend API connected.",
    "message.backendOffline": "Backend offline. Using local fallback. {error}",
    "message.stateExported": "State exported.",
    "message.noReplayForRl": "No replay package available. Solve with RL first.",
    "message.replayExported": "Replay package exported.",
    "message.replayComplete": "Replay complete.",
    "message.appliedSolutionMove": "Applied solution move {index}: {move}",
    "message.rewoundMove": "Rewound move {index}.",
    "message.appliedMove": "Applied move {move}.",
    "message.apiAppliedMove": "API applied move {move}.",
    "message.appliedMoveLocal": "Applied move {move} locally.",
    "message.replayImported": "Replay package imported.",
    "message.loadSessionsFailed": "Could not load sessions. {error}",
    "message.loadedSessions": "Loaded {count} saved sessions.",
    "message.noReplayToSave": "No replay package available to save.",
    "message.replaySaveFailed": "Replay could not be saved. {error}",
    "message.savedReplay": "Saved replay {sessionId}.",
    "message.loadSessionFailed": "Could not load session. {error}",
    "message.deleteSessionFailed": "Could not delete session. {error}",
    "message.deletedSession": "Deleted saved session {sessionId}.",
    "message.stateImported": "State imported.",
    "message.generatedScramble": "Generated {depth}-move scramble.",
    "message.apiGeneratedScramble": "API generated {depth}-move scramble.",
    "message.generatedScrambleLocal": "Generated {depth}-move scramble locally.",
    "message.cubeReset": "Cube reset.",
    "message.alreadySolved": "Cube is already solved.",
    "message.openingRlStream": "Opening RL stream.",
    "message.rlStreamSolved": "RL stream solved in {count} moves.",
    "message.rlStreamNoSolution": "RL stream finished without a solution.",
    "message.apiPreparedMoves": "API prepared {count} {solver} moves.",
    "message.backendRlRequired": "Backend RL solver is required for RL replay.",
    "message.backendSolverRequired": "Backend solver is required for imported states without move history.",
    "message.preparedInverse": "Prepared {count} inverse-history moves locally.",
    "message.classicalSelected": "Classical solver selected.",
    "message.rlSolverSelected": "RL solver selected.",
    "message.replayPaused": "Replay paused.",
    "message.appliedSequence": "Applied {count} sequence moves.",
    "message.apiAppliedSequence": "API applied {count} sequence moves.",
    "message.localFallbackSequence": "Applied {count} sequence moves with local fallback.",
    "message.stateImportedValidated": "State imported and validated by API.",
    "message.apiUrlUpdated": "API URL updated.",
    "error.replayStickers": "Replay package must contain 54 initial stickers.",
    "error.importedStickers": "Imported stickers must contain 54 values.",
  },
  vn: {
    "document.title": "MVP trình giải Rubic RFL",
    "skip.workspace": "Đến khu làm việc",
    "app.title": "Trình giải Rubik thời gian thực",
    "language.label": "Ngôn ngữ",
    "language.switchEnglish": "Chuyển giao diện sang tiếng Anh",
    "language.switchVietnamese": "Chuyển giao diện sang tiếng Việt",
    "section.cubeViewport": "Khung xem khối tương tác",
    "section.cubeControls": "Điều khiển khối",
    "section.stateTimeline": "Trạng thái và dòng thời gian",
    "section.moves": "Nước đi",
    "section.backend": "Backend",
    "section.scramble": "Xáo trộn",
    "section.replay": "Phát lại",
    "section.savedSessions": "Phiên đã lưu",
    "section.moveTimeline": "Dòng thời gian nước đi",
    "section.solution": "Lời giải",
    "section.rlDecisionTrace": "Nhật ký quyết định RL",
    "section.searchTrace": "Nhật ký tìm kiếm",
    "section.moveSequence": "Chuỗi nước đi",
    "section.stateJson": "JSON trạng thái",
    "hint.viewport": "Kéo khối để xoay góc nhìn.",
    "label.apiBase": "URL gốc API",
    "label.depth": "Độ sâu",
    "label.seed": "Seed",
    "control.depthDecrease": "Giảm độ sâu",
    "control.depthIncrease": "Tăng độ sâu",
    "label.solverMode": "Chế độ giải",
    "label.replayProgress": "Tiến trình phát lại",
    "label.currentCubeState": "Trạng thái khối hiện tại",
    "label.applyMoves": "Áp dụng nước đi vào trạng thái hiện tại",
    "label.stateJson": "JSON trạng thái hoặc gói phát lại",
    "button.resetView": "Đặt lại góc nhìn",
    "button.checkApi": "Kiểm tra API",
    "button.scramble": "Xáo trộn",
    "button.reset": "Đặt lại",
    "button.classical": "Cổ điển",
    "button.solve": "Giải",
    "button.play": "Phát",
    "button.pause": "Tạm dừng",
    "button.stepBack": "Lùi bước",
    "button.stepForward": "Tiến bước",
    "button.refresh": "Tải lại",
    "button.saveReplay": "Lưu phát lại",
    "button.applySequence": "Áp dụng chuỗi",
    "button.export": "Xuất",
    "button.exportReplay": "Xuất phát lại",
    "button.import": "Nhập",
    "button.load": "Tải",
    "button.delete": "Xóa",
    "badge.apiUnchecked": "Chưa kiểm tra API",
    "badge.apiOnline": "API trực tuyến",
    "badge.localMode": "Chế độ cục bộ",
    "badge.valid": "Hợp lệ",
    "badge.invalid": "Không hợp lệ",
    "badge.solved": "Đã giải",
    "badge.unsolved": "Chưa giải",
    "badge.moveCount": "{count} nước đi",
    "saved.notLoaded": "Chưa tải",
    "saved.count": "{count} đã lưu",
    "saved.apiOffline": "API ngoại tuyến",
    "saved.none": "Chưa có phiên đã lưu.",
    "saved.connectApi": "Kết nối API để tải lịch sử.",
    "session.unknown": "phiên không rõ",
    "session.solver": "bộ giải",
    "session.status": "trạng thái",
    "timeline.none": "Không có",
    "time.unknown": "thời gian không rõ",
    "rl.policySearch": "Tìm kiếm chính sách",
    "rl.unknownModel": "mô hình không rõ",
    "rl.noDecisionSteps": "Chưa ghi nhận bước quyết định.",
    "metric.strategy": "Chiến lược",
    "metric.depth": "Độ sâu",
    "metric.expanded": "Đã mở rộng",
    "metric.visited": "Đã thăm",
    "metric.beam": "Beam",
    "metric.topK": "Top-K",
    "trace.records": "{count} bản ghi{capped}",
    "trace.zeroRecords": "0 bản ghi",
    "trace.capped": " giới hạn",
    "trace.noBranchRecords": "Chưa ghi nhận nhánh tìm kiếm.",
    "trace.root": "gốc",
    "trace.score": "điểm {score}",
    "traceOutcome.queued": "đang chờ",
    "traceOutcome.kept": "giữ lại",
    "traceOutcome.pruned": "loại bỏ",
    "traceOutcome.solution": "lời giải",
    "traceOutcome.kept_in_beam": "giữ trong beam",
    "traceOutcome.pruned_by_beam": "loại bởi beam",
    "traceOutcome.skipped_inverse": "bỏ qua nghịch đảo",
    "traceOutcome.skipped_visited": "bỏ qua đã thăm",
    "move.applyAria": "Áp dụng nước đi {move}",
    "message.ready": "Sẵn sàng xáo trộn.",
    "message.generatingScramble": "Đang tạo xáo trộn {depth} nước...",
    "message.preparingSolution": "Đang chuẩn bị lời giải...",
    "message.playingPreparedSolution": "Đã chuẩn bị {count} nước. Đang phát lời giải.",
    "message.languageUpdated": "Đã đổi ngôn ngữ giao diện sang {language}.",
    "message.rlStreamStarted": "Luồng RL đã bắt đầu.",
    "message.rlSelected": "RL đã chọn {move}.",
    "message.rlApplied": "RL đã áp dụng nước {step}: {move}.",
    "message.backendConnected": "Đã kết nối Backend API.",
    "message.backendOffline": "Backend ngoại tuyến. Dùng chế độ cục bộ. {error}",
    "message.stateExported": "Đã xuất trạng thái.",
    "message.noReplayForRl": "Chưa có gói phát lại. Hãy giải bằng RL trước.",
    "message.replayExported": "Đã xuất gói phát lại.",
    "message.replayComplete": "Phát lại hoàn tất.",
    "message.appliedSolutionMove": "Đã áp dụng nước giải {index}: {move}",
    "message.rewoundMove": "Đã lùi về nước {index}.",
    "message.appliedMove": "Đã áp dụng nước {move}.",
    "message.apiAppliedMove": "API đã áp dụng nước {move}.",
    "message.appliedMoveLocal": "Đã áp dụng nước {move} cục bộ.",
    "message.replayImported": "Đã nhập gói phát lại.",
    "message.loadSessionsFailed": "Không thể tải phiên. {error}",
    "message.loadedSessions": "Đã tải {count} phiên đã lưu.",
    "message.noReplayToSave": "Chưa có gói phát lại để lưu.",
    "message.replaySaveFailed": "Không thể lưu phát lại. {error}",
    "message.savedReplay": "Đã lưu phát lại {sessionId}.",
    "message.loadSessionFailed": "Không thể tải phiên. {error}",
    "message.deleteSessionFailed": "Không thể xóa phiên. {error}",
    "message.deletedSession": "Đã xóa phiên đã lưu {sessionId}.",
    "message.stateImported": "Đã nhập trạng thái.",
    "message.generatedScramble": "Đã tạo xáo trộn {depth} nước.",
    "message.apiGeneratedScramble": "API đã tạo xáo trộn {depth} nước.",
    "message.generatedScrambleLocal": "Đã tạo xáo trộn {depth} nước cục bộ.",
    "message.cubeReset": "Đã đặt lại khối.",
    "message.alreadySolved": "Khối đã được giải.",
    "message.openingRlStream": "Đang mở luồng RL.",
    "message.rlStreamSolved": "Luồng RL đã giải trong {count} nước.",
    "message.rlStreamNoSolution": "Luồng RL kết thúc nhưng chưa có lời giải.",
    "message.apiPreparedMoves": "API đã chuẩn bị {count} nước bằng {solver}.",
    "message.backendRlRequired": "Cần bộ giải RL backend để phát lại RL.",
    "message.backendSolverRequired": "Cần bộ giải backend cho trạng thái nhập không có lịch sử nước đi.",
    "message.preparedInverse": "Đã chuẩn bị {count} nước đảo lịch sử cục bộ.",
    "message.classicalSelected": "Đã chọn bộ giải cổ điển.",
    "message.rlSolverSelected": "Đã chọn bộ giải RL.",
    "message.replayPaused": "Đã tạm dừng phát lại.",
    "message.appliedSequence": "Đã áp dụng {count} nước trong chuỗi.",
    "message.apiAppliedSequence": "API đã áp dụng {count} nước trong chuỗi.",
    "message.localFallbackSequence": "Đã áp dụng {count} nước trong chuỗi với fallback cục bộ.",
    "message.stateImportedValidated": "Đã nhập trạng thái và xác thực bằng API.",
    "message.apiUrlUpdated": "Đã cập nhật URL API.",
    "error.replayStickers": "Gói phát lại phải chứa 54 sticker ban đầu.",
    "error.importedStickers": "Sticker nhập vào phải có 54 giá trị.",
  },
};

function savedLanguage() {
  const language = localStorage.getItem("rubic-rfl-language");
  return SUPPORTED_LANGUAGES.has(language) ? language : DEFAULT_LANGUAGE;
}

let activeLanguage = savedLanguage();

function t(key, params = {}, fallback = undefined) {
  const template = TRANSLATIONS[activeLanguage]?.[key] ?? TRANSLATIONS.en[key] ?? fallback ?? key;
  return template.replace(/\{([A-Za-z0-9_]+)\}/g, (_, name) => String(params[name] ?? ""));
}

const elements = {
  cube: document.querySelector("#cube"),
  cubeStage: document.querySelector("#cubeStage"),
  moveButtons: document.querySelector("#moveButtons"),
  languageEnButton: document.querySelector("#languageEnButton"),
  languageVnButton: document.querySelector("#languageVnButton"),
  depthInput: document.querySelector("#depthInput"),
  depthDecreaseButton: document.querySelector("#depthDecreaseButton"),
  depthIncreaseButton: document.querySelector("#depthIncreaseButton"),
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

function translateMoveButtonLabels() {
  elements.moveButtons.querySelectorAll("button[data-move]").forEach((button) => {
    button.setAttribute("aria-label", t("move.applyAria", { move: button.dataset.move }));
  });
}

function translateStaticUi() {
  document.documentElement.lang = activeLanguage === "vn" ? "vi" : "en";
  document.title = t("document.title");
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    element.textContent = t(element.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-aria-label]").forEach((element) => {
    element.setAttribute("aria-label", t(element.dataset.i18nAriaLabel));
  });
  translateMoveButtonLabels();
}

function updateLanguageControls() {
  elements.languageEnButton.setAttribute("aria-pressed", String(activeLanguage === "en"));
  elements.languageVnButton.setAttribute("aria-pressed", String(activeLanguage === "vn"));
}

function setLanguage(language, options = {}) {
  activeLanguage = SUPPORTED_LANGUAGES.has(language) ? language : DEFAULT_LANGUAGE;
  localStorage.setItem("rubic-rfl-language", activeLanguage);
  translateStaticUi();
  updateLanguageControls();
  if (options.render) {
    updateUi(options.announce ? t("message.languageUpdated", { language: LANGUAGE_NAMES[activeLanguage] }) : undefined);
  }
}

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
        updateUi(t("message.rlStreamStarted"));
        return;
      }

      if (message.event === "decision") {
        applyRlDecisionEvent(message);
        updateUi(t("message.rlSelected", { move: message.selected_move || "a move" }));
        return;
      }

      if (message.event === "move") {
        applyRlMoveEvent(message);
        updateUi(t("message.rlApplied", { step: message.step, move: message.move }));
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
    if (announce) updateUi(t("message.backendConnected"));
    return true;
  }
  state.backendOnline = false;
  if (announce) updateUi(t("message.backendOffline", { error: state.lastApiError || "" }).trim());
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
    item.textContent = t("timeline.none");
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
  if (Number.isNaN(date.getTime())) return t("time.unknown");
  return date.toLocaleString(LOCALES[activeLanguage], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function renderSavedSessions() {
  const sessions = Array.isArray(state.savedSessions) ? state.savedSessions : [];
  const label = state.backendOnline
    ? t("saved.count", { count: sessions.length })
    : t("saved.apiOffline");
  setBadge(elements.savedSessionsBadge, label, state.backendOnline ? "info" : "");
  elements.savedSessionsList.innerHTML = "";

  if (!sessions.length) {
    const item = document.createElement("li");
    item.className = "session-empty";
    item.textContent = state.backendOnline ? t("saved.none") : t("saved.connectApi");
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
    title.textContent = session.session_id || t("session.unknown");

    const meta = document.createElement("span");
    meta.textContent = `${session.solver || t("session.solver")} | ${
      session.status || t("session.status")
    } | ${t("badge.moveCount", { count: session.move_count || 0 })} | ${formatSessionTime(session.created_at)}`;

    summary.append(title, meta);

    const actions = document.createElement("div");
    actions.className = "session-actions";

    const loadButton = document.createElement("button");
    loadButton.type = "button";
    loadButton.textContent = t("button.load");
    loadButton.addEventListener("click", () => loadSavedSession(session.session_id));

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.textContent = t("button.delete");
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
  const normalized = String(outcome || "queued").replaceAll("-", "_");
  return t(`traceOutcome.${normalized}`, {}, normalized.replaceAll("_", " "));
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
  const cappedText = trace.truncated ? t("trace.capped") : "";
  elements.rlSearchTraceBadge.textContent = t("trace.records", { count: records.length, capped: cappedText });
  elements.rlSearchTraceList.innerHTML = "";

  if (!records.length) {
    const item = document.createElement("li");
    item.className = "search-trace-record";
    item.textContent = t("trace.noBranchRecords");
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

    const path = Array.isArray(record.path) && record.path.length ? record.path.join(" ") : t("trace.root");
    const title = document.createElement("strong");
    title.className = "search-trace-title";
    title.textContent = `#${record.node_id || "?"} d${record.depth || 0} ${path}`;

    const score = document.createElement("span");
    score.className = "confidence-value";
    score.textContent = t("trace.score", { score: formatScore(record.score) });

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

  const modelVersion = details.model_version || details.model_checkpoint || t("rl.unknownModel");
  const strategy = String(details.strategy || "policy-guided search").replaceAll("-", " ");
  elements.rlStrategyBadge.textContent = modelVersion;
  elements.rlCurrentState.textContent = state.stickers.join("");

  elements.rlDecisionStats.innerHTML = "";
  appendMetric(t("metric.strategy"), strategy);
  appendMetric(t("metric.depth"), `${formatNumber(details.depth_reached)} / ${formatNumber(details.max_depth)}`);
  appendMetric(t("metric.expanded"), formatNumber(details.expanded_states));
  appendMetric(t("metric.visited"), formatNumber(details.visited_states));
  appendMetric(t("metric.beam"), formatNumber(details.beam_width));
  appendMetric(t("metric.topK"), formatNumber(details.top_k));

  elements.rlDecisionList.innerHTML = "";
  if (!steps.length) {
    const item = document.createElement("li");
    item.className = "decision-step";
    item.textContent = t("rl.noDecisionSteps");
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
  elements.playButton.textContent = t("button.play");
}

function startPlayback() {
  if (state.playTimer || !state.solution.length || state.replayIndex >= state.solution.length) return;
  elements.playButton.textContent = t("button.pause");
  state.playTimer = window.setInterval(stepForward, 520);
  stepForward();
}

function updateSolverModeControls() {
  const isClassical = state.solverMode === "classical";
  elements.classicalModeButton.setAttribute("aria-pressed", String(isClassical));
  elements.rlModeButton.setAttribute("aria-pressed", String(!isClassical));
}

function updateUi(message) {
  if (message !== undefined) state.message = message;
  const localValidation = validateCounts(state.stickers);
  const validation = state.validation || localValidation;
  const isBusy = Boolean(state.pendingAction);
  setBadge(
    elements.backendBadge,
    state.backendOnline ? t("badge.apiOnline") : t("badge.localMode"),
    state.backendOnline ? "info" : ""
  );
  setBadge(
    elements.validityBadge,
    validation.valid ? t("badge.valid") : t("badge.invalid"),
    validation.valid ? "ok" : "warn"
  );
  setBadge(
    elements.solvedBadge,
    isSolved(state.stickers) ? t("badge.solved") : t("badge.unsolved"),
    isSolved(state.stickers) ? "ok" : ""
  );
  setBadge(elements.moveCountBadge, t("badge.moveCount", { count: state.history.length }), "");

  const progress = state.solution.length
    ? Math.round((state.replayIndex / state.solution.length) * 100)
    : 0;
  elements.progressBar.style.width = `${progress}%`;
  elements.solverStatus.textContent = state.message;
  renderTimeline(elements.historyList, state.history);
  renderTimeline(elements.solutionList, state.solution, state.replayIndex);
  renderRlDecisionPanel();
  renderSavedSessions();
  renderCube();
  updateSolverModeControls();
  exportState(false);

  const hasSolution = state.solution.length > 0;
  elements.moveButtons.querySelectorAll("button").forEach((button) => {
    button.disabled = isBusy;
  });
  elements.scrambleButton.disabled = isBusy;
  elements.resetButton.disabled = isBusy;
  elements.solveButton.disabled = isBusy;
  elements.classicalModeButton.disabled = isBusy;
  elements.rlModeButton.disabled = isBusy;
  elements.applySequenceButton.disabled = isBusy;
  elements.importButton.disabled = isBusy;
  elements.stepForwardButton.disabled = isBusy || !hasSolution || state.replayIndex >= state.solution.length;
  elements.stepBackButton.disabled = isBusy || !hasSolution || state.replayIndex <= 0;
  elements.playButton.disabled = isBusy || !hasSolution || state.replayIndex >= state.solution.length;
  elements.exportReplayButton.disabled = !state.replayPackage;
  elements.saveReplayButton.disabled = !state.backendOnline || !state.replayPackage;
  elements.playButton.textContent = state.playTimer ? t("button.pause") : t("button.play");
  const scrambleDepth = readScrambleDepth();
  elements.depthDecreaseButton.disabled = isBusy || scrambleDepth <= MIN_SCRAMBLE_DEPTH;
  elements.depthIncreaseButton.disabled = isBusy || scrambleDepth >= MAX_SCRAMBLE_DEPTH;
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
  if (announce) updateUi(t("message.stateExported"));
}

function exportReplayPackage() {
  if (!state.replayPackage) {
    state.jsonMode = "state";
    updateUi(t("message.noReplayForRl"));
    return;
  }
  state.jsonMode = "replay";
  updateUi(t("message.replayExported"));
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
    updateUi(t("message.replayComplete"));
    return;
  }
  const move = state.solution[state.replayIndex];
  state.stickers = applyMoveToStickers(state.stickers, move);
  state.replayIndex += 1;
  updateUi(t("message.appliedSolutionMove", { index: state.replayIndex, move }));
  if (state.replayIndex >= state.solution.length) stopPlayback();
}

function stepBack() {
  if (state.replayIndex <= 0) return;
  state.replayIndex -= 1;
  const move = inverseMove(state.solution[state.replayIndex]);
  state.stickers = applyMoveToStickers(state.stickers, move);
  updateUi(t("message.rewoundMove", { index: state.replayIndex + 1 }));
}

function buildMoveButtons() {
  ALL_MOVES.forEach((move) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = move;
    button.dataset.move = move;
    button.setAttribute("aria-label", t("move.applyAria", { move }));
    button.addEventListener("click", async () => {
      resetSolution(t("message.appliedMove", { move }));
      const payload = await tryBackend("/cube/apply-move", { ...cubePayload(), move });
      if (payload) {
        applyApiCube(payload);
        updateUi(t("message.apiAppliedMove", { move }));
        return;
      }
      applyMoves([move], true);
      updateUi(t("message.appliedMoveLocal", { move }));
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
  if (stickers.length !== 54) throw new Error(t("error.replayStickers"));

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
  updateUi(t("message.replayImported"));
}

async function loadSavedSessions(announce = true) {
  const payload = await tryBackend("/sessions?limit=25", null, { method: "GET", timeout: 3000 });
  if (!payload) {
    state.savedSessions = [];
    if (announce) updateUi(t("message.loadSessionsFailed", { error: state.lastApiError || "" }).trim());
    else updateUi();
    return;
  }

  state.savedSessions = Array.isArray(payload.sessions) ? payload.sessions : [];
  updateUi(announce ? t("message.loadedSessions", { count: state.savedSessions.length }) : undefined);
}

async function saveCurrentReplay() {
  if (!state.replayPackage) {
    updateUi(t("message.noReplayToSave"));
    return;
  }
  const payload = await tryBackend("/sessions", { replay_package: state.replayPackage }, { timeout: 3000 });
  if (!payload?.saved) {
    updateUi(t("message.replaySaveFailed", { error: state.lastApiError || "" }).trim());
    return;
  }
  await loadSavedSessions(false);
  updateUi(t("message.savedReplay", { sessionId: payload.session?.session_id || state.replayPackage.session_id }));
}

async function loadSavedSession(sessionId) {
  if (!sessionId) return;
  const payload = await tryBackend(`/sessions/${encodeURIComponent(sessionId)}`, null, {
    method: "GET",
    timeout: 3000,
  });
  if (!payload) {
    updateUi(t("message.loadSessionFailed", { error: state.lastApiError || "" }).trim());
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
    updateUi(t("message.deleteSessionFailed", { error: state.lastApiError || "" }).trim());
    return;
  }
  state.savedSessions = state.savedSessions.filter((session) => session.session_id !== sessionId);
  updateUi(t("message.deletedSession", { sessionId }));
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
  if (stickers.length !== 54) throw new Error(t("error.importedStickers"));
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
  updateUi(t("message.stateImported"));
}

function savedSolverMode() {
  return localStorage.getItem("rubic-rfl-solver-mode") === "rl" ? "rl" : "classical";
}

function clampScrambleDepth(value, fallback = MIN_SCRAMBLE_DEPTH) {
  const numericDepth = value === "" ? fallback : Number(value);
  const integerDepth = Number.isFinite(numericDepth) ? Math.trunc(numericDepth) : fallback;
  return Math.max(MIN_SCRAMBLE_DEPTH, Math.min(MAX_SCRAMBLE_DEPTH, integerDepth));
}

function writeScrambleDepth(value) {
  const depth = clampScrambleDepth(value);
  elements.depthInput.value = String(depth);
  return depth;
}

function readScrambleDepth() {
  return writeScrambleDepth(elements.depthInput.value);
}

function adjustScrambleDepth(delta) {
  writeScrambleDepth(readScrambleDepth() + delta);
  updateUi();
}

function preventScrambleDepthEditing(event) {
  event.preventDefault();
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
  message: t("message.ready"),
  language: activeLanguage,
  solverMode: savedSolverMode(),
  rlDetails: null,
  replayPackage: null,
  savedSessions: [],
  jsonMode: "state",
  pendingAction: null,
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
setLanguage(activeLanguage);
elements.apiBaseInput.value = state.apiBase;
elements.depthInput.min = String(MIN_SCRAMBLE_DEPTH);
elements.depthInput.max = String(MAX_SCRAMBLE_DEPTH);
elements.depthInput.step = "1";
elements.depthInput.readOnly = true;
elements.depthInput.tabIndex = -1;
elements.depthInput.setAttribute("aria-readonly", "true");

function handleLanguageChange(language) {
  state.language = language;
  setLanguage(language, { announce: true, render: true });
}

elements.languageEnButton.addEventListener("click", () => handleLanguageChange("en"));
elements.languageVnButton.addEventListener("click", () => handleLanguageChange("vn"));
elements.depthDecreaseButton.addEventListener("click", () => adjustScrambleDepth(-1));
elements.depthIncreaseButton.addEventListener("click", () => adjustScrambleDepth(1));
elements.depthInput.addEventListener("beforeinput", preventScrambleDepthEditing);
elements.depthInput.addEventListener("paste", preventScrambleDepthEditing);
elements.depthInput.addEventListener("drop", preventScrambleDepthEditing);
elements.depthInput.addEventListener("wheel", preventScrambleDepthEditing, { passive: false });

elements.scrambleButton.addEventListener("click", async () => {
  if (state.pendingAction) return;
  const depth = readScrambleDepth();
  const seed = elements.seedInput.value.trim() || Date.now();
  state.pendingAction = "scramble";
  state.stickers = solvedStickers();
  state.history = [];
  resetSolution(t("message.generatingScramble", { depth }));
  updateUi();
  const payload = await tryBackend("/cube/scramble", { depth, seed });
  let message;
  if (payload) {
    applyApiCube(payload);
    message = t("message.apiGeneratedScramble", { depth });
  } else {
    applyMoves(generateScramble(depth, seed), true);
    message = t("message.generatedScrambleLocal", { depth });
  }
  state.pendingAction = null;
  updateUi(message);
});

elements.resetButton.addEventListener("click", () => {
  if (state.pendingAction) return;
  state.stickers = solvedStickers();
  state.history = [];
  state.validation = null;
  resetSolution(t("message.cubeReset"));
  updateUi();
});

elements.solveButton.addEventListener("click", async () => {
  if (state.pendingAction) return;
  state.pendingAction = "solve";
  stopPlayback();
  updateUi(t("message.preparingSolution"));
  if (isSolved(state.stickers)) {
    state.solution = [];
    state.replayIndex = 0;
    state.rlDetails = null;
    state.replayPackage = null;
    state.pendingAction = null;
    updateUi(t("message.alreadySolved"));
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
    updateUi(t("message.openingRlStream"));

    const streamResult = await tryRlSolveStream(
      { stickers: originalStickers.join(""), history: originalHistory },
      { timeout: 8000 }
    );

    if (streamResult?.status === "solved") {
      state.rlDetails = streamResult.details || state.rlDetails;
      state.solution = parseMoves(streamResult.moves || state.solution);
      state.replayIndex = state.solution.length;
      state.pendingAction = null;
      loadSavedSessions(false);
      updateUi(t("message.rlStreamSolved", { count: state.solution.length }));
      return;
    }

    if (streamResult) {
      state.stickers = originalStickers;
      state.history = originalHistory;
      state.solution = [];
      state.replayIndex = 0;
      state.rlDetails = streamResult.details || state.rlDetails;
      state.pendingAction = null;
      updateUi(streamResult.message || t("message.rlStreamNoSolution"));
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
    state.pendingAction = null;
    loadSavedSessions(false);
    updateUi(t("message.playingPreparedSolution", { count: state.solution.length }));
    startPlayback();
    return;
  }
  if (result?.message) {
    state.solution = [];
    state.replayIndex = 0;
    state.pendingAction = null;
    updateUi(result.message);
    return;
  }
  if (isRlMode) {
    state.solution = [];
    state.replayIndex = 0;
    state.pendingAction = null;
    updateUi(t("message.backendRlRequired"));
    return;
  }
  if (!state.history.length) {
    state.solution = [];
    state.replayIndex = 0;
    state.pendingAction = null;
    updateUi(t("message.backendSolverRequired"));
    return;
  }
  state.solution = inverseSequence(state.history);
  state.replayIndex = 0;
  state.pendingAction = null;
  updateUi(t("message.playingPreparedSolution", { count: state.solution.length }));
  startPlayback();
});

elements.classicalModeButton.addEventListener("click", () => {
  state.solverMode = "classical";
  state.rlDetails = null;
  state.replayPackage = null;
  localStorage.setItem("rubic-rfl-solver-mode", state.solverMode);
  updateUi(t("message.classicalSelected"));
});

elements.rlModeButton.addEventListener("click", () => {
  state.solverMode = "rl";
  localStorage.setItem("rubic-rfl-solver-mode", state.solverMode);
  updateUi(t("message.rlSolverSelected"));
});

elements.playButton.addEventListener("click", () => {
  if (state.playTimer) {
    stopPlayback();
    updateUi(t("message.replayPaused"));
    return;
  }
  startPlayback();
});

elements.stepForwardButton.addEventListener("click", stepForward);
elements.stepBackButton.addEventListener("click", stepBack);

elements.applySequenceButton.addEventListener("click", async () => {
  try {
    const moves = parseMoves(elements.sequenceInput.value);
    resetSolution(t("message.appliedSequence", { count: moves.length }));
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
        ? t("message.apiAppliedSequence", { count: moves.length })
        : t("message.localFallbackSequence", { count: moves.length })
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
      updateUi(t("message.stateImportedValidated"));
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
  updateUi(t("message.apiUrlUpdated"));
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
