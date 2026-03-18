const ACTIONS = [
  { name: "U", arrow: "↑", dx: -1, dy: 0 },
  { name: "D", arrow: "↓", dx: 1, dy: 0 },
  { name: "L", arrow: "←", dx: 0, dy: -1 },
  { name: "R", arrow: "→", dx: 0, dy: 1 },
];

const GAMMA = 0.9;
const STEP_REWARD = -1;
const GOAL_REWARD = 20;
const EVAL_THETA = 1e-4;

const state = {
  n: 5,
  start: null,
  goal: null,
  obstacles: new Set(),
  values: [],
  policy: new Map(),
  bestPath: new Set(),
  mode: "start",
};

const gridEl = document.getElementById("grid");
const statusEl = document.getElementById("status");
const metricsEl = document.getElementById("metrics");
const policyMatrixEl = document.getElementById("policyMatrix");
const valueMatrixEl = document.getElementById("valueMatrix");
const gridSizeEl = document.getElementById("gridSize");
const buildGridBtn = document.getElementById("buildGridBtn");
const randomPolicyBtn = document.getElementById("randomPolicyBtn");
const valueIterationBtn = document.getElementById("valueIterationBtn");
const resetBtn = document.getElementById("resetBtn");

function keyOf(x, y) {
  return `${x},${y}`;
}

function isInside(x, y) {
  return x >= 0 && x < state.n && y >= 0 && y < state.n;
}

function isObstacle(x, y) {
  return state.obstacles.has(keyOf(x, y));
}

function isGoal(x, y) {
  return state.goal && state.goal.x === x && state.goal.y === y;
}

function isTerminal(x, y) {
  return isGoal(x, y);
}

function obstacleLimit() {
  return Math.max(0, state.n - 2);
}

function clampGridSize(raw) {
  const n = Number(raw);
  if (Number.isNaN(n)) {
    return 5;
  }
  return Math.min(9, Math.max(5, n));
}

function initValues() {
  state.values = Array.from({ length: state.n }, () => Array(state.n).fill(0));
}

function clearPolicy() {
  state.policy = new Map();
}

function clearBestPath() {
  state.bestPath = new Set();
}

function resetWorld() {
  state.start = null;
  state.goal = null;
  state.obstacles = new Set();
  initValues();
  clearPolicy();
  clearBestPath();
  renderGrid();
  updateStatus("請先放置起點與終點；障礙物可設 0 到 n-2 個。", true);
}

function updateStatus(message, clearMetrics = false) {
  statusEl.textContent = message;
  if (clearMetrics) {
    metricsEl.textContent = "";
  }
}

function updateMetrics() {
  const count = state.obstacles.size;
  const limit = obstacleLimit();
  metricsEl.textContent = `障礙物: ${count}/${limit}`;
}

function setMode(mode) {
  state.mode = mode;
}

function transition(x, y, action) {
  if (isTerminal(x, y)) {
    return { nx: x, ny: y, reward: 0 };
  }

  const tx = x + action.dx;
  const ty = y + action.dy;

  if (!isInside(tx, ty) || isObstacle(tx, ty)) {
    return { nx: x, ny: y, reward: STEP_REWARD };
  }

  if (isGoal(tx, ty)) {
    return { nx: tx, ny: ty, reward: GOAL_REWARD };
  }

  return { nx: tx, ny: ty, reward: STEP_REWARD };
}

function chooseRandomAction() {
  const idx = Math.floor(Math.random() * ACTIONS.length);
  return ACTIONS[idx].name;
}

function actionByName(name) {
  return ACTIONS.find((a) => a.name === name);
}

function ensureReadyForAlgorithms() {
  if (!state.start || !state.goal) {
    updateStatus("需要先設定起點與終點。", true);
    return false;
  }
  const limit = obstacleLimit();
  if (state.obstacles.size > limit) {
    updateStatus(`障礙物不可超過 n-2，目前為 ${state.obstacles.size}，上限為 ${limit}。`, true);
    return false;
  }
  return true;
}

function generateRandomPolicy() {
  clearPolicy();

  for (let x = 0; x < state.n; x += 1) {
    for (let y = 0; y < state.n; y += 1) {
      if (isObstacle(x, y) || isTerminal(x, y)) {
        continue;
      }
      state.policy.set(keyOf(x, y), chooseRandomAction());
    }
  }
}

function evaluatePolicy() {
  initValues();
  let delta = Infinity;
  let iterations = 0;

  while (delta > EVAL_THETA && iterations < 1000) {
    delta = 0;
    const nextValues = state.values.map((row) => [...row]);

    for (let x = 0; x < state.n; x += 1) {
      for (let y = 0; y < state.n; y += 1) {
        if (isObstacle(x, y)) {
          nextValues[x][y] = 0;
          continue;
        }
        if (isTerminal(x, y)) {
          nextValues[x][y] = 0;
          continue;
        }

        const actionName = state.policy.get(keyOf(x, y)) || "U";
        const action = actionByName(actionName) || ACTIONS[0];
        const { nx, ny, reward } = transition(x, y, action);
        const updated = reward + GAMMA * state.values[nx][ny];

        delta = Math.max(delta, Math.abs(updated - state.values[x][y]));
        nextValues[x][y] = updated;
      }
    }

    state.values = nextValues;
    iterations += 1;
  }

  return iterations;
}

function deriveBestPathFromPolicy() {
  clearBestPath();
  if (!state.start || !state.goal) {
    return;
  }

  let cx = state.start.x;
  let cy = state.start.y;
  const visited = new Set();
  const maxSteps = state.n * state.n;

  for (let i = 0; i < maxSteps; i += 1) {
    const k = keyOf(cx, cy);
    if (visited.has(k)) {
      break;
    }
    visited.add(k);
    state.bestPath.add(k);

    if (isGoal(cx, cy)) {
      break;
    }

    const actionName = state.policy.get(k);
    const action = actionByName(actionName);
    if (!action) {
      break;
    }
    const { nx, ny } = transition(cx, cy, action);
    cx = nx;
    cy = ny;
  }
}

function formatPolicyMatrix() {
  const rows = [];
  for (let x = 0; x < state.n; x += 1) {
    const row = [];
    for (let y = 0; y < state.n; y += 1) {
      if (isObstacle(x, y)) {
        row.push("X");
      } else if (state.start && state.start.x === x && state.start.y === y) {
        row.push("S");
      } else if (isGoal(x, y)) {
        row.push("G");
      } else {
        const actionName = state.policy.get(keyOf(x, y));
        const action = actionByName(actionName);
        row.push(action ? action.arrow : "·");
      }
    }
    rows.push(row.join("\t"));
  }
  return rows.join("\n");
}

function formatValueMatrix() {
  const rows = [];
  for (let x = 0; x < state.n; x += 1) {
    const row = [];
    for (let y = 0; y < state.n; y += 1) {
      if (isObstacle(x, y)) {
        row.push("X");
      } else {
        row.push(state.values[x][y].toFixed(2));
      }
    }
    rows.push(row.join("\t"));
  }
  return rows.join("\n");
}

function updateMatrixDisplays() {
  policyMatrixEl.textContent = formatPolicyMatrix();
  valueMatrixEl.textContent = formatValueMatrix();
}

function runValueIteration() {
  initValues();
  let delta = Infinity;
  let iterations = 0;

  while (delta > EVAL_THETA && iterations < 1000) {
    delta = 0;
    const nextValues = state.values.map((row) => [...row]);

    for (let x = 0; x < state.n; x += 1) {
      for (let y = 0; y < state.n; y += 1) {
        if (isObstacle(x, y) || isTerminal(x, y)) {
          nextValues[x][y] = 0;
          continue;
        }

        let best = -Infinity;

        for (const action of ACTIONS) {
          const { nx, ny, reward } = transition(x, y, action);
          const candidate = reward + GAMMA * state.values[nx][ny];
          if (candidate > best) {
            best = candidate;
          }
        }

        delta = Math.max(delta, Math.abs(best - state.values[x][y]));
        nextValues[x][y] = best;
      }
    }

    state.values = nextValues;
    iterations += 1;
  }

  clearPolicy();

  for (let x = 0; x < state.n; x += 1) {
    for (let y = 0; y < state.n; y += 1) {
      if (isObstacle(x, y) || isTerminal(x, y)) {
        continue;
      }

      let bestAction = ACTIONS[0];
      let bestValue = -Infinity;

      for (const action of ACTIONS) {
        const { nx, ny, reward } = transition(x, y, action);
        const q = reward + GAMMA * state.values[nx][ny];
        if (q > bestValue) {
          bestValue = q;
          bestAction = action;
        }
      }

      state.policy.set(keyOf(x, y), bestAction.name);
    }
  }

  return iterations;
}

function onCellClick(x, y) {
  const k = keyOf(x, y);
  const limit = obstacleLimit();

  if (state.mode === "start") {
    if (isObstacle(x, y)) {
      state.obstacles.delete(k);
    }
    if (isGoal(x, y)) {
      state.goal = null;
    }
    state.start = { x, y };
  }

  if (state.mode === "goal") {
    if (isObstacle(x, y)) {
      state.obstacles.delete(k);
    }
    if (state.start && state.start.x === x && state.start.y === y) {
      state.start = null;
    }
    state.goal = { x, y };
  }

  if (state.mode === "obstacle") {
    if (state.start && state.start.x === x && state.start.y === y) {
      updateStatus("該格是起點，請先改放其他位置或切換到清除模式。", false);
      return;
    }
    if (state.goal && state.goal.x === x && state.goal.y === y) {
      updateStatus("該格是終點，請先改放其他位置或切換到清除模式。", false);
      return;
    }

    if (state.obstacles.has(k)) {
      state.obstacles.delete(k);
    } else if (state.obstacles.size < limit) {
      state.obstacles.add(k);
    } else {
      updateStatus(`障礙物上限為 n-2 = ${limit}。`, false);
      return;
    }
  }

  if (state.mode === "erase") {
    if (state.start && state.start.x === x && state.start.y === y) {
      state.start = null;
    }
    if (state.goal && state.goal.x === x && state.goal.y === y) {
      state.goal = null;
    }
    state.obstacles.delete(k);
  }

  clearPolicy();
  clearBestPath();
  initValues();
  renderGrid();
  updateStatus("地圖已更新。可執行策略評估或價值迭代。", false);
}

function renderGrid() {
  gridEl.innerHTML = "";
  gridEl.style.gridTemplateColumns = `repeat(${state.n}, minmax(0, 84px))`;

  for (let x = 0; x < state.n; x += 1) {
    for (let y = 0; y < state.n; y += 1) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";

      if (state.start && state.start.x === x && state.start.y === y) {
        cell.classList.add("start");
      }
      if (state.goal && state.goal.x === x && state.goal.y === y) {
        cell.classList.add("goal");
      }
      if (isObstacle(x, y)) {
        cell.classList.add("obstacle");
      }
      if (state.bestPath.has(keyOf(x, y)) && !isObstacle(x, y) && !isGoal(x, y)) {
        cell.classList.add("best-path");
      }

      const a = document.createElement("div");
      a.className = "arrow";

      if (isObstacle(x, y)) {
        a.textContent = "■";
      } else if (isTerminal(x, y)) {
        a.textContent = "G";
      } else {
        const actionName = state.policy.get(keyOf(x, y));
        if (actionName) {
          const action = actionByName(actionName);
          a.textContent = action.arrow;
        } else {
          a.textContent = "·";
        }
      }

      const v = document.createElement("div");
      v.className = "value";
      v.textContent = state.values[x][y].toFixed(2);

      cell.append(a, v);
      cell.addEventListener("click", () => onCellClick(x, y));
      gridEl.appendChild(cell);
    }
  }

  updateMetrics();
  updateMatrixDisplays();
}

function setupModeListeners() {
  const radios = document.querySelectorAll('input[name="mode"]');
  radios.forEach((radio) => {
    radio.addEventListener("change", (event) => {
      setMode(event.target.value);
      updateStatus(`已切換模式：${event.target.parentElement.textContent.trim()}`, false);
    });
  });
}

buildGridBtn.addEventListener("click", () => {
  state.n = clampGridSize(gridSizeEl.value);
  gridSizeEl.value = state.n;
  resetWorld();
});

randomPolicyBtn.addEventListener("click", () => {
  if (!ensureReadyForAlgorithms()) {
    return;
  }
  clearBestPath();
  generateRandomPolicy();
  const iters = evaluatePolicy();
  renderGrid();
  updateStatus("已顯示隨機策略，並完成 value function 策略評估。", false);
  metricsEl.textContent += ` | 評估迭代次數: ${iters}`;
});

valueIterationBtn.addEventListener("click", () => {
  if (!ensureReadyForAlgorithms()) {
    return;
  }
  const iters = runValueIteration();
  deriveBestPathFromPolicy();
  renderGrid();
  updateStatus("已完成價值迭代，並更新為最佳策略。", false);
  metricsEl.textContent += ` | 價值迭代次數: ${iters}`;
});

resetBtn.addEventListener("click", () => {
  resetWorld();
});

setupModeListeners();
resetWorld();
