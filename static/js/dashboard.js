/* Dashboard DQN — Clássica vs Quântica */
(() => {
  "use strict";

  const COLORS = {
    s1: "#3987e5", s2: "#d95926", s7: "#9085e9",
    grid: "rgba(195,194,183,0.12)", tick: "#8f8e86", text: "#c3c2b7",
  };
  const ARROWS = ["←", "↓", "→", "↑"];
  let DEFAULTS = null;

  // Se o CDN do Chart.js não carregar, o painel continua funcionando sem os gráficos
  if (typeof window.Chart === "undefined") {
    console.warn("Chart.js indisponível — gráficos desativados.");
    window.Chart = class { constructor() { this.data = { labels: [], datasets: [{ data: [] }, { data: [] }] }; } update() {} };
    window.Chart.defaults = { font: {} };
  }
  Chart.defaults.color = COLORS.text;
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.animation = false;

  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);

  function lineChart(canvas, datasets, yTitle) {
    return new Chart(canvas, {
      type: "line",
      data: { labels: [], datasets },
      options: {
        responsive: true, maintainAspectRatio: false, parsing: true, spanGaps: true,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: datasets.length > 1, labels: { boxWidth: 12, boxHeight: 2 } },
          tooltip: { callbacks: { title: (it) => `Episódio ${it[0].label}` } },
        },
        elements: { point: { radius: 0, hoverRadius: 4, hitRadius: 8 }, line: { borderWidth: 2, tension: 0.15 } },
        scales: {
          x: { title: { display: true, text: "episódio", color: COLORS.tick }, grid: { color: COLORS.grid }, ticks: { color: COLORS.tick, maxTicksLimit: 10 } },
          y: { title: { display: !!yTitle, text: yTitle, color: COLORS.tick }, grid: { color: COLORS.grid }, ticks: { color: COLORS.tick } },
        },
      },
    });
  }

  class Panel {
    constructor(root) {
      this.root = root;
      this.mode = root.dataset.mode;
      this.color = this.mode === "quantum" ? COLORS.s7 : COLORS.s1;
      this.jobId = null;
      this.count = 0;
      this.timer = null;
      this.$ = (sel) => root.querySelector(sel);
      this.form = root.querySelector(".config");
      this.envSel = this.form.querySelector("select[name=env]");
      this.status = this.$(".status");
      this.bar = this.$(".progress .bar");
      this.btnStart = this.$("[data-action=start]");
      this.btnStop = this.$("[data-action=stop]");
      this.btnPolicy = this.$("[data-action=policy]");

      this.charts = {
        reward: lineChart(this.$("[data-chart=reward]"), [
          { label: "Recompensa do episódio", data: [], borderColor: this.color, backgroundColor: this.color, borderWidth: 1.5 },
          { label: "Média móvel (20)", data: [], borderColor: COLORS.s2, backgroundColor: COLORS.s2, borderWidth: 2.5 },
        ], "recompensa"),
        loss: lineChart(this.$("[data-chart=loss]"), [
          { label: "Perda", data: [], borderColor: this.color, backgroundColor: this.color },
        ], "perda"),
        eps: lineChart(this.$("[data-chart=eps]"), [
          { label: "ε", data: [], borderColor: this.color, backgroundColor: this.color },
        ], "ε"),
      };

      if (this.mode === "quantum") this.initQuantum();

      this.envSel.addEventListener("change", () => { this.fillDefaults(); this.drawCircuit(); });
      this.btnStart.addEventListener("click", () => this.start());
      this.btnStop.addEventListener("click", () => this.stop());
      this.$("[data-action=reset]").addEventListener("click", () => { this.fillDefaults(); this.drawCircuit(); });
      this.btnPolicy.addEventListener("click", () => this.showPolicy());
    }

    field(name) { return this.form.querySelector(`[name=${name}]`); }

    fillDefaults() {
      const d = DEFAULTS[this.mode][this.envSel.value];
      for (const [k, v] of Object.entries(d)) {
        const el = this.field(k);
        if (el) el.value = v;
      }
    }

    readConfig() {
      const cfg = {};
      this.form.querySelectorAll("input[name]").forEach((el) => {
        if (el.name !== "seed" && el.value !== "") cfg[el.name] = Number(el.value);
      });
      return cfg;
    }

    setStatus(msg, err = false) {
      this.status.textContent = msg;
      this.status.classList.toggle("err", err);
    }

    resetCharts() {
      for (const ch of Object.values(this.charts)) {
        ch.data.labels = [];
        ch.data.datasets.forEach((d) => (d.data = []));
        ch.update();
      }
      this.count = 0;
    }

    async start() {
      this.resetCharts();
      this.btnStart.disabled = true;
      this.btnPolicy.disabled = true;
      this.setStatus("Iniciando treino…");
      const seed = this.field("seed").value;
      try {
        const res = await fetch("/api/train", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: this.mode, env: this.envSel.value, config: this.readConfig(), seed }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "falha ao iniciar");
        this.jobId = data.job_id;
        this.btnStop.disabled = false;
        this.setStatus(`Treinando (semente ${data.seed})…`);
        this.poll();
      } catch (e) {
        this.setStatus(e.message, true);
        this.btnStart.disabled = false;
      }
    }

    async stop() {
      if (!this.jobId) return;
      await fetch(`/api/stop/${this.jobId}`, { method: "POST" });
      this.setStatus("Parando…");
    }

    async poll() {
      clearTimeout(this.timer);
      try {
        const res = await fetch(`/api/status/${this.jobId}?since=${this.count}`);
        const s = await res.json();
        if (!res.ok) throw new Error(s.error);
        this.ingest(s);
        if (s.done) return this.finish(s);
      } catch (e) {
        this.setStatus(e.message, true);
        this.btnStart.disabled = false;
        this.btnStop.disabled = true;
        return;
      }
      this.timer = setTimeout(() => this.poll(), 800);
    }

    ingest(s) {
      const h = s.history;
      const n = h.reward.length;
      for (let i = 0; i < n; i++) {
        const ep = this.count + i + 1;
        this.charts.reward.data.labels.push(ep);
        this.charts.reward.data.datasets[0].data.push(h.reward[i]);
        this.charts.reward.data.datasets[1].data.push(h.avg[i]);
        this.charts.loss.data.labels.push(ep);
        this.charts.loss.data.datasets[0].data.push(h.loss[i]);
        this.charts.eps.data.labels.push(ep);
        this.charts.eps.data.datasets[0].data.push(h.epsilon[i]);
      }
      this.count += n;
      if (n) Object.values(this.charts).forEach((c) => c.update());

      const avg = this.charts.reward.data.datasets[1].data.at(-1);
      const eps = this.charts.eps.data.datasets[0].data.at(-1);
      const dig = s.env === "cartpole" ? 1 : 3;
      this.kpi("episode", `${s.episode}/${s.episodes}`);
      this.kpi("avg", fmt(avg, dig));
      this.kpi("best", fmt(s.best_avg, dig));
      this.kpi("eps", fmt(eps, 3));
      this.kpi("params", s.n_params ?? "–");
      this.kpi("time", `${fmt(s.elapsed, 1)} s`);
      this.bar.style.width = `${(100 * s.episode) / s.episodes}%`;
      if (!s.done) this.setStatus(`Treinando… ${s.total_steps} passos de interação`);
    }

    kpi(name, val) { this.$(`[data-kpi=${name}]`).textContent = val; }

    finish(s) {
      this.btnStart.disabled = false;
      this.btnStop.disabled = true;
      this.btnPolicy.disabled = false;
      if (s.status === "erro") this.setStatus(`Erro: ${s.error}`, true);
      else {
        const be = s.best_eval === null ? "–" : s.env === "cartpole" ? `${s.best_eval} passos` : `retorno ${fmt(s.best_eval, 2)}`;
        this.setStatus(`Treino ${s.status} em ${fmt(s.elapsed, 1)} s · ${s.total_steps} passos · melhor política gulosa: ${be}.`);
      }
      this.showPolicy();
      if (this.mode === "quantum") this.loadStateVector();
    }

    async showPolicy() {
      if (!this.jobId) return;
      const box = this.$("[data-viz=policy]");
      const p = await (await fetch(`/api/policy/${this.jobId}`)).json();
      if (p.error) { box.innerHTML = `<p class="hint">${p.error}</p>`; return; }
      if (p.env === "gridworld") this.renderGrid(box, p);
      else this.renderCartPole(box, p);
    }

    renderGrid(box, p) {
      const flat = p.layout.join("");
      const path = new Set(p.path);
      let html = '<div class="gw">';
      for (let i = 0; i < 16; i++) {
        const t = flat[i];
        const cls = t === "H" ? "hole" : t === "G" ? "goal" : "";
        const icon = t === "H" ? "🕳️" : t === "G" ? "🏁" : ARROWS[p.greedy[i]];
        const qmax = Math.max(...p.q[i]).toFixed(2);
        const label = t === "S" ? `${icon}` : icon;
        html += `<div class="cell ${cls} ${path.has(i) && t !== "H" ? "path" : ""}"
                   title="estado ${i} · Q = [${p.q[i].map((v) => v.toFixed(2)).join(", ")}]">${label}<small>${qmax}</small></div>`;
      }
      html += "</div>";
      html += `<p class="gw-legend">Setas = ação gulosa argmax Q · número = max Q · borda laranja = trajetória ·
        ${p.success ? '<span class="ok">✔ objetivo alcançado</span>' : '<span class="bad">✘ não alcançou o objetivo</span>'}
        (${p.path.length - 1} passos)</p>`;
      box.innerHTML = html;
    }

    renderCartPole(box, p) {
      box.innerHTML = `<canvas class="cartpole" width="640" height="220"></canvas>
        <p class="gw-legend">Política gulosa (melhor modelo): <strong>${p.steps}</strong> passos equilibrando
        ${p.steps >= 500 ? '<span class="ok">✔ máximo atingido</span>' : ""} · <a href="#" data-replay>repetir</a></p>`;
      const cv = box.querySelector("canvas");
      const ctx = cv.getContext("2d");
      let i = 0;
      cancelAnimationFrame(this.anim);
      const draw = () => {
        const [x, th] = p.frames[Math.min(i, p.frames.length - 1)];
        const W = cv.width, H = cv.height, sx = W / 5.2, cx = W / 2 + x * sx, cy = H - 50;
        ctx.clearRect(0, 0, W, H);
        ctx.strokeStyle = "#55544f"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0, cy + 16); ctx.lineTo(W, cy + 16); ctx.stroke();
        ctx.fillStyle = this.color; ctx.fillRect(cx - 32, cy - 12, 64, 28);
        ctx.strokeStyle = COLORS.s2; ctx.lineWidth = 7; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(cx, cy - 6);
        ctx.lineTo(cx + Math.sin(th) * 130, cy - 6 - Math.cos(th) * 130); ctx.stroke();
        ctx.fillStyle = "#f0efec"; ctx.font = "13px JetBrains Mono";
        ctx.fillText(`passo ${Math.min(i, p.steps)}  x=${x.toFixed(2)}  θ=${(th * 57.3).toFixed(1)}°`, 10, 20);
        i++;
        if (i < p.frames.length) this.anim = requestAnimationFrame(() => setTimeout(draw, 16));
      };
      draw();
      box.querySelector("[data-replay]").addEventListener("click", (e) => { e.preventDefault(); this.renderCartPole(box, p); });
    }

    /* -------------------- quântico -------------------- */
    initQuantum() {
      this.svChart = new Chart(this.$("[data-chart=sv]"), {
        type: "bar",
        data: { labels: [], datasets: [{ label: "|amplitude|²", data: [], backgroundColor: COLORS.s7, borderRadius: 4, borderSkipped: "bottom" }] },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: { title: (it) => `|${it[0].label}⟩` } } },
          scales: {
            x: { grid: { display: false }, ticks: { color: COLORS.tick, font: { family: "JetBrains Mono", size: 10 } } },
            y: { min: 0, title: { display: true, text: "probabilidade", color: COLORS.tick }, grid: { color: COLORS.grid }, ticks: { color: COLORS.tick } },
          },
        },
      });
      const inp = this.$("[data-input=sv-state]");
      inp.addEventListener("change", () => this.loadStateVector());
      this.field("layers").addEventListener("change", () => this.drawCircuit());
    }

    async loadStateVector() {
      if (!this.jobId) return;
      const s = this.$("[data-input=sv-state]").value || 0;
      const d = await (await fetch(`/api/statevector/${this.jobId}?s=${s}`)).json();
      if (d.error) return;
      this.svChart.data.labels = d.basis;
      this.svChart.data.datasets[0].data = d.probs;
      this.svChart.update();
      const note = this.$("[data-sv-note]");
      const ctx = this.envSel.value === "cartpole" ? "estado de exemplo do CartPole (θ = 0,05 rad)" : `estado s = ${s}`;
      note.textContent = `${ctx} · ⟨Z⟩ por qubit = [${d.expect_z.map((v) => v.toFixed(3)).join(", ")}]`;
    }

    async drawCircuit() {
      if (this.mode !== "quantum") return;
      const layers = this.field("layers").value || 3;
      const c = await (await fetch(`/api/circuit?layers=${layers}&env=${this.envSel.value}`)).json();
      // agrupa as rotações de uma camada em 3 colunas (RX, RY, RZ) para todos os qubits
      const cols = [];
      const rotCol = {};
      for (const op of c.ops) {
        if (op.type === "rot") {
          const key = `${op.layer}-${op.gate}`;
          if (!rotCol[key]) { rotCol[key] = { kind: "rot", gate: op.gate, ops: [] }; cols.push(rotCol[key]); }
          rotCol[key].ops.push(op);
        } else {
          cols.push({ kind: op.type, op });
        }
      }
      const n = c.n_qubits, dx = 46, dy = 40, x0 = 48, y0 = 26;
      const W = x0 + cols.length * dx + 40, H = y0 + (n - 1) * dy + 34;
      let svg = `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" font-family="JetBrains Mono" font-size="11">`;
      for (let q = 0; q < n; q++) {
        const y = y0 + q * dy;
        svg += `<text x="6" y="${y + 4}" fill="#c3c2b7">q${q}</text>`;
        svg += `<line x1="34" y1="${y}" x2="${W - 10}" y2="${y}" stroke="#55544f" stroke-width="1.5"/>`;
        if (q < c.n_qubits) svg += `<text x="${W - 34}" y="${y - 6}" fill="#8f8e86">⟨Z⟩</text>`;
      }
      cols.forEach((col, i) => {
        const x = x0 + i * dx + dx / 2;
        if (col.kind === "encode") {
          svg += `<rect x="${x - 19}" y="${y0 - 14}" width="38" height="${(n - 1) * dy + 28}" rx="5" fill="#2d2c29" stroke="#d95926"/>`;
          svg += `<text x="${x}" y="${y0 + ((n - 1) * dy) / 2 + 4}" fill="#f0efec" text-anchor="middle" font-size="9"
                   transform="rotate(-90 ${x} ${y0 + ((n - 1) * dy) / 2})">${col.op.label}</text>`;
        } else if (col.kind === "rot") {
          for (const op of col.ops) {
            const y = y0 + op.qubit * dy;
            svg += `<g><title>${op.gate}(θ${op.param}) — parâmetro treinável</title>
              <rect x="${x - 17}" y="${y - 12}" width="34" height="24" rx="4" fill="#2b2846" stroke="#9085e9"/>
              <text x="${x}" y="${y + 4}" fill="#f0efec" text-anchor="middle">${op.gate}</text></g>`;
          }
        } else {
          const yc = y0 + col.op.control * dy, yt = y0 + col.op.target * dy;
          svg += `<line x1="${x}" y1="${yc}" x2="${x}" y2="${yt}" stroke="#c3c2b7" stroke-width="1.5"/>`;
          svg += `<circle cx="${x}" cy="${yc}" r="4.5" fill="#c3c2b7"/>`;
          svg += `<circle cx="${x}" cy="${yt}" r="10" fill="#1a1a19" stroke="#c3c2b7" stroke-width="1.5"/>`;
          svg += `<line x1="${x - 10}" y1="${yt}" x2="${x + 10}" y2="${yt}" stroke="#c3c2b7" stroke-width="1.5"/>`;
          svg += `<line x1="${x}" y1="${yt - 10}" x2="${x}" y2="${yt + 10}" stroke="#c3c2b7" stroke-width="1.5"/>`;
        }
      });
      svg += "</svg>";
      this.$("[data-viz=circuit]").innerHTML = svg;
    }
  }

  /* -------------------- abas -------------------- */
  function initTabs() {
    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
        window.dispatchEvent(new Event("resize"));
      });
    });
  }

  async function main() {
    initTabs();
    DEFAULTS = await (await fetch("/api/defaults")).json();
    document.querySelectorAll(".panel").forEach((root) => {
      const p = new Panel(root);
      p.fillDefaults();
      p.drawCircuit();
    });
  }
  main();
})();
