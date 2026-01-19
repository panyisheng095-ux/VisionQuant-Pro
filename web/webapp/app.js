const { createApp } = Vue;

createApp({
  data() {
    return {
      dataSource: "akshare",
      mode: "single",
      symbol: "601899",
      date: "",
      topK: 10,
      multiScale: true,
      batchInput: "600519\n000001\n300750",
      loading: false,
      error: "",

      visual: null,
      overview: null,
      batch: null,
      activeTab: "bt",

      icSymbol: "600519",
      icStart: "20100101",
      icEnd: "",
      icLoading: false,
      icError: "",
      icResult: null,
    };
  },
  computed: {
    reportTitle() {
      if (this.overview && this.overview.name) {
        return `${this.overview.name} (${this.symbol})`;
      }
      return this.symbol;
    },
  },
  methods: {
    async runAnalyze() {
      this.error = "";
      this.visual = null;
      this.overview = null;
      this.batch = null;
      this.loading = true;

      try {
        if (this.mode === "single") {
          const visualResp = await fetch("/api/visual/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              symbol: this.symbol.trim(),
              date: this.date.trim() || null,
              top_k: this.topK,
              multi_scale: this.multiScale,
              include_images: true,
            }),
          });
          if (!visualResp.ok) {
            const err = await visualResp.json();
            throw new Error(err.detail || "视觉识别失败");
          }
          this.visual = await visualResp.json();

          const overviewResp = await fetch("/api/single/overview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              symbol: this.symbol.trim(),
              visual_win_rate: this.visual?.metrics?.hybrid_win_rate ?? 50,
            }),
          });
          if (!overviewResp.ok) {
            const err = await overviewResp.json();
            throw new Error(err.detail || "概览数据失败");
          }
          this.overview = await overviewResp.json();

          this.$nextTick(() => {
            this.renderSimilarityChart();
            this.renderTrajectoryChart();
          });
        } else {
          const symbols = this.batchInput
            .split(/\s+/)
            .map((s) => s.trim())
            .filter((s) => s.length > 0);
          const resp = await fetch("/api/batch/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              symbols,
              risk_aversion: 1.0,
              cvar_limit: 0.05,
            }),
          });
          if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "批量分析失败");
          }
          this.batch = await resp.json();
        }
      } catch (e) {
        this.error = e.message || "请求失败";
      } finally {
        this.loading = false;
      }
    },
    async runIC() {
      this.icError = "";
      this.icResult = null;
      this.icLoading = true;
      try {
        const resp = await fetch("/factor/ic", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            symbol: this.icSymbol.trim(),
            start_date: this.icStart.trim() || "20100101",
            end_date: this.icEnd.trim() || "",
            robust: true,
          }),
        });
        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || "请求失败");
        }
        this.icResult = await resp.json();
      } catch (e) {
        this.icError = e.message || "请求失败";
      } finally {
        this.icLoading = false;
      }
    },
    renderSimilarityChart() {
      if (!this.visual || !this.visual.matches || !window.Plotly) return;
      const labels = this.visual.matches.map((m, i) => `${i + 1}-${m.symbol}`);
      const dtw = this.visual.matches.map((m) => m.dtw_sim ?? 0);
      const corr = this.visual.matches.map((m) => m.correlation ?? 0);
      const feat = this.visual.matches.map((m) => m.feature_sim ?? 0);
      const sim = this.visual.matches.map((m) => m.sim_score ?? 0);

      const data = [
        { x: labels, y: dtw, name: "DTW", type: "bar" },
        { x: labels, y: corr, name: "Corr", type: "bar" },
        { x: labels, y: feat, name: "形态", type: "bar" },
        { x: labels, y: sim, name: "视觉", type: "bar" },
      ];
      const layout = {
        barmode: "group",
        margin: { t: 20, r: 10, b: 60, l: 40 },
        height: 300,
        legend: { orientation: "h" },
        xaxis: { tickangle: -45 },
        yaxis: { rangemode: "tozero" },
      };
      Plotly.newPlot("chart-sim", data, layout, { displayModeBar: false });
    },
    renderTrajectoryChart() {
      if (!this.visual || !this.visual.trajectories || !window.Plotly) return;
      const traj = this.visual.trajectories;
      if (!traj.trajectories || traj.trajectories.length === 0) return;

      const traces = traj.trajectories.map((p, i) => ({
        y: p,
        mode: "lines",
        line: { color: "rgba(180,180,180,0.5)", width: 1 },
        name: traj.labels[i] || `M${i + 1}`,
      }));
      if (traj.mean_path) {
        traces.push({
          y: traj.mean_path,
          mode: "lines+markers",
          line: { color: "#d62728", width: 3 },
          name: "平均预期",
        });
      }
      const layout = {
        height: 320,
        margin: { t: 10, r: 10, b: 30, l: 40 },
        xaxis: { title: "天数" },
        yaxis: { title: "收益%" },
      };
      Plotly.newPlot("chart-traj", traces, layout, { displayModeBar: false });
    },
    formatNum(v) {
      if (v === null || v === undefined || Number.isNaN(v)) return "-";
      return Number(v).toFixed(3);
    },
    formatPercent(v) {
      if (v === null || v === undefined || Number.isNaN(v)) return "-";
      return `${Number(v).toFixed(2)}%`;
    },
  },
}).mount("#app");
