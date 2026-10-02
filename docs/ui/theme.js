"use strict";
(() => {
  const root = document.documentElement;
  let theme = "light";
  try { theme = localStorage.getItem("wafer-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"); } catch (_) {}
  if (theme !== "dark") theme = "light";
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
  function apply(next) {
    root.dataset.theme = next;
    root.style.colorScheme = next;
    const dark = next === "dark";
    const button = document.querySelector("#theme-toggle");
    button.textContent = dark ? "☀ 라이트" : "◐ 다크";
    button.setAttribute("aria-label", dark ? "라이트 모드로 전환" : "다크 모드로 전환");
    button.setAttribute("aria-pressed", String(dark));
    document.querySelectorAll(".plotly-graph-div").forEach(plot => {
      if (window.Plotly && plot.layout) window.Plotly.relayout(plot, {
        paper_bgcolor: dark ? "#1b1c1e" : "#f6f7f9",
        plot_bgcolor: dark ? "#1b1c1e" : "#f6f7f9",
        "font.color": dark ? "#d8dade" : "#252b35",
        "xaxis.gridcolor": dark ? "#313236" : "#d8dde5",
        "yaxis.gridcolor": dark ? "#313236" : "#d8dde5"
      });
    });
  }
  document.addEventListener("DOMContentLoaded", () => {
    apply(root.dataset.theme);
    document.querySelector("#theme-toggle").addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem("wafer-theme", next); } catch (_) {}
      apply(next);
    });
    window.addEventListener("load", () => apply(root.dataset.theme), {once:true});
  });
})();
