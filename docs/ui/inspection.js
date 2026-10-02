"use strict";
const cases = [...document.querySelectorAll(".case")];
const picker = document.querySelector(".case-picker");
const workspace = document.querySelector(".workspace");
const previous = document.querySelector("#previous-case");
const next = document.querySelector("#next-case");
let selected = 0;
function setSection(section) {
  workspace.dataset.section = section;
  document.querySelectorAll("a[data-section]").forEach(link => {
    if (link.dataset.section === section) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (section === "map" && window.Plotly) {
    requestAnimationFrame(() => document.querySelectorAll(".plotly-graph-div").forEach(plot => window.Plotly.Plots.resize(plot)));
  }
}
function selectCase(index) {
  selected = Math.max(0, Math.min(cases.length - 1, index));
  cases.forEach((panel, n) => {
    panel.hidden = n !== selected;
    picker.children[n].setAttribute("aria-pressed", String(n === selected));
  });
  const current = cases[selected];
  const label = picker.children[selected].querySelector(".case-name").textContent;
  document.querySelector("#case-label").textContent = `사례 ${String(selected+1).padStart(2,"0")} / ${String(cases.length).padStart(2,"0")} · ${label}`;
  document.querySelector("#selection-status").textContent = `선택: ${label}`;
  document.querySelector("#validation-status").textContent = current.querySelector(".card-head .ok, .card-head .bad")?.textContent || "기록된 판독 카드";
  previous.disabled = selected === 0;
  next.disabled = selected === cases.length - 1;
  setSection("inspection");
  workspace.scrollTop = 0;
}
if (picker && cases.length) {
  cases.forEach((item, index) => {
    [...item.querySelectorAll(".maps .big")].forEach((figure, i) => {
      figure.classList.add(i ? "heat-figure" : "original-figure");
      const label = document.createElement("span");
      label.className = "map-label";
      label.textContent = i ? "GRAD-CAM" : "WAFER MAP";
      figure.prepend(label);
    });
    const unknown = !!item.querySelector(".pattern .warn");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-button";
    const number = document.createElement("span");
    number.className = "case-number";
    number.textContent = String(index+1).padStart(2,"0");
    const dot = document.createElement("i");
    dot.className = "case-dot" + (unknown ? " unknown" : "");
    const name = document.createElement("span");
    name.className = "case-name";
    name.textContent = item.querySelector(".pattern").childNodes[0].textContent.trim();
    const badge = document.createElement("span");
    badge.className = "case-badge";
    badge.textContent = unknown ? "확인" : "분류";
    button.append(number, dot, name, badge);
    button.setAttribute("aria-controls", item.id);
    button.addEventListener("click", () => selectCase(index));
    button.addEventListener("keydown", event => {
      if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        event.preventDefault(); selectCase(Math.min(index+1,cases.length-1)); picker.children[selected].focus();
      } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        event.preventDefault(); selectCase(Math.max(index-1,0)); picker.children[selected].focus();
      }
    });
    picker.append(button);
  });
  selectCase(0);
  previous.addEventListener("click", () => selectCase(selected-1));
  next.addEventListener("click", () => selectCase(selected+1));
}
document.querySelectorAll("[data-map-view]").forEach(button => {
  if (button.tagName !== "BUTTON") return;
  button.addEventListener("click", () => {
    setSection("inspection");
    workspace.dataset.mapView = button.dataset.mapView;
    document.querySelectorAll("button[data-map-view]").forEach(other => other.setAttribute("aria-pressed",String(other === button)));
  });
});
function showSection() {
  setSection(location.hash === "#map" ? "map" : "inspection");
}
document.querySelectorAll("a[data-section]").forEach(link => link.addEventListener("click", () => setSection(link.dataset.section)));
document.querySelectorAll("[data-open]").forEach(link => link.addEventListener("click", () => {
  document.getElementById(link.dataset.open).open = true;
}));
window.addEventListener("hashchange",showSection);
showSection();
