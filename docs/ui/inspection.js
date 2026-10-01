"use strict";
const cases = [...document.querySelectorAll(".case")];
const picker = document.querySelector(".case-picker");
if (picker && cases.length) {
  cases.forEach((item, index) => {
    const maps = document.createElement("div");
    maps.className = "maps";
    [...item.querySelectorAll(":scope > figure")].forEach((figure, i) => {
      const label = document.createElement("span");
      label.className = "map-label";
      label.textContent = i ? "02 / GRAD-CAM" : "01 / WAFER MAP";
      figure.prepend(label);
      const image = figure.querySelector("img");
      image.alt = figure.querySelector("figcaption").textContent;
      maps.append(figure);
    });
    const legend = document.createElement("div");
    legend.className = "legend";
    legend.innerHTML = '<span><i></i>정상 다이</span><span><i class="fail"></i>불량 다이</span><span>히트맵: 모델이 주목한 영역</span>';
    maps.append(legend);
    item.prepend(maps);
    item.querySelectorAll(".sims img").forEach(img => {img.alt = img.nextElementSibling.textContent;});
    const button = document.createElement("button");
    button.type = "button";
    button.className = "case-button";
    button.textContent = `${String(index + 1).padStart(2,"0")} · ${item.querySelector(".pattern").childNodes[0].textContent.trim()}`;
    button.setAttribute("aria-controls", `case-${index}`);
    item.id = `case-${index}`;
    button.setAttribute("aria-pressed", String(index === 0));
    item.hidden = index !== 0;
    button.addEventListener("click", () => {
      cases.forEach((panel, n) => {
        panel.hidden = n !== index;
        picker.children[n].setAttribute("aria-pressed", String(n === index));
      });
    });
    picker.append(button);
  });
}
