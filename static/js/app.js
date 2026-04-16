const ta = document.getElementById("text");
const counter = document.getElementById("counter");

function updateCounter() {
  if (!ta || !counter) return;
  counter.textContent = (ta.value || "").length.toString();
}

if (ta) {
  ta.addEventListener("input", updateCounter);
  window.addEventListener("load", updateCounter);
}

document.addEventListener("DOMContentLoaded", function () {
  const analyzeForm = document.getElementById("analyzeForm");
  const analysisForm = document.querySelector(".result__analysis-form");
  const overlay = document.getElementById("loadingOverlay");
  const spinner = document.getElementById("loadingSpinner");
  const title = document.getElementById("loadingTitle");
  const subtitle = document.getElementById("loadingSubtitle");
  const predictBtn = document.getElementById("predictBtn");
  const analysisBtn = document.querySelector(".result__analysis-btn");

  if (!overlay || !spinner || !title || !subtitle) return;

  function showOverlay(mode) {
    overlay.style.display = "flex";
    spinner.className = "spinner";

    if (mode === "extended") {
      spinner.classList.add("spinner--green");
      title.textContent = "Идёт расширенный анализ";
      subtitle.textContent = "Это может занять некоторое время. Пожалуйста, подождите…";
    } else {
      title.textContent = "Идёт анализ текста";
      subtitle.textContent = "Это может занять некоторое время. Пожалуйста, подождите…";
    }
  }

  function resetOverlay() {
    overlay.style.display = "none";
    spinner.className = "spinner";
    title.textContent = "Идёт анализ текста";
    subtitle.textContent = "Это может занять некоторое время. Пожалуйста, подождите…";

    if (analyzeForm) {
      analyzeForm.classList.remove("form--loading");
    }
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.textContent = "Проанализировать текст";
    }
    if (analysisBtn) {
      analysisBtn.disabled = false;
      analysisBtn.textContent = "Расширенный анализ";
    }
  }

  if (analyzeForm) {
    analyzeForm.addEventListener("submit", function () {
      showOverlay("predict");
      analyzeForm.classList.add("form--loading");
      if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.textContent = "Идёт анализ...";
      }
    });
  }

  if (analysisForm) {
    analysisForm.addEventListener("submit", function () {
      showOverlay("extended");
      if (analysisBtn) {
        analysisBtn.disabled = true;
        analysisBtn.textContent = "Идёт расширенный анализ...";
      }
    });
  }

  window.addEventListener("pageshow", resetOverlay);
});
