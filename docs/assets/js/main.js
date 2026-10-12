"use strict";

// Progressive enhancement: both result tables remain readable without JavaScript.
const tabList = document.querySelector(".dataset-tabs");
if (tabList) {
  const tabs = [...tabList.querySelectorAll('[role="tab"]')];
  const selectTab = (selected, focus = false) => {
    for (const tab of tabs) {
      const active = tab === selected;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      document.getElementById(tab.getAttribute("aria-controls")).hidden = !active;
    }
    if (focus) selected.focus();
  };

  for (const [index, tab] of tabs.entries()) {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      selectTab(tabs[next], true);
    });
  }
  selectTab(tabs[0]);
  tabList.hidden = false;
  document.documentElement.classList.add("js-enabled");
}

const copyButton = document.getElementById("copy-bibtex");
const citation = document.getElementById("bibtex-code");
const copyStatus = document.getElementById("copy-status");
if (copyButton && citation && copyStatus) {
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(citation.textContent.trim() + "\n");
      copyStatus.textContent = "Citation copied.";
      copyButton.textContent = "Copied!";
      window.setTimeout(() => { copyButton.textContent = "Copy citation"; }, 2200);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(citation);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copyStatus.textContent = "Citation selected. Press Ctrl+C or Command+C to copy.";
    }
  });
}
