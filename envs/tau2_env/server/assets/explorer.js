// Task explorer: search and filter on the client, and report the opened task as a `select` event.
// The list is re-rendered when the domain or split changes, so nothing is looked up ahead of time.
let filter = "all";
let query = "";

function apply() {
  const cards = [...element.querySelectorAll("[data-task]")];
  let shown = 0;
  for (const card of cards) {
    const visible =
      (filter === "all" || card.dataset.writes === filter) && card.dataset.search.includes(query);
    card.hidden = !visible;
    shown += visible;
  }
  const count = element.querySelector(".t2-count");
  if (count) count.textContent = `${shown} of ${cards.length} tasks`;
}

element.addEventListener("input", (ev) => {
  if (!ev.target.classList.contains("t2-q")) return;
  query = ev.target.value.trim().toLowerCase();
  apply();
});

element.addEventListener("click", (ev) => {
  const button = ev.target.closest("[data-filter]");
  if (button) {
    filter = button.dataset.filter;
    element.querySelectorAll("[data-filter]").forEach((b) => b.classList.toggle("active", b === button));
    apply();
    return;
  }
  const card = ev.target.closest("[data-task]");
  if (!card) return;
  element.querySelectorAll("[data-task]").forEach((c) => c.classList.toggle("active", c === card));
  trigger("select", { index: Number(card.dataset.index), value: card.dataset.task });
});

// A new list starts unfiltered.
new MutationObserver(() => {
  filter = "all";
  query = "";
}).observe(element, { childList: true });
