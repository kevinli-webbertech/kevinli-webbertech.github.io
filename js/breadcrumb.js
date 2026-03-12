(function () {
  function labelize(segment) {
    return decodeURIComponent(segment)
      .replace(/[-_]+/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  }

  function buildBreadcrumb() {
    const nav = document.querySelector("#breadcrumb") || document.querySelector("[data-breadcrumb]");
    if (!nav) return;

    const path = window.location.pathname
      .replace(/\/+/g, "/")
      .replace(/index\.html$/i, "")
      .replace(/\.html$/i, "");

    const parts = path.split("/").filter(Boolean);
    if (!parts.length) return;

    let html = '<ol class="breadcrumb__list"><li class="breadcrumb__item"><a href="/">Home</a></li>';
    let acc = "";

    for (let i = 0; i < parts.length; i++) {
      acc += "/" + parts[i];
      if (parts[i] === "html" && parts[i - 1] === "blog") continue;

      const isLast = i === parts.length - 1;
      const text = labelize(parts[i]);
      html += isLast
        ? `<li class="breadcrumb__item" aria-current="page">${text}</li>`
        : `<li class="breadcrumb__item"><a href="${acc}/">${text}</a></li>`;
    }

    html += "</ol>";
    nav.innerHTML = html;
  }

  document.addEventListener("DOMContentLoaded", buildBreadcrumb);
})();
