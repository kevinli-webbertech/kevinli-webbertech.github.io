(function () {
  'use strict';
  function resizeChart(frame) {
    const width = Math.floor(frame.getBoundingClientRect().width);
    if (width < 1) return;
    const url = new URL(frame.src);
    if (url.searchParams.get('width') === String(width)) return;
    url.searchParams.set('width', String(width));
    url.searchParams.set('height', '475');
    frame.src = url.href;
  }
  document.querySelectorAll('iframe[data-fred-chart]').forEach(function (frame) {
    resizeChart(frame);
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(function () { resizeChart(frame); });
      observer.observe(frame);
    } else {
      window.addEventListener('resize', function () { resizeChart(frame); });
    }
  });

  const indexNames = { NASDAQ100: 'Nasdaq 100', SP500: 'S&P 500', DJIA: 'Dow Jones Industrial Average (DJI)' };
  document.querySelectorAll('[data-index-series]').forEach(function (button) {
    button.addEventListener('click', function () {
      const series = button.dataset.indexSeries;
      const frame = document.getElementById('selected-index-chart');
      if (!frame || !Object.prototype.hasOwnProperty.call(indexNames, series)) return;
      const url = new URL(frame.src);
      if (url.searchParams.get('id') !== series) {
        url.searchParams.set('id', series);
        frame.src = url.href;
      }
      frame.title = 'FRED: ' + indexNames[series];
      document.getElementById('selected-index-title').textContent = indexNames[series];
      const source = document.getElementById('selected-index-source');
      source.href = 'https://fred.stlouisfed.org/series/' + series;
      source.textContent = 'Open ' + series + ' on FRED';
      document.querySelectorAll('[data-index-series]').forEach(function (other) {
        other.setAttribute('aria-pressed', String(other === button));
      });
    });
  });
}());
