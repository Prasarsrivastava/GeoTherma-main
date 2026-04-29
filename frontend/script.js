const map = L.map('map').setView([12.9716, 77.5946], 11);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

// Sample heatmap data (replace with backend fetch)
const heat = L.heatLayer([
  [12.9716, 77.5946, 0.5],
  [28.6139, 77.2090, 0.8]
], {radius: 25}).addTo(map);
