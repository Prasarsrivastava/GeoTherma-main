const ctx = document.getElementById('trendChart').getContext('2d');
new Chart(ctx, {
  type: 'line',
  data: {
    labels: ['2013', '2018', '2023'],
    datasets: [{
      label: 'Avg Temp (°C)',
      data: [29.5, 30.2, 31.7],
      borderColor: 'red',
      fill: false
    }]
  }
});
