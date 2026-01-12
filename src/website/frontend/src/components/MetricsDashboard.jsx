import React from 'react'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

function MetricsDashboard({ metrics }) {
  if (!metrics) return <div>Loading metrics...</div>

  const testMetrics = metrics.test_metrics || {}

  // Prepare data for charts
  const metricsData = [
    { name: 'MSE', value: testMetrics['test/position_loss'] || 0 },
    { name: 'MAE', value: testMetrics['test/mae'] || 0 },
    { name: 'RMSE', value: testMetrics['test/rmse'] || 0 },
    { name: 'R²', value: testMetrics['test/r2'] || 0 },
  ]

  return (
    <div className="metrics-dashboard">
      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Best Validation Loss</h3>
          <p className="metric-value">{metrics.best_val_loss?.toFixed(6)}</p>
        </div>

        <div className="metric-card">
          <h3>Test R² Score</h3>
          <p className="metric-value">{(testMetrics['test/r2'] || 0).toFixed(4)}</p>
        </div>

        <div className="metric-card">
          <h3>Test MAE</h3>
          <p className="metric-value">{(testMetrics['test/mae'] || 0).toFixed(6)}</p>
        </div>

        <div className="metric-card">
          <h3>Test RMSE</h3>
          <p className="metric-value">{(testMetrics['test/rmse'] || 0).toFixed(6)}</p>
        </div>
      </div>

      <div className="charts">
        <div className="chart-container">
          <h3>Test Metrics Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={metricsData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="config-info">
        <h3>Configuration</h3>
        <pre>{JSON.stringify(metrics.config, null, 2)}</pre>
      </div>
    </div>
  )
}

export default MetricsDashboard
