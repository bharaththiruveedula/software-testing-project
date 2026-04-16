"""
HTML Report Generator Module

Generates a standalone HTML report with interactive charts (using Chart.js via CDN)
for test results.
"""

import json
from html import escape
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

from load_runner import LoadTestResult
from comparator import ComparisonMetric

if TYPE_CHECKING:
    from niah_runner import NIAHTestResult
    from ramp_runner import RampTestResult

# To avoid circular imports if any, but better use string annotations or direct import
def generate_html_report(
    result: LoadTestResult,
    output_path: str = "report.html",
    pass_rate_threshold: float = 0.95,
    ramp_result: Optional["RampTestResult"] = None,
) -> None:
    """
    Generate an HTML report with charts for the load test results.

    Args:
        result: The LoadTestResult containing performance and validation metrics.
        output_path: Path to write the HTML file.
        pass_rate_threshold: Minimum fraction of valid responses to pass (0.0–1.0).
    """
    tracker = result.tracker
    summary = tracker.summary()

    # Calculate validation pass rate
    valid_count = sum(1 for v in result.validation_results if v.is_valid)
    total_validated = len(result.validation_results)
    pass_rate = valid_count / total_validated if total_validated > 0 else 0.0
    passed = pass_rate >= pass_rate_threshold and summary["successful_requests"] > 0

    # Prepare chart data
    latencies = [r.latency_ms for r in tracker.results if r.success]
    
    # Histogram data (10 bins)
    if latencies:
        min_lat = min(latencies)
        max_lat = max(latencies)
        bin_count = 10
        bin_width = (max_lat - min_lat) / bin_count if max_lat > min_lat else 1
        
        bins = [0] * bin_count
        labels = [f"{(min_lat + i * bin_width):.0f}-{min_lat + (i + 1) * bin_width:.0f}ms" for i in range(bin_count)]
        
        for lat in latencies:
            b = int((lat - min_lat) / bin_width)
            if b >= bin_count:
                b = bin_count - 1
            bins[b] += 1
    else:
        labels = []
        bins = []

    # Timeline data
    timeline_labels = [f"Req #{r.request_id}" for r in tracker.results if r.success]
    timeline_data = [r.latency_ms for r in tracker.results if r.success]

    # JSON injects
    chart_data = {
        "success": summary["successful_requests"],
        "failed": summary["failed_requests"],
        "histLabels": labels,
        "histData": bins,
        "timeLabels": timeline_labels,
        "timeData": timeline_data,
        "valid": valid_count,
        "invalid": total_validated - valid_count,
    }

    if ramp_result:
        chart_data["rampLabels"] = [f"{n} Users" for n in ramp_result.levels]
        # Extract average latencies
        ramp_lats = []
        for r in ramp_result.results:
            ramp_lats.append(r.tracker.average_latency())
        chart_data["rampData"] = ramp_lats

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Load Test Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f7; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #111; }}
        .badge {{ padding: 8px 16px; border-radius: 20px; font-weight: bold; color: white; display: inline-block; }}
        .badge.pass {{ background-color: #34c759; }}
        .badge.fail {{ background-color: #ff3b30; }}
        .summary-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center; }}
        .card h3 {{ margin: 0 0 10px 0; color: #666; font-size: 14px; text-transform: uppercase; }}
        .card .value {{ font-size: 28px; font-weight: bold; color: #111; }}
        .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        .full-width {{ grid-column: 1 / -1; }}
        table {{ width: 100%; width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background-color: #fafafa; font-weight: bold; color: #555; }}
        tr:last-child td {{ border-bottom: none; }}
        .status-ok {{ color: #34c759; font-weight: bold; }}
        .status-err {{ color: #ff3b30; font-weight: bold; }}
        .header-section {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }}
        .timestamp {{ color: #888; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-section">
            <h1>LLM Inference Test Report</h1>
            <div>
                <span class="badge {'pass' if passed else 'fail'}">
                    {'PASSED' if passed else 'FAILED'}
                </span>
            </div>
        </div>
        
        <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary-cards">
            <div class="card">
                <h3>Total Requests</h3>
                <div class="value">{summary['total_requests']}</div>
            </div>
            <div class="card">
                <h3>Avg Latency</h3>
                <div class="value">{summary['average_latency_ms']}ms</div>
            </div>
            <div class="card">
                <h3>Avg TTFT</h3>
                <div class="value">{summary['average_ttft_ms']}ms</div>
            </div>
            <div class="card">
                <h3>Avg TPOT</h3>
                <div class="value">{summary.get('average_tpot_ms', 0)}ms</div>
            </div>
            <div class="card">
                <h3>Throughput</h3>
                <div class="value">{summary['tokens_per_second']} tok/s</div>
            </div>
            <div class="card">
                <h3>Valid JSON</h3>
                <div class="value">{valid_count}/{total_validated}</div>
            </div>
        </div>

        <div class="charts">
            <div class="chart-container">
                <canvas id="successChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="validChart"></canvas>
            </div>
            <div class="chart-container full-width">
                <canvas id="latencyHist"></canvas>
            </div>
            <div class="chart-container full-width">
                <canvas id="latencyTime"></canvas>
            </div>
        </div>
"""
    if ramp_result:
        html_content += """
        <h2>Ramp-Up Test Results</h2>
        <div class="charts">
            <div class="chart-container full-width">
                <canvas id="rampChart"></canvas>
            </div>
        </div>
        """

    html_content += """
        <h2>Failed Validations</h2>
        """
    
    # Add failed validations table if any
    failed_vals = [v for v in result.validation_results if not v.is_valid]
    if failed_vals:
        html_content += """
        <table>
            <tr><th>#</th><th>Error Detail</th></tr>
        """
        for i, val in enumerate(failed_vals, 1):
            safe_msg = escape(val.error_message or "Unknown error")
            html_content += f"<tr><td>{i}</td><td class='status-err'>{safe_msg}</td></tr>"
        html_content += "</table>"
    else:
        html_content += "<p>No validation failures.</p>"

    # Script injection for Chart.js
    chart_json = json.dumps(chart_data)
    html_content += f"""
        <script>
            const data = {chart_json};
            
            // Success/Fail Pie
            new Chart(document.getElementById('successChart'), {{
                type: 'doughnut',
                data: {{
                    labels: ['Successful', 'Failed'],
                    datasets: [{{
                        data: [data.success, data.failed],
                        backgroundColor: ['#34c759', '#ff3b30']
                    }}]
                }},
                options: {{ plugins: {{ title: {{ display: true, text: 'HTTP Success Rate' }} }} }}
            }});

            // Valid/Invalid Pie
            new Chart(document.getElementById('validChart'), {{
                type: 'doughnut',
                data: {{
                    labels: ['Valid JSON', 'Invalid/Error'],
                    datasets: [{{
                        data: [data.valid, data.invalid],
                        backgroundColor: ['#007aff', '#ff9500']
                    }}]
                }},
                options: {{ plugins: {{ title: {{ display: true, text: 'Structure Pass Rate' }} }} }}
            }});

            // Latency Histogram
            new Chart(document.getElementById('latencyHist'), {{
                type: 'bar',
                data: {{
                    labels: data.histLabels,
                    datasets: [{{
                        label: 'Requests',
                        data: data.histData,
                        backgroundColor: '#5ac8fa'
                    }}]
                }},
                options: {{
                    plugins: {{ title: {{ display: true, text: 'Latency Distribution' }} }},
                    scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }} }}
                }}
            }});

            // Latency Timeline
            new Chart(document.getElementById('latencyTime'), {{
                type: 'line',
                data: {{
                    labels: data.timeLabels,
                    datasets: [{{
                        label: 'Latency (ms)',
                        data: data.timeData,
                        borderColor: '#5856d6',
                        tension: 0.1
                    }}]
                }},
                options: {{
                    plugins: {{ title: {{ display: true, text: 'Latency Timeline' }} }},
                    scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: 'Latency (ms)' }} }} }}
                }}
            }});
"""
    if ramp_result:
        html_content += """
            // Ramp Timeline
            new Chart(document.getElementById('rampChart'), {
                type: 'line',
                data: {
                    labels: data.rampLabels,
                    datasets: [{
                        label: 'Avg Latency (ms)',
                        data: data.rampData,
                        borderColor: '#ff9500',
                        backgroundColor: 'rgba(255, 149, 0, 0.2)',
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    plugins: { title: { display: true, text: 'Ramp-Up Latency Profile' } },
                    scales: { 
                        y: { beginAtZero: true, title: { display: true, text: 'Avg Latency (ms)' } },
                        x: { title: { display: true, text: 'Concurrent Users' } }
                    }
                }
            });
        """

    html_content += """
        </script>
    </div>
</body>
</html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nHtml report successfully generated at {output_path}")


def generate_comparison_report(
    metrics: list[ComparisonMetric],
    output_path: str = "comparison_report.html",
) -> None:
    """Generate a comparative HTML report across multiple scenarios."""
    
    labels = []
    latencies = []
    throughputs = []
    pass_rates = []
    
    for m in metrics:
        name = f"{m.scenario_name} ({m.target})"
        labels.append(name)
        latencies.append(m.avg_latency)
        throughputs.append(m.throughput)
        pass_rates.append(m.pass_rate * 100)
        
    chart_data = {
        "labels": labels,
        "latencies": latencies,
        "throughputs": throughputs,
        "passRates": pass_rates,
    }
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Comparative Test Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f7; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #111; }}
        .charts {{ display: grid; grid-template-columns: 1fr; gap: 40px; margin-bottom: 30px; margin-top: 40px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 20px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background-color: #fafafa; font-weight: bold; color: #555; }}
        tr:last-child td {{ border-bottom: none; }}
        .badge {{ padding: 6px 12px; border-radius: 20px; font-weight: bold; color: white; display: inline-block; font-size: 12px; }}
        .badge.pass {{ background-color: #34c759; }}
        .badge.fail {{ background-color: #ff3b30; }}
        .timestamp {{ color: #888; font-size: 14px; text-align: right; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Comparative Test Report</h1>
        <p class="timestamp">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <table>
            <tr>
                <th>Scenario</th>
                <th>Target Model</th>
                <th>Users</th>
                <th>Avg Latency</th>
                <th>Throughput</th>
                <th>Pass Rate</th>
                <th>Status</th>
            </tr>
    """
    
    for m in metrics:
        safe_scenario_name = escape(m.scenario_name)
        safe_target = escape(m.target)
        status_badge = "<span class='badge pass'>PASSED</span>" if m.passed else "<span class='badge fail'>FAILED</span>"
        html_content += f"""
            <tr>
                <td>{safe_scenario_name}</td>
                <td>{safe_target}</td>
                <td>{m.users}</td>
                <td>{m.avg_latency:.0f}ms</td>
                <td>{m.throughput:.1f} tok/s</td>
                <td>{m.pass_rate*100:.0f}%</td>
                <td>{status_badge}</td>
            </tr>
        """
        
    html_content += f"""
        </table>

        <div class="charts">
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="throughputChart"></canvas>
            </div>
        </div>

        <script>
            const data = {json.dumps(chart_data)};
            
            new Chart(document.getElementById('latencyChart'), {{
                type: 'bar',
                data: {{
                    labels: data.labels,
                    datasets: [{{
                        label: 'Avg Latency (ms) - Lower is Better',
                        data: data.latencies,
                        backgroundColor: '#ff9500'
                    }}]
                }},
                options: {{ plugins: {{ title: {{ display: true, text: 'Comparative Latency' }} }} }}
            }});

            new Chart(document.getElementById('throughputChart'), {{
                type: 'bar',
                data: {{
                    labels: data.labels,
                    datasets: [{{
                        label: 'Throughput (tok/s) - Higher is Better',
                        data: data.throughputs,
                        backgroundColor: '#34c759'
                    }}]
                }},
                options: {{ plugins: {{ title: {{ display: true, text: 'Comparative Throughput' }} }} }}
            }});
        </script>
    </div>
</body>
</html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nComparative HTML report generated at {output_path}")


def generate_niah_html_report(
    niah_result: "NIAHTestResult",
    output_path: str = "niah_report.html",
) -> None:
    """Generate an HTML report for NIAH Context Scaling results."""
    
    labels = []
    latencies = []
    tpot_vals = []
    ttft_vals = []
    
    for level in niah_result.levels:
        labels.append(f"{level.context_length} tok")
        summary = level.load_result.tracker.summary()
        latencies.append(summary["average_latency_ms"])
        tpot_vals.append(summary.get("average_tpot_ms", 0))
        ttft_vals.append(summary["average_ttft_ms"])
        
    chart_data = {
        "labels": labels,
        "latencies": latencies,
        "tpots": tpot_vals,
        "ttfts": ttft_vals,
    }
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM NIAH Scaling Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f7; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #111; }}
        .charts {{ display: grid; grid-template-columns: 1fr; gap: 40px; margin-bottom: 30px; margin-top: 40px; }}
        .chart-container {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Context Scaling (NIAH) Performance</h1>
        
        <div class="charts">
            <div class="chart-container">
                <canvas id="scalingChart"></canvas>
            </div>
        </div>

        <script>
            const data = {json.dumps(chart_data)};
            
            new Chart(document.getElementById('scalingChart'), {{
                type: 'line',
                data: {{
                    labels: data.labels,
                    datasets: [
                        {{
                            label: 'Avg Latency (ms)',
                            data: data.latencies,
                            borderColor: '#ff3b30',
                            tension: 0.1
                        }},
                        {{
                            label: 'Avg TTFT (ms)',
                            data: data.ttfts,
                            borderColor: '#5ac8fa',
                            tension: 0.1
                        }},
                        {{
                            label: 'Avg TPOT (ms)',
                            data: data.tpots,
                            borderColor: '#34c759',
                            tension: 0.1
                        }}
                    ]
                }},
                options: {{ 
                    plugins: {{ title: {{ display: true, text: 'Exponential Metrics vs Context Size' }} }},
                    scales: {{
                        y: {{ beginAtZero: true, title: {{ display: true, text: 'Milliseconds' }} }},
                        x: {{ title: {{ display: true, text: 'Context Range (Tokens)' }} }}
                    }}
                }}
            }});
        </script>
    </div>
</body>
</html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nNIAH HTML report generated at {output_path}")


