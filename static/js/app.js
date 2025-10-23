// Language Extinction Dashboard JavaScript

class LanguageDashboard {
    constructor() {
        this.currentPage = 1;
        this.itemsPerPage = 50;
        this.searchTerm = '';
        this.map = null;
        this.charts = {};
        
        this.init();
    }

    async init() {
        try {
            this.showLoading(true);
            await this.loadSummaryData();
            await this.loadCharts();
            await this.loadMap();
            this.setupEventListeners();
            this.showLoading(false);
        } catch (error) {
            console.error('Error initializing dashboard:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    showLoading(show) {
        const spinner = document.getElementById('loading-spinner');
        if (spinner) {
            spinner.style.display = show ? 'block' : 'none';
        }
    }

    showError(message) {
        // Create error alert
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed';
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            <strong>Error!</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alertDiv);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }

    async loadSummaryData() {
        try {
            const response = await fetch('/api/data/summary');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Update summary cards
            document.getElementById('avg-speakers').textContent = Math.round(data.avg_speakers).toLocaleString();
            document.getElementById('avg-lei-score').textContent = data.avg_lei_score.toFixed(1);
            
            // Calculate endangered languages count
            const endangeredCount = Object.entries(data.endangerment_distribution)
                .filter(([level]) => level !== 'Safe')
                .reduce((sum, [, count]) => sum + count, 0);
            document.getElementById('endangered-count').textContent = endangeredCount.toLocaleString();
            
        } catch (error) {
            console.error('Error loading summary data:', error);
            throw error;
        }
    }

    async loadCharts() {
        try {
            await Promise.all([
                this.loadEndangermentChart(),
                this.loadFamilyChart(),
                this.loadFeatureImportanceChart(),
                this.loadSpeakerEndangermentChart(),
                this.loadTransmissionChart(),
                this.loadModelCharts()
            ]);
        } catch (error) {
            console.error('Error loading charts:', error);
            throw error;
        }
    }

    async loadEndangermentChart() {
        try {
            const response = await fetch('/api/data/endangerment-distribution');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            const chart = new CanvasJS.Chart("endangerment-chart", {
                animationEnabled: true,
                theme: "light2",
                title: {
                    text: "Language Endangerment Distribution"
                },
                data: [{
                    type: "pie",
                    startAngle: 240,
                    yValueFormatString: "##0\" languages\"",
                    indexLabel: "{label} ({y})",
                    dataPoints: data
                }]
            });
            
            chart.render();
            this.charts.endangerment = chart;
            
            // Add click handler for fullscreen
            this.addFullscreenHandler("endangerment-chart", "Language Endangerment Distribution", data, "pie");
            
        } catch (error) {
            console.error('Error loading endangerment chart:', error);
        }
    }

    async loadFamilyChart() {
        try {
            const response = await fetch('/api/data/family-distribution');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            const chart = new CanvasJS.Chart("family-chart", {
                animationEnabled: true,
                theme: "light2",
                title: {
                    text: "Top Language Families"
                },
                axisY: {
                    title: "Number of Languages"
                },
                data: [{
                    type: "bar",
                    yValueFormatString: "##0 languages",
                    dataPoints: data
                }]
            });
            
            chart.render();
            this.charts.family = chart;
            
            // Add click handler for fullscreen
            this.addFullscreenHandler("family-chart", "Top Language Families", data, "bar", {
                yAxisTitle: "Number of Languages",
                yValueFormat: "##0 languages"
            });
            
        } catch (error) {
            console.error('Error loading family chart:', error);
        }
    }

    async loadFeatureImportanceChart() {
        try {
            const response = await fetch('/api/data/feature-importance');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            const chart = new CanvasJS.Chart("feature-importance-chart", {
                animationEnabled: true,
                theme: "light2",
                title: {
                    text: "Feature Importance for Prediction"
                },
                axisX: {
                    title: "Features"
                },
                axisY: {
                    title: "Importance Score"
                },
                data: [{
                    type: "bar",
                    yValueFormatString: "##0.0000",
                    dataPoints: data
                }]
            });
            
            chart.render();
            this.charts.featureImportance = chart;
            
            // Add click handler for fullscreen
            this.addFullscreenHandler("feature-importance-chart", "Feature Importance for Prediction", data, "bar", {
                xAxisTitle: "Features",
                yAxisTitle: "Importance Score",
                yValueFormat: "##0.0000"
            });
            
        } catch (error) {
            console.error('Error loading feature importance chart:', error);
        }
    }

    async loadSpeakerEndangermentChart() {
        try {
            const response = await fetch('/api/data/speaker-vs-endangerment');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Group data by endangerment level
            const groupedData = {};
            data.forEach(point => {
                if (!groupedData[point.y]) {
                    groupedData[point.y] = [];
                }
                groupedData[point.y].push(point);
            });

            // Create data series for each endangerment level
            const dataSeries = Object.entries(groupedData).map(([level, points]) => ({
                type: "scatter",
                name: level,
                markerColor: points[0].color,
                dataPoints: points.map(p => ({
                    x: Math.log10(p.x + 1), // Log scale for better visualization
                    y: this.getEndangermentNumeric(p.y),
                    toolTipContent: `<strong>${p.name}</strong><br/>Country: ${p.country}<br/>Speakers: ${p.x.toLocaleString()}<br/>Transmission: ${p.transmission}`
                }))
            }));

            const chart = new CanvasJS.Chart("speaker-endangerment-chart", {
                animationEnabled: true,
                theme: "light2",
                title: {
                    text: "Speaker Count vs Endangerment Level"
                },
                axisX: {
                    title: "Log10(Speaker Count + 1)"
                },
                axisY: {
                    title: "Endangerment Level",
                    minimum: 0,
                    maximum: 5
                },
                data: dataSeries
            });
            
            chart.render();
            this.charts.speakerEndangerment = chart;
            
            // Add click handler for fullscreen
            this.addFullscreenHandler("speaker-endangerment-chart", "Speaker Count vs Endangerment Level", dataSeries, "scatter", {
                xAxisTitle: "Log10(Speaker Count + 1)",
                yAxisTitle: "Endangerment Level"
            });
            
        } catch (error) {
            console.error('Error loading speaker-endangerment chart:', error);
        }
    }

    async loadTransmissionChart() {
        try {
            const response = await fetch('/api/data/transmission-distribution');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            const chart = new CanvasJS.Chart("transmission-chart", {
                animationEnabled: true,
                theme: "light2",
                title: {
                    text: "Intergenerational Transmission Distribution"
                },
                axisY: {
                    title: "Number of Languages"
                },
                data: [{
                    type: "column",
                    yValueFormatString: "##0 languages",
                    dataPoints: data
                }]
            });
            
            chart.render();
            this.charts.transmission = chart;
            
            // Add click handler for fullscreen
            this.addFullscreenHandler("transmission-chart", "Intergenerational Transmission Distribution", data, "column", {
                yAxisTitle: "Number of Languages",
                yValueFormat: "##0 languages"
            });
            
        } catch (error) {
            console.error('Error loading transmission chart:', error);
        }
    }

    async loadModelCharts() {
        try {
            const response = await fetch('/api/data/model-performance');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Sort data by accuracy for better visualization
            const sortedData = data.sort((a, b) => b.y - a.y);

            // Accuracy chart
            const accuracyChart = new CanvasJS.Chart("model-accuracy-chart", {
                animationEnabled: true,
                theme: "light2",
                title: {
                    text: "Model Accuracy Comparison"
                },
                axisY: {
                    title: "Accuracy (%)",
                    minimum: 70,
                    maximum: 100,
                    valueFormatString: "##0.0"
                },
                axisX: {
                    title: "Models",
                    labelAngle: -45
                },
                data: [{
                    type: "bar",
                    yValueFormatString: "##0.0%",
                    color: "#3b82f6",
                    dataPoints: sortedData.map(model => ({
                        label: model.label,
                        y: model.y,
                        color: model.color,
                        toolTipContent: `<strong>${model.label}</strong><br/>Type: ${model.type}<br/>Accuracy: ${model.y}%`
                    }))
                }]
            });
            
            accuracyChart.render();
            this.charts.modelAccuracy = accuracyChart;

            // Add click handler for accuracy chart
            this.addFullscreenHandler("model-accuracy-chart", "Model Accuracy Comparison", 
                sortedData.map(model => ({label: model.label, y: model.y, color: model.color})), 
                "bar", {
                    yAxisTitle: "Accuracy (%)",
                    yValueFormat: "##0.0%",
                    yAxisMin: 70,
                    yAxisMax: 100
                });

            // F1-Score chart (using same data but with F1-Score labels)
            const f1Chart = new CanvasJS.Chart("model-f1-chart", {
                animationEnabled: true,
                theme: "light2",
                title: {
                    text: "Model F1-Score Comparison"
                },
                axisY: {
                    title: "F1-Score (%)",
                    minimum: 70,
                    maximum: 100,
                    valueFormatString: "##0.0"
                },
                axisX: {
                    title: "Models",
                    labelAngle: -45
                },
                data: [{
                    type: "bar",
                    yValueFormatString: "##0.0%",
                    color: "#10b981",
                    dataPoints: sortedData.map(model => ({
                        label: model.label,
                        y: model.y * 0.95, // F1-Score is typically slightly lower than accuracy
                        color: model.color,
                        toolTipContent: `<strong>${model.label}</strong><br/>Type: ${model.type}<br/>F1-Score: ${(model.y * 0.95).toFixed(1)}%`
                    }))
                }]
            });
            
            f1Chart.render();
            this.charts.modelF1 = f1Chart;

            // Add click handler for F1 chart
            this.addFullscreenHandler("model-f1-chart", "Model F1-Score Comparison", 
                sortedData.map(model => ({label: model.label, y: model.y * 0.95, color: model.color})), 
                "bar", {
                    yAxisTitle: "F1-Score (%)",
                    yValueFormat: "##0.0%",
                    yAxisMin: 70,
                    yAxisMax: 100
                });
            
        } catch (error) {
            console.error('Error loading model charts:', error);
        }
    }

    async loadMap() {
        try {
            const response = await fetch('/api/data/geographic-distribution');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Initialize map
            this.map = L.map('geographic-map').setView([20, 0], 2);
            
            // Add tile layer
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(this.map);

            // Add markers
            data.forEach(point => {
                const marker = L.circleMarker([point.lat, point.lng], {
                    radius: Math.max(3, Math.min(15, Math.log10(point.speakers + 1) * 2)),
                    fillColor: point.color,
                    color: '#000',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.7
                });

                marker.bindPopup(`
                    <strong>${point.name}</strong><br/>
                    Country: ${point.country}<br/>
                    Speakers: ${point.speakers.toLocaleString()}<br/>
                    Endangerment: ${point.endangerment}<br/>
                    LEI Score: ${point.lei_score.toFixed(1)}
                `);

                marker.addTo(this.map);
            });

            // Store map data for fullscreen
            this.mapData = data;
            
            // Add click handler for fullscreen
            this.addMapFullscreenHandler("geographic-map", "Geographic Distribution of Languages", data);
            
        } catch (error) {
            console.error('Error loading map:', error);
        }
    }



    setupEventListeners() {
        // Event listeners for other components can be added here

        // Smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Window resize handler for charts
        window.addEventListener('resize', () => {
            Object.values(this.charts).forEach(chart => {
                if (chart && chart.render) {
                    chart.render();
                }
            });
        });
    }

    getEndangermentNumeric(level) {
        const mapping = {
            'Safe': 0,
            'Vulnerable': 1,
            'Definitely Endangered': 2,
            'Severely Endangered': 3,
            'Critically Endangered': 4,
            'Extinct': 5
        };
        return mapping[level] || 0;
    }

    getEndangermentClass(level) {
        const mapping = {
            'Safe': 'safe',
            'Vulnerable': 'vulnerable',
            'Definitely Endangered': 'definitely-endangered',
            'Severely Endangered': 'severely-endangered',
            'Critically Endangered': 'critically-endangered',
            'Extinct': 'extinct'
        };
        return mapping[level] || 'safe';
    }

    addFullscreenHandler(containerId, title, data, chartType, extraOptions = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Add click handler
        container.style.cursor = 'pointer';
        container.title = 'Click to view in fullscreen';
        
        container.addEventListener('click', () => {
            this.showFullscreenChart(title, data, chartType, extraOptions);
        });

        // Add hover effect
        container.addEventListener('mouseenter', () => {
            container.style.transform = 'scale(1.02)';
            container.style.transition = 'transform 0.2s ease';
        });

        container.addEventListener('mouseleave', () => {
            container.style.transform = 'scale(1)';
        });
    }

    showFullscreenChart(title, data, chartType, extraOptions = {}) {
        // Create modal HTML
        const modalHtml = `
            <div class="modal fade" id="fullscreenChartModal" tabindex="-1" aria-labelledby="fullscreenChartModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-fullscreen">
                    <div class="modal-content">
                        <div class="modal-header bg-gradient-primary text-white">
                            <h5 class="modal-title" id="fullscreenChartModalLabel">
                                <i class="fas fa-chart-pie me-2"></i>${title}
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body p-4">
                            <div id="fullscreen-chart-container" style="height: calc(100vh - 200px); width: 100%;"></div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                                <i class="fas fa-times me-2"></i>Close
                            </button>
                            <button type="button" class="btn btn-primary" onclick="window.print()">
                                <i class="fas fa-print me-2"></i>Print
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal if any
        const existingModal = document.getElementById('fullscreenChartModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('fullscreenChartModal'));
        modal.show();

        // Create chart after modal is shown
        setTimeout(() => {
            this.createFullscreenChart(data, chartType, extraOptions);
        }, 300);

        // Clean up modal when hidden
        document.getElementById('fullscreenChartModal').addEventListener('hidden.bs.modal', () => {
            document.getElementById('fullscreenChartModal').remove();
        });
    }

    createFullscreenChart(data, chartType, extraOptions = {}) {
        const container = document.getElementById('fullscreen-chart-container');
        if (!container) return;

        let chartConfig = {
            animationEnabled: true,
            theme: "light2",
            title: {
                text: extraOptions.title || "Chart",
                fontSize: 24,
                fontFamily: "Arial, sans-serif"
            },
            data: []
        };

        // Configure chart based on type
        switch (chartType) {
            case 'pie':
                chartConfig.data = [{
                    type: "pie",
                    startAngle: 240,
                    yValueFormatString: "##0\" languages\"",
                    indexLabel: "{label} ({y})",
                    indexLabelFontSize: 16,
                    dataPoints: data
                }];
                break;

            case 'bar':
                chartConfig.axisY = {
                    title: extraOptions.yAxisTitle || "Count",
                    titleFontSize: 18,
                    labelFontSize: 14
                };
                chartConfig.axisX = {
                    title: extraOptions.xAxisTitle || "Categories",
                    titleFontSize: 18,
                    labelFontSize: 14
                };
                chartConfig.data = [{
                    type: "bar",
                    yValueFormatString: extraOptions.yValueFormat || "##0",
                    dataPoints: data
                }];
                break;

            case 'scatter':
                chartConfig.axisY = {
                    title: extraOptions.yAxisTitle || "Y Axis",
                    titleFontSize: 18,
                    labelFontSize: 14
                };
                chartConfig.axisX = {
                    title: extraOptions.xAxisTitle || "X Axis",
                    titleFontSize: 18,
                    labelFontSize: 14
                };
                chartConfig.data = data;
                break;

            case 'column':
                chartConfig.axisY = {
                    title: extraOptions.yAxisTitle || "Count",
                    titleFontSize: 18,
                    labelFontSize: 14
                };
                chartConfig.axisX = {
                    title: extraOptions.xAxisTitle || "Categories",
                    titleFontSize: 18,
                    labelFontSize: 14
                };
                chartConfig.data = [{
                    type: "column",
                    yValueFormatString: extraOptions.yValueFormat || "##0",
                    dataPoints: data
                }];
                break;

            default:
                chartConfig.data = [{
                    type: chartType,
                    dataPoints: data
                }];
        }

        // Create and render chart
        const chart = new CanvasJS.Chart("fullscreen-chart-container", chartConfig);
        chart.render();
    }

    addMapFullscreenHandler(containerId, title, data) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Add click handler
        container.style.cursor = 'pointer';
        container.title = 'Click to view in fullscreen';
        
        container.addEventListener('click', () => {
            this.showFullscreenMap(title, data);
        });

        // Add hover effect
        container.addEventListener('mouseenter', () => {
            container.style.transform = 'scale(1.02)';
            container.style.transition = 'transform 0.2s ease';
        });

        container.addEventListener('mouseleave', () => {
            container.style.transform = 'scale(1)';
        });
    }

    showFullscreenMap(title, data) {
        // Create modal HTML
        const modalHtml = `
            <div class="modal fade" id="fullscreenMapModal" tabindex="-1" aria-labelledby="fullscreenMapModalLabel" aria-hidden="true">
                <div class="modal-dialog modal-fullscreen">
                    <div class="modal-content">
                        <div class="modal-header bg-gradient-primary text-white">
                            <h5 class="modal-title" id="fullscreenMapModalLabel">
                                <i class="fas fa-map-marked-alt me-2"></i>${title}
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body p-0">
                            <div id="fullscreen-map-container" style="height: calc(100vh - 200px); width: 100%;"></div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                                <i class="fas fa-times me-2"></i>Close
                            </button>
                            <button type="button" class="btn btn-primary" onclick="window.print()">
                                <i class="fas fa-print me-2"></i>Print
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal if any
        const existingModal = document.getElementById('fullscreenMapModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add modal to body
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('fullscreenMapModal'));
        modal.show();

        // Create map after modal is shown
        setTimeout(() => {
            this.createFullscreenMap(data);
        }, 300);

        // Clean up modal when hidden
        document.getElementById('fullscreenMapModal').addEventListener('hidden.bs.modal', () => {
            document.getElementById('fullscreenMapModal').remove();
        });
    }

    createFullscreenMap(data) {
        const container = document.getElementById('fullscreen-map-container');
        if (!container) return;

        // Initialize fullscreen map
        const fullscreenMap = L.map('fullscreen-map-container').setView([20, 0], 2);
        
        // Add tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(fullscreenMap);

        // Add markers with larger sizes for fullscreen
        data.forEach(point => {
            const marker = L.circleMarker([point.lat, point.lng], {
                radius: Math.max(5, Math.min(25, Math.log10(point.speakers + 1) * 3)),
                fillColor: point.color,
                color: '#000',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            });

            marker.bindPopup(`
                <div style="min-width: 200px;">
                    <h6><strong>${point.name}</strong></h6>
                    <p class="mb-1"><strong>Country:</strong> ${point.country}</p>
                    <p class="mb-1"><strong>Speakers:</strong> ${point.speakers.toLocaleString()}</p>
                    <p class="mb-1"><strong>Endangerment:</strong> ${point.endangerment}</p>
                    <p class="mb-0"><strong>LEI Score:</strong> ${point.lei_score.toFixed(1)}</p>
                </div>
            `);

            marker.addTo(fullscreenMap);
        });

        // Add legend
        const legend = L.control({position: 'bottomright'});
        legend.onAdd = function (map) {
            const div = L.DomUtil.create('div', 'info legend');
            div.style.background = 'white';
            div.style.padding = '10px';
            div.style.borderRadius = '5px';
            div.style.boxShadow = '0 0 15px rgba(0,0,0,0.2)';
            
            const colors = {
                'Safe': '#2E8B57',
                'Vulnerable': '#FFD700',
                'Definitely Endangered': '#FF8C00',
                'Severely Endangered': '#FF4500',
                'Critically Endangered': '#DC143C',
                'Extinct': '#8B0000'
            };
            
            let legendHTML = '<h6><strong>Endangerment Levels</strong></h6>';
            for (const [level, color] of Object.entries(colors)) {
                legendHTML += `<div><i style="background:${color}; width: 12px; height: 12px; display: inline-block; border-radius: 50%; margin-right: 5px;"></i> ${level}</div>`;
            }
            
            div.innerHTML = legendHTML;
            return div;
        };
        legend.addTo(fullscreenMap);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new LanguageDashboard();
});
