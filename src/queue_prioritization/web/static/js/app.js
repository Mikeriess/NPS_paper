// Add this to ensure any dynamically created plots use our styling
function createPlot(element, data, layout) {
    // Add white background to plots
    if (!layout.paper_bgcolor) {
        layout.paper_bgcolor = 'white';
    }
    if (!layout.plot_bgcolor) {
        layout.plot_bgcolor = 'white';
    }
    
    // Ensure plot text is visible
    if (!layout.font) {
        layout.font = { color: 'black' };
    }
    
    Plotly.newPlot(element, data, layout);
} 