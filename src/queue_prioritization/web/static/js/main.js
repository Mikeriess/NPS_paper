document.addEventListener('DOMContentLoaded', function() {
    const experimentForm = document.getElementById('experiment-form');
    const runButton = document.getElementById('run-button');
    const loadingIndicator = document.getElementById('loading');
    const resultsCard = document.getElementById('results-card');
    
    // Load previous experiments
    fetchPreviousExperiments();
    
    experimentForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate form - ensure at least one priority scheme is selected
        const selectedSchemes = document.querySelectorAll('input[name="priority_schemes"]:checked');
        if (selectedSchemes.length === 0) {
            alert('Please select at least one priority scheme');
            return;
        }
        
        // Show loading indicator
        runButton.disabled = true;
        loadingIndicator.classList.remove('hidden');
        
        // Collect form data
        const formData = new FormData(experimentForm);
        const params = {};
        
        // Handle multiple checkbox values
        const priority_schemes = [];
        for (const entry of formData.entries()) {
            if (entry[0] === 'priority_schemes') {
                priority_schemes.push(entry[1]);
            } else {
                params[entry[0]] = entry[1];
            }
        }
        params.priority_schemes = priority_schemes;
        
        // Send request to run experiment
        fetch('/api/run_experiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(params)
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            runButton.disabled = false;
            loadingIndicator.classList.add('hidden');
            
            if (data.status === 'success') {
                // Redirect to the detailed results page
                window.location.href = `/experiment/${data.run_id}`;
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            runButton.disabled = false;
            loadingIndicator.classList.add('hidden');
            alert('Error: ' + error.message);
        });
    });
    
    function fetchPreviousExperiments() {
        fetch('/api/experiments')
        .then(response => response.json())
        .then(experiments => {
            const experimentsList = document.getElementById('experiments-list');
            
            if (experiments.length === 0) {
                experimentsList.innerHTML = '<p>No previous experiments found.</p>';
                return;
            }
            
            // Sort experiments by timestamp (newest first)
            experiments.sort((a, b) => b.timestamp - a.timestamp);
            
            // Display experiments
            experimentsList.innerHTML = experiments.map(exp => {
                const date = new Date(parseInt(exp.timestamp) * 1000);
                return `
                    <div class="experiment-item" data-run-id="${exp.run_id}">
                        <div class="experiment-header">
                            <strong>${exp.run_id}</strong>
                            <span>${date.toLocaleString()}</span>
                        </div>
                        <div class="experiment-details">
                            <div class="detail-item">
                                <span class="detail-label">Priority:</span>
                                <span>${exp.params.priority_scheme}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Agents:</span>
                                <span>${exp.params.number_of_agents}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Ceiling:</span>
                                <span>${exp.params.hard_ceiling}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">Days:</span>
                                <span>${exp.params.days}</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            // Add click event to experiment items
            document.querySelectorAll('.experiment-item').forEach(item => {
                item.addEventListener('click', function() {
                    const runId = this.getAttribute('data-run-id');
                    window.location.href = `/experiment/${runId}`;
                });
            });
        })
        .catch(error => {
            console.error('Error fetching experiments:', error);
        });
    }
}); 