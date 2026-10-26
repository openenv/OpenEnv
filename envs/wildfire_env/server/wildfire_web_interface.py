"""
Custom web interface for Wildfire Environment.

This module provides a wildfire-specific web interface with visual grid display
and wildfire-specific features, without modifying the base web_interface.py.
"""

import json
from typing import Optional

from openenv.core.env_server.types import EnvironmentMetadata

from ..models import WildfireAction


def get_wildfire_web_interface_html(
    metadata: Optional[EnvironmentMetadata] = None,
) -> str:
    """Generate custom HTML for the wildfire environment web interface."""

    # Prepare README markdown and a simple HTML fallback
    instructions_html = ""
    instructions_json = "null"
    if metadata and metadata.readme_content:
        # Fallback lightweight conversion (in case JS markdown library fails)
        instructions_html = _markdown_to_html_simple(metadata.readme_content)
        # Primary: pass raw markdown to the client for proper rendering via marked.js
        instructions_json = json.dumps(metadata.readme_content)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wildfire Environment - Web Interface</title>
    <!-- Markdown rendering libraries -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
    <script>
      // Embed raw README markdown for client-side rendering
      window.__WILDFIRE_README__ = {instructions_json};
    </script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            height: 100vh;
            overflow: hidden;
        }}
        
        .container {{
            display: flex;
            height: 100vh;
        }}
        
        .left-pane {{
            width: 50%;
            background: white;
            border-right: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
        }}
        
        .right-pane {{
            width: 50%;
            background: #fafafa;
            display: flex;
            flex-direction: column;
        }}
        
        .pane-header {{
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
            background: #f8f9fa;
            font-weight: 600;
            font-size: 16px;
        }}
        
        .pane-content {{
            flex: 1;
            padding: 20px;
            overflow-y: auto;
        }}
        
        /* Action Form Styles */
        .action-form {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .form-group {{
            margin-bottom: 15px;
        }}
        
        .form-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #333;
        }}
        
        .form-group select, .form-group input {{
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .form-group select:focus, .form-group input:focus {{
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }}
        
        /* Buttons */
        .btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 10px;
            margin-bottom: 10px;
        }}
        
        .btn:hover {{
            background: #0056b3;
        }}
        
        .btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        
        .btn-secondary {{
            background: #6c757d;
        }}
        
        .btn-secondary:hover {{
            background: #545b62;
        }}
        
        /* Grid Visualization */
        .grid-container {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .grid-display {{
            display: inline-block;
            border: 2px solid #333;
            background: #fff;
            padding: 5px;
            margin: 10px 0;
        }}
        
        .grid {{
            display: grid;
            gap: 1px;
            background: #333;
        }}
        
        .cell {{
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            cursor: pointer;
            position: relative;
        }}
        
        .cell.ash {{ background-color: #2f2f2f; }}
        .cell.fuel {{ background-color: #228b22; }}
        .cell.burning {{ background-color: #ff4500; }}
        .cell.firebreak {{ background-color: #8b4513; }}
        .cell.watered {{ background-color: #4169e1; }}
        
        .cell:hover {{
            opacity: 0.8;
            transform: scale(1.1);
            z-index: 10;
        }}
        
        /* Stats Display */
        .stats-display {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 10px;
        }}
        
        .stat-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #007bff;
        }}
        
        /* Instructions Section */
        .instructions-section {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        
        .instructions-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .instructions-title {{
            font-size: 18px;
            font-weight: 600;
            color: #333;
            margin: 0;
        }}
        
        .instructions-toggle {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 5px 10px;
            cursor: pointer;
            font-size: 12px;
            color: #6c757d;
        }}
        
        .instructions-toggle:hover {{
            background: #e9ecef;
        }}
        
        .instructions-content {{
            display: none;
            max-height: 400px;
            overflow-y: auto;
            border-top: 1px solid #e0e0e0;
            padding-top: 15px;
        }}
        
        .instructions-content.expanded {{
            display: block;
        }}
        
        /* Legend */
        .legend {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .legend-items {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 10px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border: 1px solid #333;
        }}
        
        /* Connection Status */
        .status-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        .status-connected {{
            background: #28a745;
        }}
        
        .status-disconnected {{
            background: #dc3545;
        }}
        
        /* Action Logs */
        .logs-container {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        .log-entry {{
            border-bottom: 1px solid #f0f0f0;
            padding: 10px 0;
        }}
        
        .log-entry:last-child {{
            border-bottom: none;
        }}
        
        .log-timestamp {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .log-action {{
            background: #e3f2fd;
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 5px;
            font-family: monospace;
            font-size: 12px;
        }}
        
        .log-reward {{
            font-weight: 600;
            color: #28a745;
        }}
        
        .log-done {{
            font-weight: 600;
            color: #dc3545;
        }}
        
        /* State Display */
        .state-display {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .state-item {{
            margin-bottom: 8px;
        }}
        
        .state-label {{
            font-weight: 500;
            color: #666;
        }}
        
        .state-value {{
            color: #333;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Left Pane: Action Interface -->
        <div class="left-pane">
            <div class="pane-header">
                <span class="status-indicator status-disconnected" id="connection-status"></span>
                Wildfire Containment Interface
            </div>
            <div class="pane-content">
                <!-- Instructions Section -->
                {_generate_instructions_section(instructions_html, metadata)}
                
                <!-- Action Form -->
                <div class="action-form">
                    <h3>Take Action</h3>
                    <form id="action-form">
                        <div class="form-group">
                            <label for="action">Action Type <span style="color: red;">*</span></label>
                            <select name="action" id="action" required>
                                <option value="">-- Select Action --</option>
                                <option value="water">Water (Extinguish Fire)</option>
                                <option value="break">Break (Create Firebreak)</option>
                                <option value="wait">Wait (Do Nothing)</option>
                            </select>
                            <small style="display: block; margin-top: 5px; color: #666;">
                                Water: Extinguishes fire at target cell<br>
                                Break: Creates firebreak to prevent spread<br>
                                Wait: Fire continues spreading
                            </small>
                        </div>
                        
                        <div class="form-group" id="coordinates-group" style="display: none;">
                            <label for="x">X Coordinate</label>
                            <input type="number" name="x" id="x" min="0" placeholder="Enter X coordinate">
                            
                            <label for="y" style="margin-top: 10px;">Y Coordinate</label>
                            <input type="number" name="y" id="y" min="0" placeholder="Enter Y coordinate">
                            <small style="display: block; margin-top: 5px; color: #666;">
                                Coordinates are required for water and break actions
                            </small>
                        </div>
                        
                        <button type="submit" class="btn" id="step-btn">Execute Action</button>
                    </form>
                </div>
                
                <!-- Control Buttons -->
                <div style="margin-bottom: 20px;">
                    <button class="btn btn-secondary" id="reset-btn">Reset Environment</button>
                    <button class="btn btn-secondary" id="state-btn">Get State</button>
                </div>
                
                <!-- Stats Display -->
                <div class="stats-display">
                    <h3>Environment Stats</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">Step Count</span>
                            <span class="stat-value" id="step-count">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Water Remaining</span>
                            <span class="stat-value" id="water-remaining">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Breaks Remaining</span>
                            <span class="stat-value" id="breaks-remaining">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Burning Cells</span>
                            <span class="stat-value" id="burning-count">0</span>
                        </div>
                    </div>
                </div>
                
                <!-- Current State Display -->
                <div class="state-display">
                    <h3>Current State</h3>
                    <div id="current-state">
                        <div class="state-item">
                            <span class="state-label">Status:</span>
                            <span class="state-value" id="env-status">Not initialized</span>
                        </div>
                        <div class="state-item">
                            <span class="state-label">Episode ID:</span>
                            <span class="state-value" id="episode-id">-</span>
                        </div>
                        <div class="state-item">
                            <span class="state-label">Wind Direction:</span>
                            <span class="state-value" id="wind-dir">-</span>
                        </div>
                        <div class="state-item">
                            <span class="state-label">Humidity:</span>
                            <span class="state-value" id="humidity">-</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Right Pane: Visual Grid and Logs -->
        <div class="right-pane">
            <div class="pane-header">
                Fire Grid Visualization
            </div>
            <div class="pane-content">
                <!-- Legend -->
                <div class="legend">
                    <h3>Legend</h3>
                    <div class="legend-items">
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #2f2f2f;"></div>
                            <span>Ash (Burned)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #228b22;"></div>
                            <span>Fuel (Safe)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #ff4500;"></div>
                            <span>Burning (Fire)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #8b4513;"></div>
                            <span>Firebreak</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background-color: #4169e1;"></div>
                            <span>Watered (Damp)</span>
                        </div>
                    </div>
                </div>
                
                <!-- Grid Visualization -->
                <div class="grid-container">
                    <h3>Fire Grid</h3>
                    <div id="grid-status" style="margin-bottom: 10px; font-size: 12px; color: #666;">
                        Waiting for grid data... (Click "Reset Environment" to initialize)
                    </div>
                    <div class="grid-display">
                        <div id="fire-grid" class="grid">
                            <!-- Grid will be rendered here -->
                        </div>
                    </div>
                    <p style="margin-top: 10px; font-size: 12px; color: #666;">
                        Click on a cell to set coordinates for water/break actions
                    </p>
                </div>
                
                <!-- Action Logs -->
                <div class="logs-container">
                    <h3>Action History</h3>
                    <div id="action-logs">
                        No actions taken yet
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        class WildfireWebInterface {{
            constructor() {{
                this.ws = null;
                this.isConnected = false;
                this.currentGrid = null;
                this.gridWidth = 0;
                this.gridHeight = 0;
                this.isInitialized = false; // Track if environment has been reset
                this.init();
            }}
            
            init() {{
                this.connectWebSocket();
                this.setupEventListeners();
            }}
            
            connectWebSocket() {{
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${{protocol}}//${{window.location.host}}/ws/ui`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {{
                    this.isConnected = true;
                    this.updateConnectionStatus(true);
                    console.log('WebSocket connected');
                    // Trigger initial state fetch
                    this.fetchInitialState();
                }};
                
                this.ws.onmessage = (event) => {{
                    try {{
                        const data = JSON.parse(event.data);
                        console.log('WebSocket message received:', data.type, data);
                        if (data.type === 'state_update') {{
                            console.log('Updating UI from WebSocket state_update:', data.episode_state);
                            this.updateUI(data.episode_state);
                        }}
                    }} catch (error) {{
                        console.error('Error parsing WebSocket message:', error, event.data);
                    }}
                }};
                
                this.ws.onclose = () => {{
                    this.isConnected = false;
                    this.updateConnectionStatus(false);
                    console.log('WebSocket disconnected');
                    setTimeout(() => this.connectWebSocket(), 3000);
                }};
                
                this.ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                }};
            }}
            
            async fetchInitialState() {{
                // Fetch current state on connection to display grid
                try {{
                    // Try to get current observation from state
                    const stateResponse = await fetch('/web/state');
                    const state = await stateResponse.json();
                    
                    // If we have grid data in state, render it
                    if (state.grid && Array.isArray(state.grid) && state.width && state.height) {{
                        console.log('Rendering grid from state');
                        this.renderGrid(state.grid, state.width, state.height);
                        return;
                    }}
                    
                    // If no grid in state, try to get it from the current episode state
                    // The WebSocket will send the current observation shortly
                    console.log('No grid in state, waiting for WebSocket update...');
                }} catch (error) {{
                    console.error('Error fetching initial state:', error);
                }}
            }}
            
            updateActionButtonState() {{
                const stepBtn = document.getElementById('step-btn');
                if (stepBtn) {{
                    if (this.isInitialized) {{
                        stepBtn.disabled = false;
                        stepBtn.textContent = 'Execute Action';
                    }} else {{
                        stepBtn.disabled = true;
                        stepBtn.textContent = 'Reset Environment First';
                    }}
                }}
            }}
            
            setupEventListeners() {{
                // Instructions toggle
                const instructionsToggle = document.getElementById('instructions-toggle');
                const instructionsContent = document.getElementById('instructions-content');
                if (instructionsToggle && instructionsContent) {{
                    instructionsToggle.addEventListener('click', () => {{
                        instructionsContent.classList.toggle('expanded');
                        instructionsToggle.textContent = instructionsContent.classList.contains('expanded') 
                            ? 'Hide Instructions' : 'Show Instructions';
                    }});
                }}
                
                // Initialize action button state
                this.updateActionButtonState();
                
                // Render README markdown into instructions (client-side, with proper markdown support)
                const readmeMarkdown = window.__WILDFIRE_README__;
                const instructionsTarget = document.getElementById('instructions-markdown');
                if (instructionsTarget && readmeMarkdown) {{
                    try {{
                        if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {{
                            const html = DOMPurify.sanitize(marked.parse(readmeMarkdown));
                            instructionsTarget.innerHTML = html;
                        }}
                    }} catch (e) {{
                        console.error('Failed to render README markdown:', e);
                    }}
                }}
                
                // Action type change - show/hide coordinates
                document.getElementById('action').addEventListener('change', (e) => {{
                    const coordsGroup = document.getElementById('coordinates-group');
                    if (e.target.value === 'water' || e.target.value === 'break') {{
                        coordsGroup.style.display = 'block';
                        document.getElementById('x').required = true;
                        document.getElementById('y').required = true;
                    }} else {{
                        coordsGroup.style.display = 'none';
                        document.getElementById('x').required = false;
                        document.getElementById('y').required = false;
                    }}
                }});
                
                // Form submission
                document.getElementById('action-form').addEventListener('submit', (e) => {{
                    e.preventDefault();
                    this.submitAction();
                }});
                
                // Reset button
                document.getElementById('reset-btn').addEventListener('click', () => {{
                    this.resetEnvironment();
                }});
                
                // State button
                document.getElementById('state-btn').addEventListener('click', () => {{
                    this.getState();
                }});
            }}
            
            async submitAction() {{
                // Check if environment has been reset
                if (!this.isInitialized) {{
                    alert('Please click "Reset Environment" first to initialize the environment before taking actions.');
                    return;
                }}
                
                const formData = new FormData(document.getElementById('action-form'));
                const action = {{}};
                
                for (const [key, value] of formData.entries()) {{
                    if (value !== '') {{
                        if (key === 'x' || key === 'y') {{
                            action[key] = parseInt(value);
                        }} else {{
                            action[key] = value;
                        }}
                    }}
                }}
                
                // Remove x/y if action is 'wait'
                if (action.action === 'wait') {{
                    delete action.x;
                    delete action.y;
                }}
                
                try {{
                    const response = await fetch('/web/step', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ action }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    
                    const result = await response.json();
                    console.log('Step result:', result);
                    console.log('Step result observation:', result.observation);
                    console.log('Step result observation grid:', result.observation?.grid);
                    
                    // Update UI immediately from the response (don't wait for WebSocket)
                    if (result.observation) {{
                        const obs = result.observation;
                        // Update grid if available - check both direct grid and nested observation structure
                        let gridData = obs.grid;
                        let gridWidth = obs.width;
                        let gridHeight = obs.height;
                        
                        // If grid is missing but we have dimensions, try to get from nested structure
                        if ((!gridData || gridData.length === 0) && gridWidth && gridHeight) {{
                            console.warn('Grid data is empty in observation, checking nested structure...');
                            // The observation might be nested differently - check the full structure
                            console.log('Full observation structure:', JSON.stringify(obs, null, 2));
                        }}
                        
                        if (gridData && Array.isArray(gridData) && gridData.length > 0 && gridWidth && gridHeight) {{
                            console.log('Updating grid from step response:', gridWidth, 'x', gridHeight, 'cells:', gridData.length);
                            this.renderGrid(gridData, gridWidth, gridHeight);
                        }} else {{
                            console.warn('Grid data not available in step response - will rely on WebSocket update');
                        }}
                        // Update stats
                        if (obs.remaining_water !== undefined) {{
                            document.getElementById('water-remaining').textContent = obs.remaining_water;
                        }}
                        if (obs.remaining_breaks !== undefined) {{
                            document.getElementById('breaks-remaining').textContent = obs.remaining_breaks;
                        }}
                        if (obs.burning_count !== undefined) {{
                            document.getElementById('burning-count').textContent = obs.burning_count;
                        }}
                        if (obs.wind_dir) {{
                            document.getElementById('wind-dir').textContent = obs.wind_dir;
                        }}
                        if (obs.humidity !== undefined) {{
                            document.getElementById('humidity').textContent = obs.humidity.toFixed(2);
                        }}
                        if (obs.step !== undefined) {{
                            document.getElementById('step-count').textContent = obs.step;
                        }}
                        // Status and Episode ID are updated via WebSocket state_update
                    }}
                }} catch (error) {{
                    console.error('Error submitting action:', error);
                    alert('Error submitting action: ' + error.message);
                }}
            }}
            
            async resetEnvironment() {{
                try {{
                    const response = await fetch('/web/reset', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }}
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    
                    const result = await response.json();
                    console.log('Reset result:', result);
                    console.log('Reset observation:', result.observation);
                    
                    // Mark environment as initialized
                    this.isInitialized = true;
                    this.updateActionButtonState();
                    
                    // Render grid immediately after reset and update stats
                    if (result.observation && result.observation.grid) {{
                        const obs = result.observation;
                        console.log('Grid data:', obs.grid);
                        console.log('Grid dimensions:', obs.width, 'x', obs.height);
                        if (obs.grid && Array.isArray(obs.grid) && obs.width && obs.height) {{
                            console.log('Rendering grid from reset...');
                            this.renderGrid(obs.grid, obs.width, obs.height);
                        }} else {{
                            console.warn('Grid data invalid:', {{
                                gridIsArray: Array.isArray(obs.grid),
                                width: obs.width,
                                height: obs.height
                            }});
                        }}
                        // Update stats from observation
                        if (obs.wind_dir) {{
                            document.getElementById('wind-dir').textContent = obs.wind_dir;
                        }}
                        if (obs.humidity !== undefined) {{
                            document.getElementById('humidity').textContent = obs.humidity.toFixed(2);
                        }}
                        if (obs.step !== undefined) {{
                            document.getElementById('step-count').textContent = obs.step;
                        }}
                        if (obs.remaining_water !== undefined) {{
                            document.getElementById('water-remaining').textContent = obs.remaining_water;
                        }}
                        if (obs.remaining_breaks !== undefined) {{
                            document.getElementById('breaks-remaining').textContent = obs.remaining_breaks;
                        }}
                        if (obs.burning_count !== undefined) {{
                            document.getElementById('burning-count').textContent = obs.burning_count;
                        }}
                    }} else {{
                        console.warn('No grid data in reset result:', result);
                    }}
                    // Status and Episode ID will be updated via WebSocket state_update
                    // But set status immediately to show it's reset
                    document.getElementById('env-status').textContent = 'Reset';
                }} catch (error) {{
                    console.error('Error resetting environment:', error);
                    alert('Error resetting environment: ' + error.message);
                }}
            }}
            
            async getState() {{
                try {{
                    const response = await fetch('/web/state');
                    const state = await response.json();
                    console.log('Current state:', state);
                    alert('Current state: ' + JSON.stringify(state, null, 2));
                }} catch (error) {{
                    console.error('Error getting state:', error);
                    alert('Error getting state: ' + error.message);
                }}
            }}
            
            updateConnectionStatus(connected) {{
                const indicator = document.getElementById('connection-status');
                if (connected) {{
                    indicator.className = 'status-indicator status-connected';
                }} else {{
                    indicator.className = 'status-indicator status-disconnected';
                }}
            }}
            
            updateUI(episodeState) {{
                console.log('updateUI called with episodeState:', episodeState);
                // Update state display
                const statusEl = document.getElementById('env-status');
                const episodeIdEl = document.getElementById('episode-id');
                const stepCountEl = document.getElementById('step-count');
                
                if (statusEl) {{
                    statusEl.textContent = episodeState.is_reset ? 'Reset' : 'Running';
                }}
                if (episodeIdEl) {{
                    episodeIdEl.textContent = episodeState.episode_id || '-';
                }}
                if (stepCountEl) {{
                    stepCountEl.textContent = episodeState.step_count.toString();
                }}
                
                // Update observation if available
                if (episodeState.current_observation) {{
                    const obs = episodeState.current_observation;
                    
                    // Update stats
                    document.getElementById('water-remaining').textContent = 
                        obs.remaining_water !== undefined ? obs.remaining_water : '-';
                    document.getElementById('breaks-remaining').textContent = 
                        obs.remaining_breaks !== undefined ? obs.remaining_breaks : '-';
                    document.getElementById('burning-count').textContent = 
                        obs.burning_count !== undefined ? obs.burning_count : '-';
                    document.getElementById('wind-dir').textContent = 
                        obs.wind_dir || '-';
                    document.getElementById('humidity').textContent = 
                        obs.humidity !== undefined ? obs.humidity.toFixed(2) : '-';
                    
                    // Update grid visualization - handle both array and list formats
                    let gridData = obs.grid;
                    let gridWidth = obs.width;
                    let gridHeight = obs.height;
                    
                    console.log('Updating grid from observation:', {{
                        hasGrid: !!gridData,
                        gridType: typeof gridData,
                        isArray: Array.isArray(gridData),
                        width: gridWidth,
                        height: gridHeight
                    }});
                    
                    // Convert grid to array if it's not already
                    if (gridData && !Array.isArray(gridData)) {{
                        if (typeof gridData === 'string') {{
                            try {{
                                gridData = JSON.parse(gridData);
                                console.log('Parsed grid from string');
                            }} catch (e) {{
                                console.error('Error parsing grid data:', e);
                                gridData = null;
                            }}
                        }}
                    }}
                    
                    // Ensure we have valid grid data
                    if (gridData && Array.isArray(gridData) && gridWidth && gridHeight) {{
                        console.log('Rendering grid from WebSocket update:', gridWidth, 'x', gridHeight, 'cells:', gridData.length);
                        this.renderGrid(gridData, gridWidth, gridHeight);
                    }} else {{
                        console.warn('Invalid grid data in WebSocket update:', {{ 
                            grid: gridData, 
                            gridLength: gridData ? (Array.isArray(gridData) ? gridData.length : 'not array') : 'null',
                            width: gridWidth, 
                            height: gridHeight 
                        }});
                    }}
                }}
                
                // Update action logs
                const logsDiv = document.getElementById('action-logs');
                if (!episodeState.action_logs || episodeState.action_logs.length === 0) {{
                    logsDiv.innerHTML = 'No actions taken yet';
                }} else {{
                    console.log('Updating action logs:', episodeState.action_logs.length, 'logs');
                    logsDiv.innerHTML = episodeState.action_logs.map(log => `
                        <div class="log-entry">
                            <div class="log-timestamp">${{log.timestamp}} (Step ${{log.step_count}})</div>
                            <div class="log-action">Action: ${{JSON.stringify(log.action, null, 2)}}</div>
                            <div>
                                <span class="log-reward">Reward: ${{log.reward !== null && log.reward !== undefined ? log.reward.toFixed(2) : 'None'}}</span>
                                ${{log.done ? '<span class="log-done">DONE</span>' : ''}}
                            </div>
                        </div>
                    `).join('');
                }}
            }}
            
            renderGrid(grid, width, height) {{
                this.gridWidth = width;
                this.gridHeight = height;
                this.currentGrid = grid;
                
                const gridContainer = document.getElementById('fire-grid');
                const gridStatus = document.getElementById('grid-status');
                
                if (!gridContainer) {{
                    console.error('Grid container not found!');
                    return;
                }}
                
                // Validate grid dimensions
                if (!width || !height || !grid || !Array.isArray(grid)) {{
                    console.error('Invalid grid parameters:', {{ width, height, grid }});
                    if (gridStatus) {{
                        gridStatus.innerHTML = '<span style="color: red;">Error: Invalid grid data</span>';
                    }}
                    gridContainer.innerHTML = '<p style="color: red;">Error: Invalid grid data</p>';
                    return;
                }}
                
                // Calculate grid size once
                const gridSize = grid.length;
                const expectedSize = width * height;
                
                // Update status
                if (gridStatus) {{
                    gridStatus.innerHTML = `Grid: ${{width}}×${{height}} (${{gridSize}} cells)`;
                }}
                
                // Check if grid size matches expected dimensions
                if (gridSize !== expectedSize) {{
                    console.warn(`Grid size mismatch: expected ${{expectedSize}}, got ${{gridSize}}`);
                }}
                
                gridContainer.style.gridTemplateColumns = `repeat(${{width}}, 20px)`;
                gridContainer.innerHTML = '';
                
                // Grid encoding: 0=ash, 1=fuel, 2=burning, 3=firebreak, 4=watered
                const cellClasses = ['ash', 'fuel', 'burning', 'firebreak', 'watered'];
                const cellLabels = ['Ash', 'Fuel', 'Burning', 'Firebreak', 'Watered'];
                
                console.log(`Rendering grid: ${{width}}x${{height}}, ${{gridSize}} cells`);
                
                let renderedCells = 0;
                for (let y = 0; y < height; y++) {{
                    for (let x = 0; x < width; x++) {{
                        const index = y * width + x;
                        const cellValue = (grid[index] !== undefined && grid[index] !== null) ? grid[index] : 0;
                        const cellClass = cellClasses[cellValue] || 'ash';
                        const cellLabel = cellLabels[cellValue] || 'Unknown';
                        
                        const cell = document.createElement('div');
                        cell.className = `cell ${{cellClass}}`;
                        cell.title = `(${{x}}, ${{y}}): ${{cellLabel}} (value: ${{cellValue}})`;
                        cell.dataset.x = x;
                        cell.dataset.y = y;
                        cell.dataset.value = cellValue;
                        
                        // Click to set coordinates
                        cell.addEventListener('click', () => {{
                            const xInput = document.getElementById('x');
                            const yInput = document.getElementById('y');
                            if (xInput) xInput.value = x;
                            if (yInput) yInput.value = y;
                        }});
                        
                        gridContainer.appendChild(cell);
                        renderedCells++;
                    }}
                }}
                
                console.log(`Grid rendered: ${{width}}x${{height}} = ${{renderedCells}} cells`);
                
                // Verify grid is visible
                if (gridStatus) {{
                    gridStatus.innerHTML = `Grid: ${{width}}×${{height}} (${{renderedCells}} cells rendered) ✅`;
                    gridStatus.style.color = '#28a745';
                }}
            }}
        }}
        
        // Initialize the web interface when the page loads
        document.addEventListener('DOMContentLoaded', () => {{
            new WildfireWebInterface();
        }});
    </script>
</body>
</html>
    """.replace(
        "{_generate_instructions_section(instructions_html, metadata)}",
        _generate_instructions_section(instructions_html, metadata),
    )


def _generate_instructions_section(
    instructions_html: str, metadata: Optional[EnvironmentMetadata]
) -> str:
    """Generate the instructions section."""
    if not instructions_html or not metadata:
        return ""

    return f"""
                <!-- Instructions Section -->
                <div class="instructions-section">
                    <div class="instructions-header">
                        <h3 class="instructions-title">{metadata.name if metadata else "Wildfire Environment"}</h3>
                        <button class="instructions-toggle" id="instructions-toggle">Show Instructions</button>
                    </div>
                    <div class="instructions-content" id="instructions-content">
                        <div class="instructions-readme">
                            <!-- Client-side rendered markdown target -->
                            <div id="instructions-markdown"></div>
                            <!-- Fallback (very simple conversion) -->
                            <noscript>
                                {instructions_html}
                            </noscript>
                        </div>
                    </div>
                </div>
    """


def _markdown_to_html_simple(markdown: str) -> str:
    """Convert basic markdown to HTML."""
    import html
    import re

    # Escape HTML first
    html_content = html.escape(markdown)

    # Convert headers
    html_content = re.sub(
        r"^# (.*?)$", r"<h1>\1</h1>", html_content, flags=re.MULTILINE
    )
    html_content = re.sub(
        r"^## (.*?)$", r"<h2>\1</h2>", html_content, flags=re.MULTILINE
    )
    html_content = re.sub(
        r"^### (.*?)$", r"<h3>\1</h3>", html_content, flags=re.MULTILINE
    )

    # Convert code blocks
    html_content = re.sub(
        r"```(.*?)\n(.*?)\n```",
        r"<pre><code>\2</code></pre>",
        html_content,
        flags=re.DOTALL,
    )
    html_content = re.sub(r"`([^`]+)`", r"<code>\1</code>", html_content)

    # Convert bold and italic
    html_content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_content)
    html_content = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html_content)

    # Convert lists
    html_content = re.sub(
        r"^- (.*?)$", r"<li>\1</li>", html_content, flags=re.MULTILINE
    )
    html_content = re.sub(
        r"(<li>.*</li>)", r"<ul>\1</ul>", html_content, flags=re.DOTALL
    )

    # Convert line breaks
    html_content = html_content.replace("\n", "<br>")

    return html_content
