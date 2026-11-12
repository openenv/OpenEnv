# The prompt strings keep the paper's exact whitespace.
# ruff: noqa: W291, W293

"""Prompts for the Spatial-Gym environment.

Copied unchanged from the Spatial-Gym verifiers environment
(https://github.com/lkaesberg/Spatial-Gym/tree/main/environments/spatial_gym), which copies them from the
SPaRC reference code. So an agent gets the text the Spatial-Gym paper used, with the polyshape
and negative polyshape rules stated the way the puzzles were built (as in SPaRC 0.7.0). The move parser is in
models.py, because clients use it too.
"""

import json
from typing import Dict


def _get_polyshapes_str(puzzle_data: Dict) -> str:
    """Extract polyshapes string from puzzle data."""
    polyshapes_str = ""
    if "polyshapes" in puzzle_data and puzzle_data["polyshapes"]:
        polyshapes_str = "POLYSHAPES DEFINITIONS:\n"
        polyshapes_json = json.loads(puzzle_data["polyshapes"])
        for shape_id, shape_def in polyshapes_json.items():
            shape_def_str = "\n".join(map(str, shape_def))
            polyshapes_str += f"Shape {shape_id}:\n{shape_def_str}\n\n"
    return polyshapes_str


def step_prompt(puzzle_data: Dict) -> Dict:
    """
    Generate the complete prompt dict for step-by-step gym mode (no traceback).

    Args:
        puzzle_data: The puzzle data dictionary

    Returns:
        Dict with 'system' and 'user' message content
    """
    polyshapes_str = _get_polyshapes_str(puzzle_data)

    system_content = f"""
    You are an autonomous agent controlling a path‐finding puzzle solver.
    Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    Core Concepts & Grid Basics:
    Grid Dimensions: You can find the puzzle grid size in the info 
    Path: The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    Revisiting: You can not traceback your path. you can not visit a cell twice.
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both x and y are odd. 
    The path goes around these rule cells, never on them. They are also marked as gaps.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.
    Valid Path Cells: The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.


    Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `V`: **Visited Node** (Path has passed through this cell)
    *   `L`: **Current Node** (Path is currently on this cell)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y
        
    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black
    
        
    Detailed Solving Rules:
    The drawn path must satisfy ALL applicable constraints:

    1.  Path Constraints:
        Path connects adjacent nodes (horizontal/vertical moves only).
        Nodes CAN NOT be revisited. You cannot visit a cell twice.
        Path MUST pass through all Dot cells.
        Path CANNOT pass through any Gap cells.

    2.  Region-Based Rules (Apply to areas enclosed by the path):
        Squares: All squares within a single region MUST be the same color. Squares of different colors MUST be separated into different regions by the path.
        Stars: Within a single region, each star symbol MUST be paired with exactly ONE other element of the same color. Other colors within the region are irrelevant to this specific star's rule.
        
        Polyshapes(poly): The region containing this symbol MUST have exactly the specified shape (defined in Polyshape Definitions): the shape fills the whole region, with no cells left over.
        If multiple positive polyshapes are in one region, their shapes together must fill the region exactly, without overlapping. Rotation of polyshapes is NOT allowed. They must fit in their given orientation. The color of a polyshape does not matter for this rule.
        
        Negative Polyshapes(ylop): These subtract area from the positive polyshapes in the same region.
        The positive shapes, minus the cells covered by the negative shapes, must fill the region exactly, so the region's size is the total size of the positive shapes minus the total size of the negative shapes.
        A negative shape does not have to match the shape or color of any positive shape. If the negative shapes cancel the positive shapes completely, the region has no shape constraint.

    3.  Path-Based Rules (Edge Touching):
        Triangles: The path MUST touch a specific number of edges of the cell containing the triangle symbol.
            (1): Path touches EXACTLY 1 edge of the triangle's cell.
            (2): Path touches EXACTLY 2 edges of the triangle's cell.
            (3): Path touches EXACTLY 3 edges of the triangle's cell.
            (4): Path touches EXACTLY 4 edges (fully surrounds) the triangle's cell.

    Polyshape Definitions: Shapes are defined by 2D arrays where 1 indicates an occupied cell and 0 indicates an empty cell. 
    {polyshapes_str}

    At each turn you'll receive the current state:
    - Step: The current step number
    - Current Position: Your current (x, y) location
    - Legal Actions: Available moves with format [digit=DIRECTION, ...]
    - Grid State: The current grid showing your path progress
    """

    return {
        "system": system_content,
        "user": None,  # User content is dynamic (observation/info/reward JSON)
    }


def step_backtrack_prompt(puzzle_data: Dict) -> Dict:
    """
    Generate the complete prompt dict for step-by-step gym mode with traceback.

    Args:
        puzzle_data: The puzzle data dictionary

    Returns:
        Dict with 'system' and 'user' message content
    """
    polyshapes_str = _get_polyshapes_str(puzzle_data)

    system_content = f"""
    You are an autonomous agent controlling a path‐finding puzzle solver.
    Your goal is to find a valid path (a continuous line) from the specified Start Node to the End Node on the provided grid, adhering to all puzzle rules.

    Core Concepts & Grid Basics:
    Grid Dimensions: You can find the puzzle grid size in the info 
    Path: The solution is a single, continuous line connecting adjacent nodes either horizontally or vertically.
    Revisiting: You can traceback your path, but you MUST do so in the same way you came, without crossing over your own path. 
    When tracing back, you can only move to the last cell you occupied, and then continue from there. Also when you trace back, the nodes you no longer use in your path are free to be used again.
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both x and y are odd. 
    The path goes around these rule cells, never on them. They are also marked as gaps.
    Regions: The drawn path divides the grid cells into one or more distinct enclosed areas (regions). Many rules apply based on the contents of these regions.
    Valid Path Cells: The path travels along the grid lines (edges between nodes). It can only occupy positions marked `+` or `.` in the grid layout (these correspond to positions with at least one even coordinate).
    Rule Cells: Cells containing rule symbols (squares, stars, etc.) have coordinates where both `x` and `y` are odd. The path goes *around* these rule cells, never *on* them.


    Symbol Legend (Grid Notation)
    *   `S`: **Start Node** (Path begins here)
    *   `E`: **End Node** (Path ends here)
    *   `V`: **Visited Node** (Path has passed through this cell)
    *   `L`: **Current Node** (Path is currently on this cell)
    *   `+`: Valid cell for the path to occupy
    *   `N`: Empty rule cell (no rule)
    *   `G`: **Gap** (Path **CANNOT** cross this cell)
    *   `.`: **Dot** (Path **MUST** pass through this cell)
    *   `o-X`: **Square** of color X
    *   `*-X`: **Star** of color X
    *   `A-X`: **Triangle** (touch 1 edge)
    *   `B-X`: **Triangle** (touch 2 edges)
    *   `C-X`: **Triangle** (touch 3 edges)
    *   `D-X`: **Triangle** (touch 4 edges)
    *   `P-X-Y`: **Polyshape** (positive) of color X and shape ID Y
    *   `Y-X-Y`: **Negative Polyshape** (ylop) of color X and shape ID Y
        
    **Color Codes:** R=Red, B=Blue, G=Green, Y=Yellow, W=White, O=Orange, P=Purple, K=Black


    Detailed Solving Rules:
    The drawn path must satisfy ALL applicable constraints:

    1.  Path Constraints:
        Path connects adjacent nodes (horizontal/vertical moves only).
        Nodes CAN be revisited. But only if you trace back to the last cell you occupied (and from there again and again ...).
        Otherwise you CANNOT cross your own path.
        Path MUST pass through all Dot cells.
        Path CANNOT pass through any Gap cells.

    2.  Region-Based Rules (Apply to areas enclosed by the path):
        Squares: All squares within a single region MUST be the same color. Squares of different colors MUST be separated into different regions by the path.
        Stars: Within a single region, each star symbol MUST be paired with exactly ONE other element of the same color. Other colors within the region are irrelevant to this specific star's rule.
        
        Polyshapes(poly): The region containing this symbol MUST have exactly the specified shape (defined in Polyshape Definitions): the shape fills the whole region, with no cells left over.
        If multiple positive polyshapes are in one region, their shapes together must fill the region exactly, without overlapping. Rotation of polyshapes is NOT allowed. They must fit in their given orientation. The color of a polyshape does not matter for this rule.
        
        Negative Polyshapes(ylop): These subtract area from the positive polyshapes in the same region.
        The positive shapes, minus the cells covered by the negative shapes, must fill the region exactly, so the region's size is the total size of the positive shapes minus the total size of the negative shapes.
        A negative shape does not have to match the shape or color of any positive shape. If the negative shapes cancel the positive shapes completely, the region has no shape constraint.

    3.  Path-Based Rules (Edge Touching):
        Triangles: The path MUST touch a specific number of edges of the cell containing the triangle symbol.
            (1): Path touches EXACTLY 1 edge of the triangle's cell.
            (2): Path touches EXACTLY 2 edges of the triangle's cell.
            (3): Path touches EXACTLY 3 edges of the triangle's cell.
            (4): Path touches EXACTLY 4 edges (fully surrounds) the triangle's cell.

    Polyshape Definitions: Shapes are defined by 2D arrays where 1 indicates an occupied cell and 0 indicates an empty cell. 
    {polyshapes_str}

    At each turn you'll receive the current state:
    - Step: The current step number
    - Current Position: Your current (x, y) location
    - Legal Actions: Available moves with format [digit=DIRECTION, ...]
    - Grid State: The current grid showing your path progress
    """

    return {
        "system": system_content,
        "user": None,  # User content is dynamic (observation/info/reward JSON)
    }


# Per-step user message (sparc/process_puzzle.py, process_puzzle_step_by_step).
# Only change: the f-string expression {step + 1} became the placeholder {step}.
STEP_USER_TEMPLATE = "Step: {step}\nCurrent Position: {agent_loc}\nLegal Actions: [{legal_str}]\n\nGrid State:\n{obs_str}\n\nYou MAY think step-by-step, but you MUST end your response with:\nFinal: <action>\nWhere <action> is one of 0=RIGHT, 1=UP, 2=LEFT, 3=DOWN (you may also write the direction name, e.g. Final: right)."
