from dash import Dash, html, dcc, Input, Output, State
import dash_cytoscape as cyto
from gradient_sheets import get_tensor_sheet, get_grads, percent_grads
from sheet_graph import build_cytoscape_component, get_labels

import dash_bootstrap_components as dbc

import gspread

gc = gspread.oauth()


spreadsheet_key = "1dGNB0YbWVPcD5-GDD9WDaFj9K3o0qppFLEuOK6e8CoI"

# gc = gspread.oauth()
app = Dash("Gradient Sheets", external_stylesheets=[
    "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/spacelab/bootstrap.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"
])


app.layout = html.Div([
    dcc.Store(id="formula_sheet"),
    html.Div(
        [
            dbc.Row(
                [   
                    # title
                    dbc.Col(
                        [
                            html.H1("Gradient Sheets")
                        ],
                    ),
                ],
                style={
                    "width": "100%",
                    "height": "5hv",
                    # border red for debugging
                    "border": "1px solid red"
                },
            ),
            dbc.Row(
                [
                    # File, View dropdowns
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("Open..."),
                            dbc.DropdownMenuItem("Save"),
                        ],
                        label="File",
                        id="file-dropdown",
                        style={
                            "display": "inline-block",
                        },
                    ),
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("Relative Gradient"),
                            dbc.DropdownMenuItem("Absolute Gradient"),
                        ],
                        label="View",
                        id="view-dropdown",
                        style={
                            "display": "inline-block",
                        },
                    ),
                ],
                style={
                    "width": "100%",
                    "height": "5hv",
                    "border": "1px solid red"
                },
            ),
            dbc.Row(
                # Sync button with icon
                [
                    dbc.Button(
                        [
                            html.I(className="fas fa-sync"), # what do I need to make this work? answer: https://fontawesome.com/how-to-use/on-the-web/setup/getting-started?using=web-fonts-with-css
                            " Sync"
                        ],
                        id="sync-button",
                        color="primary",
                        style={
                            "display": "inline-block",
                        },
                    ),
                ],
                style={
                    "width": "100%",
                    "height": "5hv",
                    "border": "1px solid red"
                },
            ),
            dbc.Row(
                # intline text inputs to modify cell label and formula
                [
                    dbc.Col(
                        [
                            dbc.Input(
                                placeholder="Cell Label",
                                type="text",
                                id="cell-label-input",
                                style={
                                    "width": "100%",
                                },
                            ),
                        ],
                        style={
                            "width": "50%",
                            "height": "5hv",
                            "border": "1px solid red"
                        },
                    ),
                    dbc.Col(
                        [
                            dbc.Input(
                                placeholder="Cell Formula",
                                type="text",
                                id="cell-formula-input",
                                style={
                                    "width": "100%",
                                },
                            ),
                        ],
                        style={
                            "width": "50%",
                            "height": "5hv",
                            "border": "1px solid red"
                        },
                    ),
                ],
                style={
                    "width": "100%",
                    "height": "5hv",
                    "border": "1px solid red"
                },
            ),
            dbc.Row(
                [
                    build_cytoscape_component(gc, spreadsheet_key, id="cytoscape"),
                ],
                style={
                    "width": "100%",
                    "height": "80hv",
                    "border": "1px solid red"
                },
            ),
        ],
        style={
            "width": "100%",
            "height": "100hv",
            # add some space on the left and right of the app
            "margin-left": "1rem",
            "margin-right": "1rem",
            # make background pink for debugging
            "background-color": "#f8f9fa",
        }
    ),
])





@app.callback(
    Output("cytoscape", "stylesheet"),
    Input("cytoscape", "tapNodeData"),
    State("cytoscape", "stylesheet"),
    State("formula_sheet", "data")
)
def update_stylesheet(node_data, default_stylesheet, formula_sheet):
    if node_data is None:
        return default_stylesheet

    tensor_sheet = get_tensor_sheet(formula_sheet)
    clicked_cell_name = node_data['id']
    grads = get_grads(tensor_sheet, clicked_cell_name, leaf_only=False)
    p_grads = percent_grads(grads, tensor_sheet, node_data['id'])

    stylesheet = default_stylesheet
    cell_names = list(p_grads.keys())
    grads = list(p_grads.values())
    labels = get_labels(cell_names, formula_sheet)

    for cell_name, grad, label in zip(cell_names, grads, labels):
        if cell_name == clicked_cell_name:
            continue

        max_grad = max(p_grads.values())
        min_grad = min(p_grads.values())

        if grad > 0:
            grad = (grad - 0) / (max_grad - 0)
            return (0, 235 * grad + 20, 0)
        elif grad < 0:
            grad = -grad
            grad = (grad - 0) / ((-min_grad) - 0)
            return (235 * grad + 20, 0, 0)
        else:
            color = (160, 162, 143)

        stylesheet.append({
            'selector': f'node[id = "{cell_name}"]',
            'style': {
                'background-color': color,
                'content': f'{grad:.2f}\n{label}'
            }
        })
    
    # make clicked node gold
    stylesheet.append({
        'selector': f'node[id = "{clicked_cell_name}"]',
        'style': {
            'background-color': (255, 215, 0)
        }
    })

    return stylesheet



if __name__ == "__main__":
    app.run_server(debug=True, port=8000)